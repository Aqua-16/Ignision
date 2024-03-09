import numpy as np
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import Model

from . import vgg16
from . import utils
from . import rpn
from . import detector

class FasterRCNN(tf.keras.Model):
    def __init__(self, num_classes, actclassoutputs, l2 = 0,dropout_prob=0):
        super().__init__()
        
        self._num_classes = num_classes
        self._outputs_convert_to_probability = actclassoutputs
        self._level1_feature_extractor = vgg16.BackBone(l2)
        self._level2_rpn = rpn.RPN(
        max_proposals_pre_nms_train = 12000,
        max_proposals_post_nms_train = 2000,
        max_proposals_pre_nms_pred = 6000,
        max_proposals_post_nms_pred = 300,
        l2 = l2
        )
        self._level3_detector_network=detector.DN(
            n_of_classes=num_classes,
            actclassoutputs=actclassoutputs,
            l2=l2,
            dropout_prob=dropout_prob
        )

    def call(self,inputs,training = False):
        input_image = inputs[0]
        anchor_map = inputs[1]
        if training:
            gt_rpn_map = inputs[2]
            gt_box_class_idx_map = inputs[3]
            gt_box_corner_map = inputs[4]

        # At first level, extract the features
        feature_map = self._level1_feature_extractor(input_image = input_image)

        # At second level, use region proposal network to find noteworthy regions
        rpn_scores, rpn_box_deltas, proposals = self._level2_rpn(
            inputs = [
                input_image,
                feature_map,
                anchor_map
            ],
            training = training
        )
        if training:
            proposals, gt_classes, gt_box_deltas = self.label_proposals(
                proposals = proposals,
                gt_box_class_idxs = gt_box_class_idx_map[0],
                gt_box_corners = gt_box_corner_map[0],
                min_bg_iou_threshold = 0.0,
                min_obj_iou_threshold = 0.5
            )
            proposals, gt_classes, gt_box_deltas = self.sample_proposals(
                proposals = proposals,
                gt_classes = gt_classes,
                gt_box_deltas = gt_box_deltas,
                max_proposals = 128,
                positive_fraction = 0.25
            )
            gt_classes = tf.expand_dims(gt_classes, axis = 0)   
            gt_box_deltas = tf.expand_dims(gt_box_deltas, axis = 0)   
            proposals = tf.stop_gradient(proposals)
            gt_classes = tf.stop_gradient(gt_classes)
            gt_box_deltas = tf.stop_gradient(gt_box_deltas)
            
        # At third level, use detector
        d_classes,d_box_deltas=self._level3_detector_network(
            inp=[
                input_image,
                feature_map,
                proposals
            ],
            train=training
        )       

        #Losses
        if training:
            rpn_class_loss = self._level2_rpn.cls_loss(y_pred = rpn_scores, gt_rpn_map = gt_rpn_map)
            rpn_reg_loss = self._level2_rpn.reg_loss(y_pred = rpn_box_deltas, gt_rpn_map = gt_rpn_map)
            d_class_loss = self._stage3_detector_network.class_loss(y_predicted = d_classes, y_true = gt_classes, from_logits = not self._outputs_convert_to_probability)
            d_reg_loss = self._stage3_detector_network.regression_loss(y_predicted = d_box_deltas, y_true = gt_box_deltas)
            self.add_loss(rpn_class_loss)
            self.add_loss(rpn_reg_loss)
            self.add_loss(d_class_loss)
            self.add_loss(d_reg_loss)
            self.add_metric(rpn_class_loss, name = "rpn_class_loss")
            self.add_metric(rpn_reg_loss, name = "rpn_reg_loss")
            self.add_metric(d_class_loss, name = "detector_class_loss")
            self.add_metric(d_reg_loss, name = "detector_regression_loss")
        else:
            # During inference, losses don't matter
            rpn_class_loss = float("inf")
            rpn_reg_loss = float("inf")
            d_reg_loss = float("inf")
            d_class_loss = float("inf")

        return [
            rpn_scores,
            rpn_box_deltas,
            d_classes,
            d_box_deltas,
            proposals,
            rpn_class_loss,
            rpn_reg_loss,
            d_class_loss,
            d_reg_loss
        ]

    def load_imgnet_wts(self):
    # Load weights from Keras VGG-16 model pre-trained on ImageNet
        keras_model = tf.keras.applications.VGG16(weights = "imagenet")
        vgg16 = self._level1_feature_extractor.layers + self._level3_detector_network.layers
        for model_layer in keras_model.layers:
            w = model_layer.get_weights()
            if(len(w)>0):
                new_layer = None
                for layer in vgg16:
                    if(layer.name == model_layer.name):
                        new_layer = layer
                        break
                if(new_layer):
                    print("Loading VGG-16 ImageNet weights into layer: %s" % new_layer.name)
                    new_layer.set_weights(w)
                
    def predict_on_batch(self,x,threshold):
        _,_,detector_cls,detector_box_deltas,proposals,_,_,_,_ = super().predict_on_batch(x=x)
        scored_bboxes = self.predictions_to_scored_bboxes(self,
                                                        input_image = x[0], 
                                                        classes = detector_cls, 
                                                        box_deltas = detector_box_deltas, 
                                                        proposals = proposals, 
                                                        score_threshhold = threshold)
        return scored_bboxes

    

    def predictions_to_scored_bboxes(self,input_image, classes, box_deltas, proposals, score_threshhold):
        # The purpose of this function is to use the predictions made by the network to create bounding boxes with predicted scores
        
        # Since batch processing will be allowed (TODO), we eliminate the batch dimension first

        input_image = np.squeeze(input_image, axis = 0)
        classes = np.squeeze(classes, axis = 0)
        box_deltas = np.squeeze(box_deltas, axis = 0)

        if self._outputs_convert_to_probability:
            classes = tf.nn.softmax(classes,axis = 1).numpy()

        prop_anchors = np.empty(proposals.shape)
        prop_anchors[:,0] = 0.5 * (proposals[:,2] + proposals[:,0]) # Center - y
        prop_anchors[:,1] = 0.5 * (proposals[:,3] + proposals[:,1]) # Center - x
        prop_anchors[:,2:4] = proposals[:,2:4] - proposals[:,0:2] # Height, Width

        preds_by_idx = {}
        for class_idx in range (1,classes.shape[1]): # Skipping background class as that is not really required
            box_delta_idx = (class_idx - 1) * 4
            box_delta_params = box_deltas[:, box_delta_idx+0 : box_delta_idx +4]
            proposal_boxes = utils.deltas_to_bboxes(
                deltas = box_delta_params,
                means = [0.0,0.0,0.0,0.0],
                stds = [0.1,0.1,0.2,0.2],
                anchors = prop_anchors
            ).numpy()

            # Clipping proposed boxes to image boundaries
            proposal_boxes[:,0::2] = np.clip(proposal_boxes[:,0::2], 0, input_image.shape[0] - 1)
            proposal_boxes[:,1::2] = np.clip(proposal_boxes[:,1::2], 0, input_image.shape[1] - 1)

            scores = classes[:,class_idx]
            required_scores = np.where(scores > score_threshhold)[0]
            proposal_boxes = proposal_boxes[required_scores]
            scores = scores[required_scores]

            preds_by_idx[class_idx] = (proposal_boxes, scores)

        # NMS
        result = {}
        for class_idx, (boxes,scores) in preds_by_idx.items():
            indexes = tf.image.non_max_suppression(
                boxes = boxes,
                scores = scores,
                max_output_size = proposals.shape[0],
                iou_threshold = 0.3
            ).numpy()
            boxes = boxes[indexes]
            scores = scores[indexes]
            scores = np.expand_dims(scores, axis = 0) # This is done for the transpose operation performed next
            scored_boxes = np.hstack([boxes,scores.T])
            result[class_idx] = scored_boxes

        return result

    def label_proposals(self, proposals, gt_box_class_idxs, gt_box_corners, min_bg_iou_threshold, min_obj_iou_threshold):
        proposals = tf.concat([ proposals, gt_box_corners ], axis = 0)#creating fake proposals so that there are always some positive examples during training
        total_ious = utils.iou(bbox1=proposals, bbox2=gt_box_corners)#N,M #computes the IoU b/w each proposal and each ground truth box, resulting in an IoU matrix.
        best_ious = tf.math.reduce_max(total_ious, axis = 1) # (N,) of max IoUs for each of the N proposals
        indexes=tf.math.argmax(total_ious,axis=1) #(N,) indexes of best proposal
        gt_box_class_idxs=tf.gather(gt_box_class_idxs, indices = indexes)   
        gt_box_corners = tf.gather(gt_box_corners, indices = indexes)#(N,4) highest IoU box for each proposal
        idxs = tf.where(best_ious >= min_bg_iou_threshold)# proposals with  sufficiently high IoU
        proposals = tf.gather_nd(proposals, indices = idxs)
        best_ious = tf.gather_nd(best_ious, indices = idxs)
        gt_box_class_idxs = tf.gather_nd(gt_box_class_idxs, indices = idxs)
        gt_box_corners = tf.gather_nd(gt_box_corners, indices = idxs)

        retain_mask = tf.cast(best_ious >= min_obj_iou_threshold, dtype = gt_box_class_idxs.dtype)# if condn true then 1 else 0
        gt_box_class_idxs = gt_box_class_idxs * retain_mask #if retain_mask=0 then it effectively labels proposals as background. 

        num_classes = self._num_classes
        gt_classes = tf.one_hot(indices = gt_box_class_idxs, depth = num_classes)

        #calculate centres and side lengths for proposals and ground truth boxes
        proposal_centers = 0.5 * (proposals[:,0:2] + proposals[:,2:4])          
        proposal_sides = proposals[:,2:4] - proposals[:,0:2]                    
        gt_box_centers = 0.5 * (gt_box_corners[:,0:2] + gt_box_corners[:,2:4])  
        gt_box_sides = gt_box_corners[:,2:4] - gt_box_corners[:,0:2]      

        detector_box_delta_means = tf.constant([0, 0, 0, 0], dtype = tf.float32)
        detector_box_delta_stds = tf.constant([0.1, 0.1, 0.2, 0.2], dtype = tf.float32)
        tyx = (gt_box_centers - proposal_centers) / proposal_sides # # ty = (gt_center_y - proposal_center_y) / proposal_height, tx = (gt_center_x - proposal_center_x) / proposal_width
        thw = tf.math.log(gt_box_sides / proposal_sides) # th = log(gt_height / proposal_height), tw = (gt_width / proposal_width)
        box_delta_targets = tf.concat([ tyx, thw ], axis = 1) #      
        box_delta_targets = (box_delta_targets - detector_box_delta_means) / detector_box_delta_stds

        gt_box_deltas_mask = tf.repeat(gt_classes, repeats = 4, axis = 1)[:,4:]          
        gt_box_deltas_values = tf.tile(box_delta_targets, multiples = [1, num_classes - 1]) 
        gt_box_deltas_mask = tf.expand_dims(gt_box_deltas_mask, axis = 0)     
        gt_box_deltas_values = tf.expand_dims(gt_box_deltas_values, axis = 0) 
        gt_box_deltas = tf.concat([ gt_box_deltas_mask, gt_box_deltas_values ], axis = 0) 
        gt_box_deltas = tf.transpose(gt_box_deltas, perm = [ 1, 0, 2])       

        return proposals, gt_classes, gt_box_deltas

    def sample_proposals(self, proposals, gt_classes, gt_box_deltas, max_proposals, positive_fraction):
        if max_proposals<=0:
            return proposals,gt_classes,gt_box_deltas
        c_indices = tf.argmax(gt_classes, axis=1)#the class index with the highest score for each ground truth box
        p_indices = tf.squeeze(tf.where(c_indices > 0), axis = 1)#identifes positve non-background class
        n_indices = tf.squeeze(tf.where(c_indices <= 0), axis = 1)#)#identifes negative background class
        
        num_p_proposals = tf.size(p_indices)
        num_n_proposals = tf.size(n_indices)        
        
        num_samples = tf.minimum(max_proposals, tf.size(c_indices))#determines no.of proposals to be considered
        num_p_samples = tf.minimum(tf.cast(tf.math.round(tf.cast(num_samples, dtype = float) * positive_fraction), dtype = num_samples.dtype), num_p_proposals)
        num_n_samples = tf.minimum(num_samples - num_p_samples, num_n_proposals)
        #randomly shuffle the positive and negative indices and select the required number of samples for each
        p_sample_indices = tf.random.shuffle(p_indices)[:num_p_samples]
        n_sample_indices = tf.random.shuffle(n_indices)[:num_n_samples]
        
        indices = tf.concat([p_sample_indices, n_sample_indices], axis=0)

        return tf.gather(proposals, indices = indices), tf.gather(gt_classes, indices = indices), tf.gather(gt_box_deltas, indices = indices)



            

