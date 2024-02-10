import numpy as np
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import Model

import vgg16
import utils
import rpn

class FasterRCNN(tf.keras.Model):
    def __init__(self, num_classes, actclassoutputs, l2 = 0): # TODO : Add level 3 of detector network
        super.__init__()
        
        self._num_classes = num_classes
        self._outputs_convert_to_probability = actclassoutputs
        self._level1_feature_extractor = vgg16.BackBone(l2)
        self._level2_rpn = rpn.RPN(
        max_proposals_pre_nms_train = 12000,
        max_proposals_post_nms_train = 2000,
        max_proposals_pre_nms_infer = 6000,
        max_proposals_post_nms_infer = 300,
          l2 = l2
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
        rpn_scores, rpn_box_deltas, rpn_proposals = self._level2_rpn(
            inputs = [
                input_image,
                feature_map,
                anchor_map
            ],
            training = training
        )

        # TODO: At third level, use detector


        #Losses
        if training:
            rpn_class_loss = self._level2_rpn.cls_loss(y_pred = rpn_scores, gt_rpn_map = gt_rpn_map)
            rpn_reg_loss = self._level2_rpn.reg_loss(y_pred = rpn_scores, gt_rpn_map = gt_rpn_map)
            self.add_loss(rpn_class_loss)
            self.add_loss(rpn_reg_loss)
            self.add_metric(rpn_class_loss, name = "rpn_class_loss")
            self.add_metric(rpn_reg_loss, name = "rpn_reg_loss")
            # TODO: Add detector losses and metrics
        else:
            # During inference, losses don't matter
            rpn_class_loss = float("inf")
            rpn_reg_loss = float("inf")
            detector_reg_loss = float("inf")
            detector_class_loss = float("inf")


    def _predictions_to_scored_bboxes(self,input_image, classes, box_deltas, proposals, score_threshhold):
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

            pred_by_idx[class_idx] = (proposal_boxes, scores)

        # NMS
        result = {}
        for class_idx, (boxes,scores) in pred_by_idx.items():
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

