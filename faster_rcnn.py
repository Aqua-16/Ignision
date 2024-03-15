# DONE

import numpy as np
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import Model
from tensorflow.keras.layers import Lambda

from . import vgg16
from . import rpn
from . import detector
from . import utils


class FasterRCNN(tf.keras.Model):
  def __init__(self, num_classes, actclassoutputs, l2 = 0,dropout_prob=0):
    super().__init__()
    self._num_classes = num_classes
    self._outputs_convert_to_probability = actclassoutputs
    self._level1_feature_extractor = vgg16.BackBone(l2=l2)
    self._level2_region_proposal_network = rpn.RPN(
      max_proposals_pre_nms_train = 12000,
      max_proposals_post_nms_train = 2000,
      max_proposals_pre_nms_pred = 6000,
      max_proposals_post_nms_pred = 300,
      l2 = l2,
    )
    self._level3_detector_network = detector.DN(
      n_of_classes = num_classes,
      actclassoutputs = actclassoutputs,
      l2 = l2,
      dropout_prob = dropout_prob
    )

  def call(self, inputs, training = False):

    input_image = inputs[0]             
    anchor_map = inputs[1]             
    anchor_valid_map = inputs[2]        
    if training:
      gt_rpn_map = inputs[3]            
      gt_box_class_idxs_map = inputs[4] 
      gt_box_corners_map = inputs[5]    

    # level 1: Extract features
    feature_map = self._level1_feature_extractor(input_image = input_image)

    # level 2: Generate object proposals using RPN
    rpn_scores, rpn_box_deltas, proposals = self._level2_region_proposal_network(
      inputs = [
        input_image,
        feature_map,
        anchor_map,
        anchor_valid_map
      ],
      training = training
    )

    if training:
      proposals, gt_classes, gt_box_deltas = self._label_proposals(
        proposals = proposals,
        gt_box_class_idxs = gt_box_class_idxs_map[0], 
        gt_box_corners = gt_box_corners_map[0],
        min_bg_iou_threshold = 0.0,
        min_obj_iou_threshold = 0.5
      )
      proposals, gt_classes, gt_box_deltas = self._sample_proposals(
        proposals = proposals,
        gt_classes = gt_classes,
        gt_box_deltas = gt_box_deltas,
        max_proposals = 128,
        positive_fraction = 0.25
      )
      gt_classes = tf.expand_dims(gt_classes, axis = 0)           
      gt_box_deltas = tf.expand_dims(gt_box_deltas, axis = 0)  

      # Proposals are to be treated as constants during training
      proposals = tf.stop_gradient(proposals)
      gt_classes = tf.stop_gradient(gt_classes)
      gt_box_deltas = tf.stop_gradient(gt_box_deltas)

    # level 3: Detector
    detector_classes, detector_box_deltas = self._level3_detector_network(
      inp = [
        input_image,
        feature_map,
        proposals
      ],
      train = training
    )

    # Losses
    if training:
      rpn_class_loss = self._level2_region_proposal_network.cls_loss(y_pred = rpn_scores, gt_rpn_map = gt_rpn_map)
      rpn_regression_loss = self._level2_region_proposal_network.reg_loss(y_pred = rpn_box_deltas, gt_rpn_map = gt_rpn_map)
      detector_class_loss = self._level3_detector_network.cls_loss(y_pred = detector_classes, y_true = gt_classes, f_logits = not self._outputs_convert_to_probability)
      detector_regression_loss = self._level3_detector_network.reg_loss(y_pred = detector_box_deltas, y_true = gt_box_deltas)
      self.add_loss(rpn_class_loss)
      self.add_loss(rpn_regression_loss)
      self.add_loss(detector_class_loss)
      self.add_loss(detector_regression_loss)
      self.add_metric(rpn_class_loss, name = "rpn_class_loss")
      self.add_metric(rpn_regression_loss, name = "rpn_regression_loss")
      self.add_metric(detector_class_loss, name = "detector_class_loss")
      self.add_metric(detector_regression_loss, name = "detector_regression_loss")
    else:
      # Losses cannot be computed during prediction and are thus ignored
      rpn_class_loss = float("inf")
      rpn_regression_loss = float("inf")
      detector_class_loss = float("inf")
      detector_regression_loss = float("inf")

    return [
      rpn_scores,
      rpn_box_deltas,
      detector_classes,
      detector_box_deltas,
      proposals,
      rpn_class_loss,
      rpn_regression_loss,
      detector_class_loss,
      detector_regression_loss
   ]

  def predict_on_batch(self, x, threshold):
   
    _, _, detector_classes, detector_box_deltas, proposals, _, _, _, _ = super().predict_on_batch(x = x)
    scored_boxes_by_class_index = self._predictions_to_scored_boxes(
      input_image = x[0],
      classes = detector_classes,
      box_deltas = detector_box_deltas,
      proposals = proposals,
      score_threshold = threshold
    )
    return scored_boxes_by_class_index

  def load_imagenet_weights(self):
    keras_model = tf.keras.applications.VGG16(weights = "imagenet")
    for keras_layer in keras_model.layers:
      weights = keras_layer.get_weights()
      if len(weights) > 0:
        vgg16_layers = self._level1_feature_extractor.layers + self._level3_detector_network.layers
        our_layer = [ layer for layer in vgg16_layers if layer.name == keras_layer.name ]
        if len(our_layer) > 0:
          print("Loading VGG-16 ImageNet weights into layer: %s" % our_layer[0].name)
          our_layer[0].set_weights(weights)

  def _predictions_to_scored_boxes(self, input_image, classes, box_deltas, proposals, score_threshold):
    # The purpose of this function is to use the predictions made by the network to create bounding boxes with predicted scores
        
    input_image = np.squeeze(input_image, axis = 0)
    classes = np.squeeze(classes, axis = 0)
    box_deltas = np.squeeze(box_deltas, axis = 0)

    # Convert logits to probability
    if not self._outputs_convert_to_probability:
      classes = tf.nn.softmax(classes, axis = 1).numpy()

    proposal_anchors = np.empty(proposals.shape)
    proposal_anchors[:,0] = 0.5 * (proposals[:,0] + proposals[:,2]) # center_y
    proposal_anchors[:,1] = 0.5 * (proposals[:,1] + proposals[:,3]) # center_x
    proposal_anchors[:,2:4] = proposals[:,2:4] - proposals[:,0:2]   # height, width

    # Currently only the fire class is being detected. The reason for doing this in a class agnostic way is to make it easier to integrate new classes, such as smoke, or fuel, etc.
    boxes_and_scores_by_class_idx = {}
    for class_idx in range(1, classes.shape[1]):  
      box_delta_idx = (class_idx - 1) * 4
      box_delta_params = box_deltas[:, (box_delta_idx + 0) : (box_delta_idx + 4)] # (N, 4)
      proposal_boxes_this_class = utils.deltas_to_bboxes_numpy(
        deltas = box_delta_params,
        anchors = proposal_anchors,
        means = [0.0, 0.0, 0.0, 0.0],
        stds = [0.1, 0.1, 0.2, 0.2]
      )

      # Clip to image boundaries
      proposal_boxes_this_class[:,0::2] = np.clip(proposal_boxes_this_class[:,0::2], 0, input_image.shape[0] - 1) # clip y1 and y2 to [0,height)
      proposal_boxes_this_class[:,1::2] = np.clip(proposal_boxes_this_class[:,1::2], 0, input_image.shape[1] - 1) # clip x1 and x2 to [0,width)
      
      scores_this_class = classes[:,class_idx]
      sufficiently_scoring_idxs = np.where(scores_this_class > score_threshold)[0]
      proposal_boxes_this_class = proposal_boxes_this_class[sufficiently_scoring_idxs]
      scores_this_class = scores_this_class[sufficiently_scoring_idxs]
      boxes_and_scores_by_class_idx[class_idx] = (proposal_boxes_this_class, scores_this_class)

    # Perform NMS per class
    scored_boxes_by_class_idx = {}
    for class_idx, (boxes, scores) in boxes_and_scores_by_class_idx.items():
      idxs = tf.image.non_max_suppression(
        boxes = boxes,
        scores = scores,
        max_output_size = proposals.shape[0],
        iou_threshold = 0.3
      )
      idxs = idxs.numpy()
      boxes = boxes[idxs]
      scores = np.expand_dims(scores[idxs], axis = 0) 
      scored_boxes = np.hstack([ boxes, scores.T ])
      scored_boxes_by_class_idx[class_idx] = scored_boxes

    return scored_boxes_by_class_idx

  def _label_proposals(self, proposals, gt_box_class_idxs, gt_box_corners, min_bg_iou_threshold, min_obj_iou_threshold):
   
    # This creates fake proposals that match the ground truth boxes exactly. It will ensure that there are always some positive examples to train on.
    proposals = tf.concat([ proposals, gt_box_corners ], axis = 0)

    ious = utils.iou(bbox1 = proposals, bbox2 = gt_box_corners)

    best_ious = tf.math.reduce_max(ious, axis = 1)  
    box_idxs = tf.math.argmax(ious, axis = 1)     
    gt_box_class_idxs = tf.gather(gt_box_class_idxs, indices = box_idxs) 
    gt_box_corners = tf.gather(gt_box_corners, indices = box_idxs) 
      
    idxs = tf.where(best_ious >= min_bg_iou_threshold) 
    proposals = tf.gather_nd(proposals, indices = idxs)
    best_ious = tf.gather_nd(best_ious, indices = idxs)
    gt_box_class_idxs = tf.gather_nd(gt_box_class_idxs, indices = idxs)
    gt_box_corners = tf.gather_nd(gt_box_corners, indices = idxs)

    retain_mask = tf.cast(best_ious >= min_obj_iou_threshold, dtype = gt_box_class_idxs.dtype) # (N,), with 0 wherever best_iou < threshold, else 1
    gt_box_class_idxs = gt_box_class_idxs * retain_mask #if retain_mask=0 then it effectively labels proposals as background. 

    num_classes = self._num_classes
    gt_classes = tf.one_hot(indices = gt_box_class_idxs, depth = num_classes) # (N,num_classes)

    proposal_centers = 0.5 * (proposals[:,0:2] + proposals[:,2:4])         
    proposal_sides = proposals[:,2:4] - proposals[:,0:2]                    
    gt_box_centers = 0.5 * (gt_box_corners[:,0:2] + gt_box_corners[:,2:4])  
    gt_box_sides = gt_box_corners[:,2:4] - gt_box_corners[:,0:2]            

    detector_box_delta_means = tf.constant([0, 0, 0, 0], dtype = tf.float32)
    detector_box_delta_stds = tf.constant([0.1, 0.1, 0.2, 0.2], dtype = tf.float32)
    tyx = (gt_box_centers - proposal_centers) / proposal_sides  
    thw = tf.math.log(gt_box_sides / proposal_sides)            
    box_delta_targets = tf.concat([ tyx, thw ], axis = 1)      
    box_delta_targets = (box_delta_targets - detector_box_delta_means) / detector_box_delta_stds 
      
    
    gt_box_deltas_mask = tf.repeat(gt_classes, repeats = 4, axis = 1)[:,4:]            
    gt_box_deltas_values = tf.tile(box_delta_targets, multiples = [1, num_classes - 1]) 
    gt_box_deltas_mask = tf.expand_dims(gt_box_deltas_mask, axis = 0)    
    gt_box_deltas_values = tf.expand_dims(gt_box_deltas_values, axis = 0) 
    gt_box_deltas = tf.concat([ gt_box_deltas_mask, gt_box_deltas_values ], axis = 0) 
    gt_box_deltas = tf.transpose(gt_box_deltas, perm = [ 1, 0, 2])        

    return proposals, gt_classes, gt_box_deltas

  def _sample_proposals(self, proposals, gt_classes, gt_box_deltas, max_proposals, positive_fraction):
    if max_proposals <= 0:
      return proposals, gt_classes, gt_box_deltas

   
    class_indices = tf.argmax(gt_classes, axis = 1)  #the class index with the highest score for each ground truth box
    positive_indices = tf.squeeze(tf.where(class_indices > 0), axis = 1)  #identifes positve non-background class
    negative_indices = tf.squeeze(tf.where(class_indices <= 0), axis = 1) #identifes negative background class
    num_positive_proposals = tf.size(positive_indices)
    num_negative_proposals = tf.size(negative_indices)

    num_samples = tf.minimum(max_proposals, tf.size(class_indices)) #determines no.of proposals to be considered
    num_positive_samples = tf.minimum(tf.cast(tf.math.round(tf.cast(num_samples, dtype = float) * positive_fraction), dtype = num_samples.dtype), num_positive_proposals)
    num_negative_samples = tf.minimum(num_samples - num_positive_samples, num_negative_proposals)
    #randomly shuffle the positive and negative indices and select the required number of samples for each
    positive_sample_indices = tf.random.shuffle(positive_indices)[:num_positive_samples]
    negative_sample_indices = tf.random.shuffle(negative_indices)[:num_negative_samples]
    indices = tf.concat([ positive_sample_indices, negative_sample_indices ], axis = 0)

    return tf.gather(proposals, indices = indices), tf.gather(gt_classes, indices = indices), tf.gather(gt_box_deltas, indices = indices)