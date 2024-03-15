#DONE

import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import models
from tensorflow.keras.layers import Conv2D
from tensorflow.keras import backend as K

from . import utils

class RPN(tf.keras.Model):
    def __init__(self, max_proposals_pre_nms_train, max_proposals_post_nms_train, max_proposals_pre_nms_pred, max_proposals_post_nms_pred, l2 = 0):
        super().__init__()
        
        self._max_proposals_pre_nms_train = max_proposals_pre_nms_train
        self._max_proposals_post_nms_train = max_proposals_post_nms_train
        self._max_proposals_pre_nms_pred = max_proposals_pre_nms_pred
        self._max_proposals_post_nms_pred = max_proposals_post_nms_pred
        
        regularizer = tf.keras.regularizers.l2(l2)
        initial_weights = tf.keras.initializers.RandomNormal(mean = 0.0, stddev = 0.01, seed = None)
        anchor_num = 9
        
        self._rpn_conv1 = Conv2D(name = "rpn_conv1", kernel_size = (3,3), strides = 1, filters = 512, padding = "same", activation = "relu", kernel_initializer = initial_weights, kernel_regularizer = regularizer)
        self._rpn_cls = Conv2D(name = "rpn_cls", kernel_size = (1,1), strides = 1, filters = anchor_num, padding = "same", activation = "sigmoid", kernel_initializer = initial_weights)
        self._rpn_reg = Conv2D(name = "rpn_reg", kernel_size = (1,1), strides = 1, filters = 4 * anchor_num, padding = "same", activation = "linear", kernel_initializer = initial_weights)
        
    def __call__(self, inputs, training):
    
        input_image = inputs[0]
        feature_map = inputs[1]
        anchor_map = inputs[2]
        anchor_valid_map = inputs[3]
        assert len(feature_map.shape) == 4
        
        if training:
            max_proposals_pre_nms = self._max_proposals_pre_nms_train
            max_proposals_post_nms = self._max_proposals_post_nms_train
        else:
            max_proposals_pre_nms = self._max_proposals_pre_nms_pred
            max_proposals_post_nms = self._max_proposals_post_nms_pred
        
        y = self._rpn_conv1(feature_map)
        scores = self._rpn_cls(y)
        bbox_regressions = self._rpn_reg(y)
        
        anchors, objectness_scores, box_deltas = self._extract_valid(
          anchor_map = anchor_map,
          anchor_valid_map = anchor_valid_map,
          objectness_score_map = scores,
          box_delta_map = bbox_regressions,
        )
        
        proposals = utils.deltas_to_bboxes(
          deltas = box_deltas,
          anchors = anchors,
          means = [ 0.0, 0.0, 0.0, 0.0 ],
          stds = [ 1.0, 1.0, 1.0, 1.0 ]
        )
        
        # Keep only the top-N scores. 
        sorted_indices = tf.argsort(objectness_scores)                  
        sorted_indices = sorted_indices[::-1]                           
        proposals = tf.gather(proposals, indices = sorted_indices)[0:max_proposals_pre_nms] 
        objectness_scores = tf.gather(objectness_scores, indices = sorted_indices)[0:max_proposals_pre_nms] 
        
        # Clip to image boundaries
        image_height = tf.cast(tf.shape(input_image)[1], dtype = tf.float32)  
        image_width = tf.cast(tf.shape(input_image)[2], dtype = tf.float32)   
        proposals_top_left = tf.maximum(proposals[:,0:2], 0.0)
        proposals_y2 = tf.reshape(tf.minimum(proposals[:,2], image_height), shape = (-1, 1))  
        proposals_x2 = tf.reshape(tf.minimum(proposals[:,3], image_width), shape = (-1, 1))
        proposals = tf.concat([ proposals_top_left, proposals_y2, proposals_x2 ], axis = 1) 
        
        # Remove anything less than 16 pixels on a side
        height = proposals[:,2] - proposals[:,0]
        width = proposals[:,3] - proposals[:,1]
        idxs = tf.where((height >= 16) & (width >= 16))
        proposals = tf.gather_nd(proposals, indices = idxs)
        objectness_scores = tf.gather_nd(objectness_scores, indices = idxs)
        
        # Perform NMS
        idxs = tf.image.non_max_suppression(
          boxes = proposals,
          scores = objectness_scores,
          max_output_size = max_proposals_post_nms,
          iou_threshold = 0.7
        )
        proposals = tf.gather(proposals, indices = idxs)
        
        return [ scores, bbox_regressions, proposals ] 
    
    def _extract_valid(self, anchor_map, anchor_valid_map, objectness_score_map, box_delta_map):
    
        height = tf.shape(anchor_valid_map)[1]
        width = tf.shape(anchor_valid_map)[2]
        num_anchors = tf.shape(anchor_valid_map)[3]
        
        anchors = tf.reshape(anchor_map, shape = (height * width * num_anchors, 4))             # [N,4]
        anchors_valid = tf.reshape(anchor_valid_map, shape = (height * width * num_anchors, 1)) # [N,1]
        scores = tf.reshape(objectness_score_map, shape = (height * width * num_anchors, 1))    # [N,1]
        box_deltas = tf.reshape(box_delta_map, shape = (height * width * num_anchors, 4))       # [N,4]
        
        anchors_valid = tf.squeeze(anchors_valid)                                               # [N]
        scores = tf.squeeze(scores)                                                             # [N]
        
        return anchors, scores, box_deltas
        
    @staticmethod
    def cls_loss(y_pred, gt_rpn_map):
    
        y_true_class = tf.reshape(gt_rpn_map[:,:,:,:,1], shape = tf.shape(y_pred))
        y_predicted_class = y_pred
        y_mask = tf.reshape(gt_rpn_map[:,:,:,:,0], shape = tf.shape(y_predicted_class))
        
        N_cls = tf.cast(tf.math.count_nonzero(y_mask), dtype = tf.float32) + K.epsilon()
        loss_all_anchors = K.binary_crossentropy(y_true_class, y_predicted_class)
        relevant_loss_terms = y_mask * loss_all_anchors
        return K.sum(relevant_loss_terms) / N_cls
    
    @staticmethod
    def reg_loss(y_pred, gt_rpn_map):
        scale_factor = 1.0  
        sigma = 3.0         # source: https://github.com/rbgirshick/py-faster-rcnn/issues/89
        sigma_squared = sigma * sigma
        
        y_predicted_regression = y_pred
        y_true_regression = tf.reshape(gt_rpn_map[:,:,:,:,2:6], shape = tf.shape(y_predicted_regression))
        y_included = tf.reshape(gt_rpn_map[:,:,:,:,0], shape = tf.shape(gt_rpn_map)[0:4])
        y_positive = tf.reshape(gt_rpn_map[:,:,:,:,1], shape = tf.shape(gt_rpn_map)[0:4]) 
        y_mask = y_included * y_positive
        y_mask = tf.repeat(y_mask, repeats = 4, axis = 3)
        
        N_cls = tf.cast(tf.math.count_nonzero(y_included), dtype = tf.float32) + K.epsilon()
        
        x = y_true_regression - y_predicted_regression
        x_abs = tf.math.abs(x)
        is_negative_branch = tf.stop_gradient(tf.cast(tf.less(x_abs, 1.0 / sigma_squared), dtype = tf.float32))
        R_negative_branch = 0.5 * x * x * sigma_squared
        R_positive_branch = x_abs - 0.5 / sigma_squared
        loss_all_anchors = is_negative_branch * R_negative_branch + (1.0 - is_negative_branch) * R_positive_branch
        
        relevant_loss_terms = y_mask * loss_all_anchors
        return scale_factor * K.sum(relevant_loss_terms) / N_cls