import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import models
from tensorflow.keras.layers import Conv2D

import utils

class RPN(tf.keras.Model):
    def __init__(self,max_proposals_pre_nms_train, max_proposals_post_nms_train, max_proposals_pre_nms_pred, max_proposals_post_nms_pred, l2 = 0):
        super.__init__()

        self.max_proposals_pre_nms_train = max_proposals_pre_nms_train
        self.max_proposals_post_nms_train = max_proposals_post_nms_train
        self.max_proposals_pre_nms_pred = max_proposals_pre_nms_pred
        self.max_proposals_post_nms_pred = max_proposals_post_nms_pred
    
        regularizer = tf.keras.regularizers.l2(l2)
        initial_weights = tf.keras.initializers.RandomNormal(mean = 0.0, stddev = 0.01, seed = 42)
        anchor_num = 9
    
        self.rpn_conv_1 = Conv2D(name = "rpn_conv_layer_1", kernel_size = (3,3), strides = 1, padding = "same", filters = 512, kernel_regularizer = regularizer, kernel_initializer = initial_weights, activation = "relu")
        self.rpn_cls = Conv2D(name = "rpn_cls", kernel_size = (1,1), strides = 1, padding = "same", filters = anchor_num, kernel_initializer = initial_weights, activation = "softmax")
        self.rpn_reg = Conv2D(name = "rpn_reg", kernel_size = (1,1), strides = 1, padding = "same", filters = 4*anchor_num, kernel_initializer = initial_weights, activation = "linear")

    def call(self, inputs, training):
        image = inputs[0]
        feature_map = inputs[1]
        anchor_map = inputs[2]

        assert len(feature_map.shape) == 3   

        if training:
            max_proposals_pre_nms, max_proposals_post_nms = self.max_proposals_pre_nms_train, self.max_proposals_post_nms_train
        else:
            max_proposals_pre_nms, max_proposals_post_nms = self.max_proposals_pre_nms_pred, self.max_proposals_post_nms_pred

        y = self.rpn_conv_1(feature_map)
        scores = self.rpn_cls(y)
        bbox_regressions = self.rpn_reg(y)

        # Assuming that anchor_map has shape of (h*w*num,4) 
        # TODO: Make sure this is true
        row = tf.shape(anchor_map)[0]
        anchors = tf.reshape(anchor_map, shape = (row,4))
        objectness_scores = tf.reshape(scores, shape = (row,1))
        box_deltas = tf.reshape(bbox_regressions, shape = (row,4))

        proposals = utils.deltas_to_bboxes(
            deltas = box_deltas,
            anchors = anchors,
            means = [0.0,0.0,0.0,0.0],
            stds = [1.0,1.0,1.0,1.0]
        )

        # Selecting the K best proposals
        sorted_indices = tf.argsort(objectness_scores)
        sorted_indices = sorted_indices[::-1]
        proposals = tf.gather(proposals, indices = sorted_indices)[:max_proposals_pre_nms]
        objectness_scores = tf.gather(objectness_scores, indices = sorted_indices)[:max_proposals_pre_nms]

        # Clipping values within image boundaries
        h = tf.cast(tf.shape(image)[1], dtype = tf.float32)
        w = tf.cast(tf.shape(image)[2], dtype = tf.float32)
        prop_top_left = tf.maximum(proposals[:,0:2], 0.0)
        prop_y2 = tf.reshape(tf.minimum(proposals[:,2],h), shape = (-1,1))
        prop_x2 = tf.reshape(tf.minimum(proposals[:,3],w), shape = (-1,1))
        proposals = tf.concat([prop_top_left, prop_y2, prop_x2], axis = -1)
        
        # Removing proposals with size lesser than the predefine feature scale of 16
        height = proposals[:,2] - proposals[:,0]
        width = proposals[:,3] - proposals[:,1]
        indices = tf.where((height>=16) & (width>=16))
        proposals = tf.gather_nd(proposals, indices = indices)
        objectness_scores = tf.gather_nd(objectness_scores, indices = indices)

        # Performing non max suppression
        indices = tf.image.non_max_suppression(
            boxes = proposals,
            scores = objectenss_scores,
            max_output_size = max_proposals_post_nms,
            iou_threshold = 0.7
        )

        proposals = tf.gather(proposals, indices = indices)
        return [ scores, box_deltas, proposals ]