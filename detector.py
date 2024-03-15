#DONE

import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import models
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras import backend as K

class DN(tf.keras.Model):
    def __init__(self, n_of_classes, actclassoutputs, l2, dropout_prob):
        super().__init__()
        
        self._num_classes = n_of_classes
        self._activate_class_outputs = actclassoutputs
        self._dropout_probability = dropout_prob
        
        regularizer = tf.keras.regularizers.l2(l2)
        class_initializer = tf.keras.initializers.RandomNormal(mean = 0.0, stddev = 0.01)
        regressor_initializer = tf.keras.initializers.RandomNormal(mean = 0.0, stddev = 0.001)
        
        self._flatten = TimeDistributed(Flatten())
        self._fc1 = TimeDistributed(name = "fc1", layer = Dense(units = 4096, activation = "relu", kernel_regularizer = regularizer))
        self._dropout1 = TimeDistributed(Dropout(dropout_prob))
        self._fc2 = TimeDistributed(name = "fc2", layer = Dense(units = 4096, activation = "relu", kernel_regularizer = regularizer))
        self._dropout2 = TimeDistributed(Dropout(dropout_prob))
        
        class_activation = "softmax" if actclassoutputs else None
        self._classifier = TimeDistributed(name = "classifier_class", layer = Dense(units = n_of_classes, activation = class_activation, kernel_initializer = class_initializer))
        self._regressor = TimeDistributed(name = "classifier_boxes", layer = Dense(units = 4 * (n_of_classes - 1), activation = "linear", kernel_initializer = regressor_initializer))
        
    def call(self, inp, train):
        input_image = inp[0]
        feature_map = inp[1]
        proposals = inp[2]
        assert len(feature_map.shape) == 4
        
        image_height = tf.shape(input_image)[1] 
        image_width = tf.shape(input_image)[2]
        rois = proposals / [ image_height, image_width, image_height, image_width ]
        
        num_rois = tf.shape(rois)[0];
        region = tf.image.crop_and_resize(image = feature_map, boxes = rois, box_indices = tf.zeros(num_rois, dtype = tf.int32), crop_size = [14, 14])
        pool = tf.nn.max_pool(region, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")
        pool = tf.expand_dims(pool, axis = 0)
              
        flattened = self._flatten(pool)
        if train and self._dropout_probability != 0:
            fc1 = self._fc1(flattened)
            do1 = self._dropout1(fc1)
            fc2 = self._fc2(do1)
            do2 = self._dropout2(fc2)
            out = do2
        else:
            fc1 = self._fc1(flattened)
            fc2 = self._fc2(fc1)
            out = fc2 
        class_activation = "softmax" if self._activate_class_outputs else None
        classes = self._classifier(out)
        box_deltas = self._regressor(out)
        
        return [ classes, box_deltas ]
    
    @staticmethod
    def cls_loss(y_pred, y_true, f_logits):
        scale_factor = 1.0
        N = tf.cast(tf.shape(y_true)[1], dtype = tf.float32) + K.epsilon()  # number of proposals
        if f_logits:
          return scale_factor * K.sum(K.categorical_crossentropy(target = y_true, output = y_pred, f_logits = True)) / N
        else:
          return scale_factor * K.sum(K.categorical_crossentropy(y_true, y_pred)) / N
    
    @staticmethod
    def reg_loss(y_pred, y_true):
        scale_factor = 1.0
        sigma = 1.0
        sigma_squared = sigma * sigma
        y_mask = y_true[:,:,0,:]
        y_true_targets = y_true[:,:,1,:]
        x = y_true_targets - y_pred
        x_abs = tf.math.abs(x)
        is_negative_branch = tf.stop_gradient(tf.cast(tf.less(x_abs, 1.0 / sigma_squared), dtype = tf.float32))
        R_negative_branch = 0.5 * x * x * sigma_squared
        R_positive_branch = x_abs - 0.5 / sigma_squared
        losses = is_negative_branch * R_negative_branch + (1.0 - is_negative_branch) * R_positive_branch
        N = tf.cast(tf.shape(y_true)[1], dtype = tf.float32) + K.epsilon() 
        relevant_loss_terms = y_mask * losses
        return scale_factor * K.sum(relevant_loss_terms) / N