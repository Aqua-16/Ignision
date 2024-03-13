# DONE

import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import models
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.initializers import GlorotNormal

class BackBone(tf.keras.Model):
    def __init__(self,l2=0):
        super().__init__()

        initial_weights = GlorotNormal(seed = 42)
        regularizer = tf.keras.regularizers.l2(l2)
        input_shape = (None,None,3)

        self.block1_conv1 = Conv2D(name = "block1_conv1", kernel_size = (3,3), input_shape = input_shape, strides = 1, padding = "same", filters = 64, activation = "relu", kernel_initializer = initial_weights, trainable = False)
        self.block1_conv2 = Conv2D(name = "block1_conv2", kernel_size = (3,3), strides = 1, padding = "same", filters = 64, activation = "relu", kernel_initializer = initial_weights, trainable = False)
        self.block1_maxpool = MaxPooling2D(name = "block1_pool", pool_size = 2, strides = 2)

        self.block2_conv1 = Conv2D(name = "block2_conv1", kernel_size = (3,3), strides = 1, padding = "same", filters = 128, activation = "relu", kernel_initializer = initial_weights, trainable = False)
        self.block2_conv2 = Conv2D(name = "block2_conv2", kernel_size = (3,3), strides = 1, padding = "same", filters = 128, activation = "relu", kernel_initializer = initial_weights, trainable = False)
        self.block2_maxpool = MaxPooling2D(name = "block2_pool", pool_size = 2, strides = 2)

        self.block3_conv1 = Conv2D(name = "block3_conv1", kernel_size = (3,3), strides = 1, padding = "same", filters = 256, activation = "relu", kernel_initializer = initial_weights, kernel_regularizer = regularizer)
        self.block3_conv2 = Conv2D(name = "block3_conv2", kernel_size = (3,3), strides = 1, padding = "same", filters = 256, activation = "relu", kernel_initializer = initial_weights, kernel_regularizer = regularizer)
        self.block3_conv3 = Conv2D(name = "block3_conv3", kernel_size = (3,3), strides = 1, padding = "same", filters = 256, activation = "relu", kernel_initializer = initial_weights, kernel_regularizer = regularizer)
        self.block3_maxpool = MaxPooling2D(name = "block3_pool", pool_size = 2, strides = 2)

        self.block4_conv1 = Conv2D(name = "block4_conv1", kernel_size = (3,3), strides = 1, padding = "same", filters = 512, activation = "relu", kernel_initializer = initial_weights, kernel_regularizer = regularizer)
        self.block4_conv2 = Conv2D(name = "block4_conv2", kernel_size = (3,3), strides = 1, padding = "same", filters = 512, activation = "relu", kernel_initializer = initial_weights, kernel_regularizer = regularizer)
        self.block4_conv3 = Conv2D(name = "block4_conv3", kernel_size = (3,3), strides = 1, padding = "same", filters = 512, activation = "relu", kernel_initializer = initial_weights, kernel_regularizer = regularizer)
        self.block4_maxpool = MaxPooling2D(name = "block4_pool", pool_size = 2, strides = 2)

        self.block5_conv1 = Conv2D(name = "block5_conv1", kernel_size = (3,3), strides = 1, padding = "same", filters = 512, activation = "relu", kernel_initializer = initial_weights, kernel_regularizer = regularizer)
        self.block5_conv2 = Conv2D(name = "block5_conv2", kernel_size = (3,3), strides = 1, padding = "same", filters = 512, activation = "relu", kernel_initializer = initial_weights, kernel_regularizer = regularizer)
        self.block5_conv3 = Conv2D(name = "block5_conv3", kernel_size = (3,3), strides = 1, padding = "same", filters = 512, activation = "relu", kernel_initializer = initial_weights, kernel_regularizer = regularizer)

    def call(self,input_image):
        blocks = [self.block1_conv1, self.block1_conv2, self.block1_maxpool, 
                  self.block2_conv1, self.block2_conv2, self.block2_maxpool,
                  self.block3_conv1, self.block3_conv2, self.block3_conv3, self.block3_maxpool,
                  self.block4_conv1, self.block4_conv2, self.block4_conv3, self.block4_maxpool,
                  self.block5_conv1, self.block5_conv2, self.block5_conv3]

        y = input_image
        for block in blocks:
            y = block(y)

        return y