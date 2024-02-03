import numpy as np
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.layers import Layer

class roi_pooling(Layer):
    def __init__(self, pool_size, **kwargs):
        self.pool_size = pool_size
        super().__init__(**kwargs)

    def get_config(self):
        config = {
        "pool_size": self.pool_size,
        }
        base_config = super(roi_pooling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    def comp_output_shape(self,input_shape):
        map_shape,roi_shape=input_shape
        assert len(map_shape)==4 #shape of feature maps should be 4
        assert len(roi_shape)==3 and roi_shape[2]==4
        assert map_shape[0]==roi_shape[0] # both input tensors should have same number of samples
        num_sample=map_shape[0]
        num_channel=map_shape[3]
        num_roi=roi_shape[1]
        return (num_sample, num_roi, self.pool_size, self.pool_size, num_channel)
    
    def call(self,inputs):
        if self.pool_size == 7 and inputs[0].shape[3] == 512:
            return tf.map_fn(
                fn = lambda input_pair:
                roi_pooling._compute_pooled_roi_7_7_512(feature_map = input_pair[0], roi = input_pair[1]),
                elements = inputs, fn_output_signature = tf.float32  
            )
        else:
            return tf.map_fn(
            fn = lambda input_pair:
            roi_pooling._compute_pooled_roi(feature_map = input_pair[0], roi = input_pair[1], pool_size = self.pool_size),
            elements = inputs,
            fn_output_signature = tf.float32 
            )
        
    @tf.function
    def _comp_pooled_rois(feature_map, rois, pool_size):
        return tf.map_fn(
        fn = lambda roi:
        roi_pooling._compute_pooled_roi(feature_map = feature_map, roi = roi, pool_size = pool_size),
        elements = rois,
        fn_output_signature = tf.float32
    )

    @tf.function
    def _comp_pooled_roi(feature_map, roi, pool_size):
        r_y = roi[0]
        r_x = roi[1]
        r_heig = roi[2]
        r_wid = roi[3]
        num_channel = feature_map.shape[2]
        r_o_i = tf.slice(feature_map, [r_y, r_x, 0], [r_heig, r_wid, num_channel])

        x_step = tf.cast(r_wid, dtype = tf.float32) / tf.cast(pool_size, dtype = tf.float32)
        y_step = tf.cast(r_heig, dtype = tf.float32) / tf.cast(pool_size, dtype = tf.float32)

        x_range = tf.cast(tf.range(pool_size), dtype = tf.float32)
        y_range = tf.cast(tf.range(pool_size), dtype = tf.float32)
        pooled_cells = tf.map_fn(
        fn = lambda y: tf.map_fn(
            fn = lambda x:
            roi_pooling._pool_one_cell(r_o_i, pool_y_start = y, pool_x_start = x, y_step = y_step, x_step = x_step, r_heig = r_heig, r_wid = r_wid, pool_size = pool_size, num_channel = num_channel),
            elements = x_range
        ),
        elements = y_range
        )
        return pooled_cells
    
    @tf.function
    def _pool_1_cell(r_o_i, pool_y_start, pool_x_start, y_step, x_step, r_heig, r_wid, pool_size, num_channel):
        pool_y_start_int = tf.cast(pool_y_start, dtype = tf.int32)
        pool_x_start_int = tf.cast(pool_x_start, dtype = tf.int32)
        y_start = tf.cast(pool_y_start * y_step, dtype = tf.int32)
        x_start = tf.cast(pool_x_start * x_step, dtype = tf.int32)
        y_end = tf.cond((pool_y_start_int + 1) < pool_size,
        lambda: tf.cast((pool_y_start + 1) * y_step, dtype = tf.int32),
        lambda: r_heig
        )
        x_end = tf.cond((pool_x_start_int + 1) < pool_size,
        lambda: tf.cast((pool_x_start + 1) * x_step, dtype = tf.int32),
        lambda: r_wid
        )
        y_size = tf.math.maximum(y_end - y_start, 1) 
        x_size = tf.math.maximum(x_end - x_start, 1)
        pool_cell = tf.slice(r_o_i, [y_start, x_start, 0], [y_size, x_size, num_channel])
        return tf.math.reduce_max(pool_cell, axis=(1,0))
    
    @tf.function
    def _compute_pooled_roi_7_7_512(feature_map,rois):
        return tf.map_fn(
        fn=lambda roi: tf.reshape(
            tf.stack([
                tf.math.reduce_max(
                    tf.slice(
                        feature_map[
                            roi[0]:roi[0] + roi[2],
                            roi[1]:roi[1] + roi[3],
                            0:512
                        ],
                        [
                            tf.cast(y * (tf.cast(roi[2], dtype=tf.float32) / 7), dtype=tf.int32),
                            tf.cast(x * (tf.cast(roi[3], dtype=tf.float32) / 7), dtype=tf.int32),
                            0
                        ],
                        [
                            tf.math.maximum(1, tf.cast((y + 1) * (tf.cast(roi[2], dtype=tf.float32) / 7), dtype=tf.int32) - tf.cast(y * (tf.cast(roi[2], dtype=tf.float32) / 7), dtype=tf.int32)),
                            tf.math.maximum(1, tf.cast((x + 1) * (tf.cast(roi[3], dtype=tf.float32) / 7), dtype=tf.int32) - tf.cast(x * (tf.cast(roi[3], dtype=tf.float32) / 7), dtype=tf.int32)),
                            512
                        ]
                    ),
                    axis=(1, 0)
                )
                for y in range(7) for x in range(7)
            ]),
            shape=(7, 7, 512)
        ),
        elements=rois,
        fn_output_signature=tf.float32
    )




