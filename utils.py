# DONE

import tensorflow as tf
import numpy as np
def iou_numpy(boxes1, boxes2):
 
  top_left_point = np.maximum(boxes1[:,None,0:2], boxes2[:,0:2])                                  
  bottom_right_point = np.minimum(boxes1[:,None,2:4], boxes2[:,2:4])                             
  well_ordered_mask = np.all(top_left_point < bottom_right_point, axis = 2)                    
  intersection_areas = well_ordered_mask * np.prod(bottom_right_point - top_left_point, axis = 2) 
  areas1 = np.prod(boxes1[:,2:4] - boxes1[:,0:2], axis = 1)                                      
  areas2 = np.prod(boxes2[:,2:4] - boxes2[:,0:2], axis = 1)                                      
  union_areas = areas1[:,None] + areas2 - intersection_areas                                  
  epsilon = 1e-7
  return intersection_areas / (union_areas + epsilon)

def iou(bbox1, bbox2):
  b1 = tf.reshape(tf.tile(tf.expand_dims(bbox1, 1),
                          [1, 1, tf.shape(bbox2)[0]]), [-1, 4])
  b2 = tf.tile(bbox2, [tf.shape(bbox1)[0], 1])
    
  # Compute intersections
  b1_y1, b1_x1, b1_y2, b1_x2 = tf.split(b1, 4, axis=1)
  b2_y1, b2_x1, b2_y2, b2_x2 = tf.split(b2, 4, axis=1)
  y1 = tf.maximum(b1_y1, b2_y1)
  x1 = tf.maximum(b1_x1, b2_x1)
  y2 = tf.minimum(b1_y2, b2_y2)
  x2 = tf.minimum(b1_x2, b2_x2)
  intersection = tf.maximum(x2 - x1, 0) * tf.maximum(y2 - y1, 0)
  # Compute unions
  b1_area = (b1_y2 - b1_y1) * (b1_x2 - b1_x1)
  b2_area = (b2_y2 - b2_y1) * (b2_x2 - b2_x1)
  union = b1_area + b2_area - intersection
  # Compute IoU and reshape to [boxes1, boxes2]
  iou = intersection / union
  overlaps = tf.reshape(iou, [tf.shape(bbox1)[0], tf.shape(bbox2)[0]])
  return overlaps

def deltas_to_bboxes(deltas,means,stds,anchors):

  deltas = deltas * stds + means
  center = anchors[:,2:4] * deltas[:,0:2] + anchors[:,0:2]  
  size = anchors[:,2:4] * tf.math.exp(deltas[:,2:4])
  boxes_top_left = center - 0.5 * size
  boxes_bottom_right = center + 0.5 * size
  boxes = tf.concat([ boxes_top_left, boxes_bottom_right ], axis = 1) 
  return boxes

def deltas_to_bboxes_numpy(deltas, anchors, means, stds):
  deltas = deltas * stds + means
  center = anchors[:,2:4] * deltas[:,0:2] + anchors[:,0:2]  
  size = anchors[:,2:4] * np.exp(deltas[:,2:4])             
  boxes = np.empty(deltas.shape)
  boxes[:,0:2] = center - 0.5 * size                            
  boxes[:,2:4] = center + 0.5 * size                            
  return boxes

class BestWeightsTracker:
  def __init__(self, filepath):
    self._filepath = filepath
    self._best_weights = None
    self._best_mAP = 0

  def on_epoch_end(self, model, mAP):
    if mAP > self._best_mAP:
      self._best_mAP = mAP
      self._best_weights = model.get_weights()

  def restore_and_save_best_weights(self, model):
    if self._best_weights is not None:
      model.set_weights(self._best_weights)
      model.save_weights(filepath = self._filepath, overwrite = True, save_format = "h5")
      print("Saved best model weights (Mean Average Precision = %1.2f%%) to '%s'" % (self._best_mAP, self._filepath))
    else:
        print("No weights have been saved yet.")