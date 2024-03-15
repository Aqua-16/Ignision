# DONE

import numpy as np
from math import sqrt
from . import utils
import itertools

def _compute_anchor_sizes():
  areas = [ 128*128, 256*256, 512*512 ]   # pixels
  x_aspects = [ 0.5, 1.0, 2.0 ]           # aspect ratios

  heights = np.array([ x_aspects[j] * sqrt(areas[i] / x_aspects[j]) for (i, j) in itertools.product(range(3), range(3)) ])
  widths = np.array([ sqrt(areas[i] / x_aspects[j]) for (i, j) in itertools.product(range(3), range(3)) ])

  return np.vstack([ heights, widths ]).T

def generate_anchor_map(image_shape, feature_scale): 

  assert len(image_shape) == 3
 
  anchor_sizes = _compute_anchor_sizes()
  num_anchors = anchor_sizes.shape[0]
  anchor_template = np.empty((num_anchors, 4))
  anchor_template[:,0:2] = -0.5 * anchor_sizes  
  anchor_template[:,2:4] = +0.5 * anchor_sizes  

  height, width = image_shape[0] // feature_scale, image_shape[1] // feature_scale

  y_cell_coords = np.arange(height)
  x_cell_coords = np.arange(width)
  cell_coords = np.array(np.meshgrid(y_cell_coords, x_cell_coords)).transpose([2, 1, 0])

  center_points = cell_coords * feature_scale + 0.5 * feature_scale
  center_points = np.tile(center_points, reps = 2)
  center_points = np.tile(center_points, reps = num_anchors)

  anchors = center_points.astype(np.float32) + anchor_template.flatten()
  anchors = anchors.reshape((height*width*num_anchors, 4))
  image_height, image_width = image_shape[0:2]
  valid = np.all((anchors[:,0:2] >= [0,0]) & (anchors[:,2:4] <= [image_height,image_width]), axis = 1)
  anchor_map = np.empty((anchors.shape[0], 4))
  anchor_map[:,0:2] = 0.5 * (anchors[:,0:2] + anchors[:,2:4])
  anchor_map[:,2:4] = anchors[:,2:4] - anchors[:,0:2]

  anchor_map = anchor_map.reshape((height, width, num_anchors * 4))
  anchor_valid_map = valid.reshape((height, width, num_anchors))
    
  return anchor_map.astype(np.float32), anchor_valid_map.astype(np.float32)

def generate_rpn_map(anchor_map, valid, gt_boxes, object_threshold = 0.7, background_threshold = 0.3):
 
  height, width, num_anchors = valid.shape
  gt_box_corners = np.array([ box.corners for box in gt_boxes ])
  num_gt_boxes = len(gt_boxes)

  gt_box_centers = 0.5 * (gt_box_corners[:,0:2] + gt_box_corners[:,2:4])
  gt_box_length = gt_box_corners[:,2:4] - gt_box_corners[:,0:2]

  anchor_map = anchor_map.reshape((-1,4))
  anchors = np.empty(anchor_map.shape)
  anchors[:,0:2] = anchor_map[:,0:2] - 0.5 * anchor_map[:,2:4]  
  anchors[:,2:4] = anchor_map[:,0:2] + 0.5 * anchor_map[:,2:4]  
  n = anchors.shape[0]

  # Initialize all anchors initially as negative (background). We will also track which ground truth box was assigned to each anchor.
  objectness_score = np.full(n, -1)   # Keys = Values : 0 = background, 1 = foreground, -1 = invalid
  gt_box_assignments = np.full(n, -1) # -1 means no box
    
  ious = utils.iou_numpy(boxes1 = anchors, boxes2 = gt_box_corners)
  ious[valid.flatten() == 0, :] = -1.0

  max_iou_per_anchor = np.max(ious, axis = 1)           
  best_box_idx_per_anchor = np.argmax(ious, axis = 1)   
  max_iou_per_gt_box = np.max(ious, axis = 0)           
  highest_iou_anchor_idxs = np.where(ious == max_iou_per_gt_box)[0]
    
  objectness_score[max_iou_per_anchor < background_threshold] = 0
  objectness_score[max_iou_per_anchor >= object_threshold] = 1

  # Anchors that overlap the most with ground truth boxes are positive
  objectness_score[highest_iou_anchor_idxs] = 1

  # We assign the highest IoU ground truth box to each anchor. If no box met
  # the IoU threshold, the highest IoU box may happen to be a box for which
  # the anchor had the highest IoU. If not, then the objectness score will be
  # negative and the box regression won't ever be used.
  gt_box_assignments[:] = best_box_idx_per_anchor

 
  mask = (objectness_score >= 0).astype(np.float32)
  objectness_score[objectness_score < 0] = 0
  
  box_delta_targets = np.empty((n, 4))
  box_delta_targets[:,0:2] = (gt_box_centers[gt_box_assignments] - anchor_map[:,0:2]) / anchor_map[:,2:4]
  box_delta_targets[:,2:4] = np.log(gt_box_length[gt_box_assignments] / anchor_map[:,2:4])                
    
  rpn_map = np.zeros((height, width, num_anchors, 6))
  rpn_map[:,:,:,0] = valid * mask.reshape((height,width,num_anchors))  
  rpn_map[:,:,:,1] = objectness_score.reshape((height,width,num_anchors))
  rpn_map[:,:,:,2:6] = box_delta_targets.reshape((height,width,num_anchors,4))
  
  
  rpn_map_coords = np.transpose(np.mgrid[0:height,0:width,0:num_anchors], (1,2,3,0))                  
  object_anchor_idxs = rpn_map_coords[np.where((rpn_map[:,:,:,1] > 0) & (rpn_map[:,:,:,0] > 0))]      
  background_anchor_idxs = rpn_map_coords[np.where((rpn_map[:,:,:,1] == 0) & (rpn_map[:,:,:,0] > 0))] 

  return rpn_map.astype(np.float32), object_anchor_idxs, background_anchor_idxs