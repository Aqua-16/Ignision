import numpy as np
from . import utils

def generate_anchor_map(image_shape,feature_scale):
    assert len(image_shape) == 3

    areas = [ 128*128, 256*256, 512*512 ] # Values used as per paper
    aspect_ratios = [ 0.5, 1.0, 2.0 ] # Values used as per paper

    h = np.array([ aspect_ratios[j] * ((areas[i]/aspect_ratios[j])**0.5) for (i,j) in [(x,y) for x in range(3) for y in range(3)]])
    w = np.array([ ((areas[i]/aspect_ratios[j])**0.5) for (i,j) in [(x,y) for x in range(3) for y in range(3)]])

    anchor_sizes = np.vstack([h,w]).T

    num = np.shape(anchor_sizes)[0]
    anchors_base = np.empty((num,4))
    anchors_base[:,0:2] = anchor_sizes*(-0.5)
    anchors_base[:,2:4] = anchor_sizes*(0.5)

    h = image_shape[0]//feature_scale
    w = image_shape[1]//feature_scale

    y_coords = np.arange(h)
    x_coords = np.arange(w)
    grid = np.array(np.meshgrid(y_coords,x_coords)).transpose(2,1,0)

    anchor_center_grid = grid * feature_scale + 0.5 * feature_scale
    anchor_center_grid = np.tile(anchor_center_grid,2*num)
    anchor_center_grid = anchor_center_grid.astype(np.float32) + anchors_base.flatten()

    anchors = anchor_center_grid.reshape((h*w*num,4))

    # Creating anchor_map of the type [center_y,center_x, height, width] as is given in the paper
    anchor_map = np.empty((anchors.shape[0],4))
    anchor_map[:,0:2] = 0.5*(anchors[:,0:2] + anchors[:,2:4])
    anchor_map[:,2:4] = anchors[:,2:4] - anchors[:,0:2]

    # This step is done only to ensure that the final shape is as expected.
    anchor_map = anchor_map.reshape((h,w,num*4))
    
    return anchor_map.astype(np.float32)

def generate_rpn_map(anchor_map, gt_boxes, object_threshold = 0.7, background_threshold = 0.3):

    h, w, num_anchors = anchor_map.shape
    num_anchors = num_anchors//4

    gt_box_corners = np.array([box.corners for box in gt_boxes])
    num_gt_boxes = len(gt_box_corners)
    gt_box_centers = 0.5*(gt_box_corners[:,0:2] + gt_box_corners[:,2:4])
    gt_box_lengths = gt_box_corners[:,2:4] - gt_box_corners[:,0:2]

    # Generating anchor corners
    anchor_map = anchor_map.reshape((-1,4))
    anchors = np.empty(anchor_map.shape)
    anchors[:,0:2] = anchor_map[:,0:2] - 0.5*anchor_map[:,2:4]
    anchors[:,2:4] = anchor_map[:,0:2] + 0.5*anchor_map[:,2:4]
    n = anchors.shape[0]

    object_score = np.full(n,-1)
    gt_box_assignment = np.full(n,-1)

    ious = utils.iou(anchors,gt_box_corners).numpy()

    max_iou_anchor = np.max(ious,axis = 1) # Best iou for each anchor
    highest_gt_box_idx = np.argmax(ious,axis = 1) # Best ground truth box for each anchor
    max_iou_gt_box = np.max(ious,axis = 0) # Best iou for each ground truth box
    highest_anchor_idx = np.where(max_iou_gt_box == ious)[0] # Best anchor for each ground truth box

    object_score[max_iou_anchor < background_threshold] = 0
    object_score[max_iou_anchor >= object_threshold] = 1
    # Setting anchor boxes with highest ious as 1
    object_score[highest_anchor_idx] = 1 
    gt_box_assignment[:] = highest_gt_box_idx

    mask = (object_score >= 0).astype(np.float32) # Creating mask for where objects are present
    object_score[object_score<0] = 0

    # Box deltas for regression of anchor boxes
    box_deltas = np.empty((n,4))
    box_deltas[:,0:2] = (gt_box_centers[gt_box_assignment] - anchor_map[:,0:2])/anchor_map[:,2:4]
    box_deltas[:,2:4] = np.log(gt_box_lengths[gt_box_assignment]/anchor_map[:,2:4])

    rpn_map = np.empty((h,w,num_anchors,6))
    rpn_map[:,:,:,0] = mask.reshape((h,w,num_anchors))
    rpn_map[:,:,:,1] = object_score.reshape((h,w,num_anchors))
    rpn_map[:,:,:,2:6] = box_deltas.reshape((h,w,num_anchors,4))

    rpn_map_coords = np.transpose( np.mgrid[0:h,0:w,0:num_anchors], (1,2,3,0)) # Every index will return its own coordinates. Useful to find indices
    pos_anchor_coords = rpn_map_coords[np.where((rpn_map[:,:,:,1] > 0) & (rpn_map[:,:,:,0] > 0))] # Anchors with object
    neg_anchor_coords = rpn_map_coords[np.where((rpn_map[:,:,:,1] == 0) & (rpn_map[:,:,:,0] > 0))] # Anchors without objects

    return rpn_map.astype(np.float32), pos_anchor_coords, neg_anchor_coords
