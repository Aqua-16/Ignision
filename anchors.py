import numpy as np

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
    anchor_center_grid = anchor_center_grid.astype("float32") + anchors_base.flatten()

    anchors = anchor_center_grid.reshape((h*w*num,4))

    # Clipping values to prevent anchors from going beyond image
    height, width = image_shape[0],image_shape[1]
    anchors = np.array([[y1/height,x1/width,y2/height,x2/width] for (y1,x1,y2,x2) in anchors])
    anchors = np.clip(anchors,0,1)
    anchors = np.array([[y1*height,x1*width,y2*height,x2*width] for (y1,x1,y2,x2) in anchors])

    # Creating anchor_map of the type [center_y,center_x, height, width] as is given in the paper
    anchor_map = np.empty((anchors.shape[0],4))
    anchor_map[:,0:2] = 0.5*(anchors[:,0:2] + anchors[:,2:4])
    anchor_map[:,2:4] = anchors[:,2:4] - anchors[:,0:2]

    # This step is done only to ensure that the final shape is as expected.
    anchor_map = anchor_map.reshape((h*w*num,4))
    
    return anchor_map.astype("float32")