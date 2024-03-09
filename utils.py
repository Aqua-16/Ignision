import tensorflow as tf

def iou(bbox1,bbox2):
    # These create tensors that will help compute the iou of each box in bbox1 with each box in bbox2
    boxes1 = tf.reshape(tf.tile(tf.expand_dims(bbox1,1),[1,1,tf.shape(bbox2)[0]]),[-1,4])
    boxes2 = tf.tile(bbox2,[tf.shape(bbox1)[0],1])

    # Extracting box coordinates of each pair of boxes
    b1_y1,b1_x1,b1_y2,b1_x2 = [tf.cast(tensor, tf.float32) for tensor in tf.split(boxes1, 4, axis = 1)]
    b2_y1,b2_x1,b2_y2,b2_x2 = [tf.cast(tensor, tf.float32) for tensor in tf.split(boxes2, 4, axis = 1)]

    # Calculating intersection area
    y1 = tf.maximum(b1_y1,b2_y1)
    y2 = tf.minimum(b1_y2,b2_y2)
    x1 = tf.maximum(b1_x1,b2_x1)
    x2 = tf.minimum(b1_x2,b2_x2)
    intersection = tf.maximum(y2-y1,0) * tf.maximum(x2-x1,0)

    # Calculating union area
    b1_area = (b1_y2-b1_y1) * (b1_x2-b1_x1)
    b2_area = (b2_y2-b2_y1) * (b2_x2-b2_x1)
    union = b1_area + b2_area - intersection

    # Calculating final iou
    iou = intersection/union
    iou = tf.reshape(iou, [tf.shape(bbox1)[0],tf.shape(bbox2)[0]])

    return iou

def deltas_to_bboxes(deltas,means,stds,anchors):

    deltas = deltas*stds + means
    center = anchors[:,2:4] * deltas[:,0:2] + anchors[:,0:2]
    sides = anchors[:,2:4] * tf.math.exp(deltas[:,2:4])

    box_left_top = center - 0.5 * sides
    box_right_bottom = box_left_top + sides

    bboxes = tf.concat([box_left_top,box_right_bottom],axis = 1)
    return bboxes

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