import tensorflow as tf
import cv2

def iou(bbox1,bbox2):
    # These create tensors that will help compute the iou of each box in bbox1 with each box in bbox2
    boxes1 = tf.reshape(tf.tile(tf.expand_dims(bbox1,1),[1,1,tf.shape(bbox2)[0]]),[-1,4])
    boxes2 = tf.tile(bbox2,[tf.shape(bbox1)[0],1])

    # Extracting box coordinates of each pair of boxes
    b1_y1,b1_x1,b1_y2,b1_x2 = tf.split(boxes1, 4, axis = 1)
    b2_y1,b2_x1,b2_y2,b2_x2 = tf.split(boxes2, 4, axis = 1)

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

def show_detections(out_path,image,scored_boxes_class_idx,class_idx_name):
    image_ = image.copy()
    color = (32,32,32)
    for cls_idx,scored_boxes in scored_boxes_class_idx.items():
        for i in range(scored_boxes.shape[0]):
            scored_box = scored_boxes[i:][0:4].astype(int)
            cls_name = class_idx_name[cls_idx]

            cv2.rectangle(image_,(scored_box[0],scored_box[1]),(scored_box[2],scored_box[3]),color, thickness = 2)
            cv2.putText(image_,cls_name,(scored_box[1],scored_box[0]),cv2.FONT_HERSHEY_SIMPLEX,1.5,color,thickness = 1.8)

    cv2.imshow("Detections", image_with_detections)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if out_path is not None:
        cv2.imwrite(out_path,image_)
        print("Successfully saved image to '%s'" %out_path)