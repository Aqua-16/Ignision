import cv2
import numpy as np

LOW_VAL = 180
HIGH_VAL = 255

LOW_SAT = 40
HIGH_SAT = 255

LOW_HUE = 0
HIGH_HUE = 39

def preprocess_vgg16(image_data):
    image_data[:, :, 0] -= 103.939                # ImageNet B mean
    image_data[:, :, 1] -= 116.779                # ImageNet G mean
    image_data[:, :, 2] -= 123.680                # ImageNet R mean 
    return image_data
    
def load_image(path,flip=None):
    image = cv2.imread(path)
    h = image.shape[0]
    w = image.shape[1]
    if flip:
        cv2.flip(image,1)
    image_HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(image_HSV, (LOW_HUE, LOW_SAT, LOW_VAL), (HIGH_HUE, HIGH_SAT, HIGH_VAL))
    image = cv2.bitwise_and(image, image, mask=mask)
    image_data = image.astype(np.float32)
    image_data = preprocess_vgg16(image_data)
    return image_data,image,(image_data.shape[0],h,w)

def show_detections(out_path,image,scored_boxes_class_idx,class_idx_name):
    image_ = image.copy()
    color = (32,32,32)
    for cls_idx,scored_boxes in scored_boxes_class_idx.items():
        for i in range(scored_boxes.shape[0]):
            scored_box = scored_boxes[i:][0:4].astype(int)
            cls_name = class_idx_name[cls_idx]

            cv2.rectangle(image_,(scored_box[0],scored_box[1]),(scored_box[2],scored_box[3]),color, thickness = 2)
            cv2.putText(image_,cls_name,(scored_box[1],scored_box[0]),cv2.FONT_HERSHEY_SIMPLEX,1.5,color,thickness = 1.8)

    cv2.imshow("Detections", image_)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if out_path is not None:
        cv2.imwrite(out_path,image_)
        print("Successfully saved image to '%s'" %out_path)

