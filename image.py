# DONE

import cv2
import imageio
from PIL import Image
import numpy as np

def preprocess_vgg16(image_data):
  image_data = image_data[:, :, ::-1]           # RGB -> BGR
  image_data[:, :, 0] -= 103.939                # ImageNet B mean
  image_data[:, :, 1] -= 116.779                # ImageNet G mean
  image_data[:, :, 2] -= 123.680                # ImageNet R mean 
  return image_data
    
def load_image(path,flip=None):
    data = imageio.imread(path, pilmode = "RGB")
    image = Image.fromarray(data, mode = "RGB")
    W, H = image.width, image.height
    if flip:
        image = image.transpose(method = Image.FLIP_LEFT_RIGHT)
    scale_factor=600/min(H,W)
    h = int(H*scale_factor)
    w = int(W*scale_factor)
    image = image.resize((w, h), resample = Image.BILINEAR)
    image_data = np.array(image).astype(np.float32)
    image_data = preprocess_vgg16(image_data = image_data)
    return image_data,image,scale_factor,(image_data.shape[0],H,W)

def show_detections(out_path,image,scored_boxes_class_idx,class_idx_name):
    image_ = np.array(image)
    image_ = cv2.cvtColor(image_, cv2.COLOR_RGB2BGR)
    color = (32,32,32)
    for cls_idx,scored_boxes in scored_boxes_class_idx.items():
        for i in range(scored_boxes.shape[0]):
            scored_box = scored_boxes[i][0:4].astype(int)
            cls_name = class_idx_name[cls_idx]

            cv2.rectangle(image_,(scored_box[0],scored_box[1]),(scored_box[2],scored_box[3]),color, thickness = 2)
            cv2.putText(image_,cls_name,(scored_box[1],scored_box[0]),cv2.FONT_HERSHEY_SIMPLEX,1.5,color,thickness = 2)

    cv2.imshow("Detections", image_)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if out_path is not None:
        cv2.imwrite(out_path,image_)
        print("Successfully saved image to '%s'" %out_path)

