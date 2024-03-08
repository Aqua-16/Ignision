import numpy as np
import os
from pathlib import Path
import random
import xml.etree.ElementTree as ET
from typing import List
from typing import Tuple
from datacls import Box,TrainingSample
import image
import anchors


class Dataset:
  
  def __init__(self, direc = "dataset", _feature_pixels = 16, augmenting = True, shuffling = True):#if dataset is in any other folder directly change the directory from here
    '''
      direc: directory of dataset
      feature_pixels: size of each cell in img pixels  faster rcnn feature map
      augment: boolean specifying whether to filp 50% of images horizontally to increase diversity
      shuffle: boolean specifying whether to shuffle dataset everytime we iterate it
    '''
    if not os.path.exists(direc):
      raise FileNotFoundError(f"Dataset directory does not exist: {direc}")
    

    self.direc = direc
    self.class_index_to_name = {0:"background",1:"fire"}
    self.class_name_to_index = {"background":0,"fire":1}
    self.num_classes = 2
    self.filepaths = self.file_paths()#returns a list of image file paths
    self.num_samples = len(self.filepaths)#number of samples
    self.gt_boxes_by_filepath = self.get_gt_boxes() 
    self.i = 0 # track current iteration index
    self.iterable_filepaths = self.filepaths.copy()# copy is used for iteration and shuffling if specified
    self.feature_pixels = _feature_pixels
    self.augment = augmenting
    self.shuffle = shuffling
    self.unaugmented_cached_sample_by_filepath = {}
    self.augmented_cached_sample_by_filepath = {}



  def __iter__(self):
    self.i = 0 #instance variable i keeps track of the current iteration index.
    if self.shuffle: # if true then execute
      random.shuffle(self.iterable_filepaths)
    return self

  def __next__(self):
    if self.i >= len(self.iterable_filepaths):
      raise StopIteration
    filepath = self.iterable_filepaths[self.i]
    self.i += 1

    if self.augment:
      flip = random.randint(0, 1) != 0
    else:
      flip = 0 
    
    if flip:
      sample_by_filepath = self.augmented_cached_sample_by_filepath  
    else:
      self.unaugmented_cached_sample_by_filepath
  
    # Load and, if caching, write back to cache
    if filepath in sample_by_filepath:
      sample = sample_by_filepath[filepath]
    else:
      sample = self.generate_training_sample(filepath = filepath, flip = flip)
    print(sample)
    return sample

  def generate_training_sample(self, filepath, flip):
    scaled_image_data, scaled_image, scale_factor, original_shape = image.load_image(url = filepath, min_dimension_pixels = 600, horizontal_flip = flip)
    _, original_height, original_width = original_shape # depth is dicarded

    # Scale ground truth boxes to new image size
    scaled_gt_boxes = []
    for box in self.gt_boxes_by_filepath[filepath]:
      if flip:
        corners = np.array([
          box.corners[0],
          original_width - 1 - box.corners[3],
          box.corners[2],
          original_width - 1 - box.corners[1]
        ]) 
      else:
        corners = box.corners
      scaled_box = Box(
        class_index = box.class_index,
        class_name = box.class_name,
        corners = corners * scale_factor 
      )
      scaled_gt_boxes.append(scaled_box)

    anchor_map = anchors.generate_anchor_map(image_shape = scaled_image_data.shape, feature_pixels = self.feature_pixels)
    gt_rpn_map, gt_rpn_object_indices, gt_rpn_background_indices = anchors.generate_rpn_map(anchor_map = anchor_map, gt_boxes = scaled_gt_boxes)

    # Return sample
    return TrainingSample(
      anchor_map = anchor_map,
      gt_rpn_map = gt_rpn_map,
      gt_rpn_object_indices = gt_rpn_object_indices,
      gt_rpn_background_indices = gt_rpn_background_indices,
      gt_boxes = scaled_gt_boxes,
      image_data = scaled_image_data,
      image = scaled_image,
      filepath = filepath
    )

  def file_paths(self):
    image_paths = [os.path.join(self.direc, "Datacluster Fire and Smoke Sample", f"Datacluster Fire and Smoke Sample ({i}).jpg") for i in range(1, 101)]
    return image_paths


  def get_gt_boxes(self):
    gt_boxes_by_filepath = {}
    annot_paths = [os.path.join(self.direc, "Annotations", f"Datacluster Fire and Smoke Sample ({i}).xml") for i in range(1, 101)]
    for annot_path in annot_paths:
      tree = ET.parse(annot_path)# Parses the XML annotation file using the ElementTree library
      root = tree.getroot()
      assert len(root.findall("size")) == 1 #checks if there is only one size element in the xml file or not
      size = root.find("size")
      boxes = []
      for obj in root.findall("object"):
        assert len(obj.findall("name")) == 1
        assert len(obj.findall("bndbox")) == 1
        class_name = obj.find("name").text
        bndbox = obj.find("bndbox")
        assert len(bndbox.findall("xmin")) == 1
        assert len(bndbox.findall("ymin")) == 1
        assert len(bndbox.findall("xmax")) == 1
        assert len(bndbox.findall("ymax")) == 1
        x_min = bndbox.find("xmin").text
        y_min = bndbox.find("ymin").text
        x_max = bndbox.find("xmax").text
        y_max = bndbox.find("ymax").text
        corners = np.array([ y_min, x_min, y_max, x_max ]).astype(np.float32)
        box = Box(class_index = self.class_name_to_index[class_name], class_name = class_name, corners = corners)
        boxes.append(box)
      #assert len(boxes) > 0                                                #TODO: Check for possible errors and fix them
        # print(f"Boxes in {annot_path}:")
        # for box in boxes:
        #     print(box)
      gt_boxes_by_filepath[annot_path] = boxes
    # for annot_path, boxes in gt_boxes_by_filepath.items():
    #     print(f"Boxes in {annot_path}:")
    #     for box in boxes:
    #         print(box)
    return gt_boxes_by_filepath