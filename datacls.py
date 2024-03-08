from dataclasses import dataclass
import numpy as np
from typing import List
from typing import Tuple

@dataclass
class Box:
    class_index : int
    class_name : str
    corners : np.ndarray

    def __repr__(self):
        return "[class=%s (%f,%f,%f,%f)]" % (self.class_name, self.corners[0], self.corners[1], self.corners[2], self.corners[3])

    def __str__(self):
        return repr(self)

@dataclass
class TrainingSample:
    anchor_map: np.ndarray 
    gt_rpn_map: np.ndarray
    gt_rpn_object_indices: List[Tuple[int,int,int]] # Stores y, x and k values, where k is the ratio, x is horizontal val and y is vertical val
    gt_rpn_background_indices: List[Tuple[int,int,int]]
    gt_boxes: List[Box]
    image_data: np.ndarray
    image : np.ndarray
    filepath: str