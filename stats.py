from collections import defaultdict
import numpy as np

from utils import iou #TODO

class train_statistics:

    def __init__(self):
        self.rpn_class_loss=float("inf")
        self.rpn_reg_loss=float("inf")
        self.detector_class_loss=float("inf")
        self.detector_reg_loss=float("inf")
        self.rpn_class_losses=[]
        self.rpn_reg_losses=[]
        self.detector_class_losses=[]
        self.detector_reg_losses=[]

    def during_training_step(self,losses):#used once in every iteration to aggregate losses
        self.rpn_class_losses.append(losses["rpn_class_loss"])
        self.rpn_reg_losses.append(losses["rpn_reg_loss"])
        self.detector_class_losses.append(losses["detector_class_loss"])
        self.detector_reg_losses.append(losses["detector_class_loss"])
        self.rpn_class_loss=np.mean(self.rpn_class_losses)
        self.rpn_reg_loss=np.mean(self.rpn_reg_losses)
        self.detector_class_loss=np.mean(self.detector_class_losses)
        self.detector_reg_loss=np.mean(self.detector_reg_losses)

    def progress_bar_postfix(self):
        return {
        "rpn_class_loss": "%1.4f" % self.rpn_class_loss,
        "rpn_regr_loss": "%1.4f" % self.rpn_reg_loss,
        "detector_class_loss": "%1.4f" % self.detector_class_loss,
        "detector_regr_loss": "%1.4f" % self.detector_reg_loss,
        "total_loss": "%1.2f" % (self.rpn_class_loss + self.rpn_reg_loss + self.detector_class_loss + self.detector_reg_losses)
    }
    
