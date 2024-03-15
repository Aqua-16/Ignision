from collections import defaultdict
import numpy as np

from .utils import iou_numpy
import matplotlib.pyplot as plt


class train_statistics:

    def __init__(self):
        self.rpn_class_loss=float("inf")
        self.rpn_regression_loss=float("inf")
        self.detector_class_loss=float("inf")
        self.detector_regression_loss=float("inf")
        self.rpn_class_losses=[]
        self.rpn_regression_losses=[]
        self.detector_class_losses=[]
        self.detector_regression_losses=[]

    def during_training_step(self,losses):#used once in every iteration to aggregate losses
        self.rpn_class_losses.append(losses["rpn_class_loss"])
        self.rpn_regression_losses.append(losses["rpn_regression_loss"])
        self.detector_class_losses.append(losses["detector_class_loss"])
        self.detector_regression_losses.append(losses["detector_regression_loss"])
        self.rpn_class_loss=np.mean(self.rpn_class_losses)
        self.rpn_regression_loss=np.mean(self.rpn_regression_losses)
        self.detector_class_loss=np.mean(self.detector_class_losses)
        self.detector_regression_loss=np.mean(self.detector_regression_losses)

    def progress_bar_postfix(self):
        return {
        "rpn_class_loss": "%1.4f" % self.rpn_class_loss,
        "rpn_reg_loss": "%1.4f" % self.rpn_regression_loss,
        "detector_class_loss": "%1.4f" % self.detector_class_loss,
        "detector_reg_loss": "%1.4f" % self.detector_regression_loss,
        "total_loss": "%1.2f" % (self.rpn_class_loss + self.rpn_regression_loss + self.detector_class_loss + self.detector_regression_loss)
    }
    
class PRCCalc:

    def __init__(self):
        self._unsorted_pred_by_classindex=defaultdict(list)
        self._obj_cnt_by_classindex=defaultdict(int)

    def compute_correctness_of_pred(self,scored_boxes_by_class_index,gt_boxes):
        unsorted_pred_by_classindex = {}
        obj_cnt_by_classindex = defaultdict(int)
        for gt_box in gt_boxes:
            obj_cnt_by_classindex[gt_box.class_index]+=1

        for class_index, scored_boxes in scored_boxes_by_class_index.items():
            gt_boxes_of_curr_class=[ gt_box for gt_box in gt_boxes if gt_box.class_index == class_index ]

            ious=[]#iou of each box with each groundtruthbox and store in a list
            for gt_index in range(len(gt_boxes_of_curr_class)):
                for box_index in range(len(scored_boxes)):
                    boxes1 = np.expand_dims(scored_boxes[box_index][0:4], axis = 0)
                    boxes2 = np.expand_dims(gt_boxes_of_curr_class[gt_index].corners, axis = 0)
                    iou=iou_numpy(boxes1 = boxes1, boxes2 = boxes2)
                    ious.append((iou, box_index, gt_index))
            ious = sorted(ious, key = lambda iou: ious[0], reverse = True)#sort in descending order

            gt_box_detected=[False]*len(gt_boxes)

            is_true_positive = [False]*len(scored_boxes)

            iou_threshold = 0.5
            for iou, box_index, gt_index in ious:
                if iou <= iou_threshold:
                    continue
                if is_true_positive[box_index] or gt_box_detected[gt_index]:
                    continue
                is_true_positive[box_index] = True
                gt_box_detected[gt_index] = True

            unsorted_pred_by_classindex[class_index]=[(scored_boxes[i][4], is_true_positive[i]) for i in range(len(scored_boxes))]

        return unsorted_pred_by_classindex, obj_cnt_by_classindex
    
    def add_img_result(self,scored_boxes_by_class_index, gt_boxes):
        unsorted_pred_by_classindex,obj_cnt_by_classindex=self.compute_correctness_of_pred(scored_boxes_by_class_index=scored_boxes_by_class_index,gt_boxes=gt_boxes)
        for class_index, preds in unsorted_pred_by_classindex.items():
            self._unsorted_pred_by_classindex[class_index] += preds
        for class_index, count in obj_cnt_by_classindex.items():
            self._obj_cnt_by_classindex[class_index] += obj_cnt_by_classindex[class_index]  

    def compute_avg_precision(self,class_index):
        sorted_preds = sorted(self._unsorted_pred_by_classindex[class_index], key = lambda prediction: prediction[0], reverse = True)
        no_of_grnd_truth_positives = self._obj_cnt_by_classindex[class_index]
        recall = []
        precision = []
        tp = 0 
        fp = 0 
        for i in range(len(sorted_preds)):
            tp += 1 if sorted_preds[i][1] == True else 0
            fp += 0 if sorted_preds[i][1] == True else 1
            r = tp / no_of_grnd_truth_positives
            p = tp / (tp + fp)
            recall.append(r)
            precision.append(p)

        recall.insert(0, 0.0)
        recall.append(1.0)
        precision.insert(0, 0.0)
        precision.append(0.0)
        for i in range(len(precision)):
            precision[i] = np.max(precision[i:])
        avg_precision = 0
        for i in range(len(recall) - 1):
            dx = recall[i + 1] - recall[i + 0]
            dy = precision[i + 1]
            avg_precision += dy * dx

        return avg_precision, recall, precision
    
    def compute_mean_avg_prec(self):
        avg_precisions = []
        for class_index in self._obj_cnt_by_classindex:
            avg_precision, _, _ = self.compute_avg_precision(class_index = class_index)
            avg_precisions.append(avg_precision)
        return np.mean(avg_precisions)
    
    def plot_pre_vs_recall(self,class_index,class_name=None,interpolated=False):
        avg_precision, recall, precision = self.compute_avg_precision(class_index = class_index, interpolated = interpolated)
        
        label = "{0} AP={1:1.2f}".format("Class {}".format(class_index) if class_name is None else class_name, avg_precision)
        plt.plot(recall, precision, label = label)
        if interpolated:
            plt.title("Precision (Interpolated) vs. Recall")
        else:
            plt.title("Precision vs. Recall")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend()
        plt.show()
        plt.clf()


    def plot_avg_precisions(self, class_index_to_name):
        labels = [ class_index_to_name[class_index] for class_index in self._obj_cnt_by_classindex ]
        avg_precisions = []
        for class_index in self._obj_cnt_by_classindex:
            avg_precision, _, _ = self.compute_avg_precision(class_index = class_index)
            avg_precisions.append(avg_precision)
        sorted_results = sorted(zip(labels, avg_precisions), reverse = True, key = lambda pair: pair[0])
        labels, avg_precisions = zip(*sorted_results) 
        avg_precisions = np.array(avg_precisions) * 100.0 
        plt.clf()
        plt.xlim([0, 100])
        plt.barh(labels, avg_precisions)
        plt.title("Model Performance")
        plt.xlabel("Average Precision (%)")
        for index, value in enumerate(avg_precisions):
            plt.text(value, index, "%1.2f" % value)
        plt.show()

    def print_avg_precisions(self, class_index_to_name):
        labels = [ class_index_to_name[class_index] for class_index in self._obj_cnt_by_classindex ]
        avg_precisions = []
        for class_index in self._obj_cnt_by_classindex:
            avg_precision, _, _ = self.compute_avg_precision(class_index = class_index)
            avg_precisions.append(avg_precision)
        sorted_results = sorted(zip(labels, avg_precisions), reverse = True, key = lambda pair: pair[1])
        _, avg_precisions = zip(*sorted_results)
        label_width = max([ len(label) for label in labels ])   
        print("Average Precisions")
        print(" ")
        for (label, avg_precision) in sorted_results:
            print("%s: %1.2f%%" % (label.ljust(label_width), avg_precision * 100.0))
        print(" ")

