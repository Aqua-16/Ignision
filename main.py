import argparse
import numpy as np
import os
import random
from tqdm import tqdm
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

from stats import train_statistics, PRCCalc
import image_n_annotation_loader as dataset
import vgg16
import faster_rcnn
import utils
import anchors
import image

def _get_sample_rpn_minibatch(rpn_map,object_indices,background_indices,mini_size):
    # This selects a subset of anchors for training and returns a copy of the ground truth RPN map with only those anchors marked for training

    assert rpn_map.shape[0] == 1, "Batch size must be 1"
    assert len(object_indices) == 1, "Batch size must be 1"
    assert len(background_indices) == 1, "Batch size must be 1"
    positive_anchors = object_indices[0]
    negative_anchors = background_indices[0]

    assert len(positive_anchors) + len(negative_anchors) >= mini_size, f"Image has insufficient anchors for RPN minibatch size"
    assert len(positive_anchors) > 0, "Image does not have any positive anchors"
    assert mini_size % 2 == 0, "RPN minibatch size must be even"

    num_pos = len(positive_anchors)
    num_neg = len(negative_anchors)
    num_pos_samples = min(mini_size//2, num_pos) # At least half of the minibatch should comprise of positive samples
    num_neg_samples = mini_size - num_pos_samples
    pos_indices = random.sample(range(num_pos), num_pos_samples)
    neg_indices = random.sample(range(num_neg), num_neg_samples)

    positive_anchors = positive_anchors[pos_indices]
    negative_anchors = negative_anchors[neg_indices]
    train_anchors = np.concatenate([positive_anchors, negative_anchors])
    batch_idx = np.zeros(len(train_anchors), dtype = int)
    train_indices = (batch_idx, train_anchors[:,0], train_anchors[:,1], train_anchors[:,2], 0)

    rpn_minibatch = rpn_map.copy()
    rpn_minibatch[:,:,:,:,0] = 0
    rpn_minibatch[train_indices] = 1

    return rpn_minibatch

def _convert_sample_to_model_input(sample,mode):
    gt_box_corners = np.array([ box.corners for box in sample.gt_boxes ]).astype(np.float32) 
    gt_box_class_idxs = np.array([ box.class_index for box in sample.gt_boxes ]).astype(np.int32) 
    gt_box_corners = np.expand_dims(gt_box_corners, axis = 0)
    gt_box_class_idxs = np.expand_dims(gt_box_class_idxs, axis = 0)

    image_data = np.expand_dims(sample.image_data, axis = 0)
    image_shape_map = np.array([ [ image_data.shape[1], image_data.shape[2], image_data.shape[3] ] ])
    anchor_map = np.expand_dims(sample.anchor_map, axis = 0)
    gt_rpn_map = np.expand_dims(sample.gt_rpn_map, axis = 0)
    gt_rpn_object_indices = [ sample.gt_rpn_object_indices ]
    gt_rpn_background_indices = [ sample.gt_rpn_background_indices ]

    gt_rpn_minibatch =  _get_sample_rpn_minibatch(
      rpn_map = gt_rpn_map,
      object_indices = gt_rpn_object_indices,
      background_indices = gt_rpn_background_indices,
      mini_size = 256
    )

    if mode == "train":
      x = [ image_data, anchor_map, gt_rpn_minibatch, gt_box_class_idxs, gt_box_corners ]
    else: # prediction
      x = [ image_data, anchor_map, anchor_valid_map ]

    return x, image_data, gt_rpn_minibatch # Returned like so for convenience

def _predict(model,url,show_image,output_path):
    image_data, image, _ = load_image(url = url)
    anchor_map = anchors.generate_anchor_map(image_shape = image_data.shape, feature_scale = 16)
    anchor_map = np.expand_dims(anchor_map,axis = 0)
    image_data = np.expand_dims(image_data,axis = 0) # Converting to Batch size of 1
    x = [ image_data, anchor_map ]
    scored_bboxes = model.predict_on_batch(x = x, threshold = 0.7)
    image.show_detections(
        out_path = output_path,
        image = image,
        scored_boxes_class_idx = scored_bboxes,
        class_idx_name = dataset.Dataset.class_index_to_name
    )


if __name__ == '__main__':
    # User Interface
    parser = argparse.ArgumentParser("Ignision")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--train", action = "store_true", help = "Train model")
    parser.add_argument("--save-best-to", metavar = "file", action = "store", help = "Save best weights (highest mean average precision) to file")
    parser.add_argument("--checkpoint-dir", metavar = "dir", action = "store", help = "Save checkpoints after each epoch to the given directory")
    group.add_argument("--eval", action = "store_true", help = "Evaluate model")
    group.add_argument("--predict", metavar = "url", action = "store", type = str, help = "Run inference on image and display detected boxes")
    group.add_argument("--predict-to-file", metavar = "url", action = "store", type = str, help = "Run inference on image and render detected boxes to 'pred.jpg'")
    parser.add_argument("--load-from", metavar = "file", action = "store", help = "Load initial model weights from file")
    parser.add_argument("--train-split", metavar = "name", action = "store", default = "trainval", help = "Dataset split to use for training")
    parser.add_argument("--eval-split", metavar = "name", action = "store", default = "test", help = "Dataset split to use for evaluation")
    parser.add_argument("--plot", action = "store_true", help = "Plots the average precision after evaluation (use with --train or --eval)")
    parser.add_argument("--epochs", metavar = "count", type = int, action = "store", default = 1, help = "Number of epochs to train for")
    parser.add_argument("--learning-rate", metavar = "value", type = float, action = "store", default = 1e-3, help = "Learning rate")
    parser.add_argument("--logits", action = "store_true", help = "Do not apply softmax to detector class output and compute loss from logits directly")
    parser.add_argument("--weight-decay", metavar = "value", type = float, action = "store", default = 5e-4, help = "Weight decay")
    parser.add_argument("--dropout", metavar = "probability", type = float, action = "store", default = 0.0, help = "Dropout probability after each of the two fully-connected detector layers")
    options = parser.parse_args()

    # Run-time environment
    cuda_available = tf.test.is_built_with_cuda()
    gpu_available = tf.test.is_gpu_available(cuda_only = False, min_cuda_compute_capability = None)
    print("CUDA Available : %s" % ("yes" if cuda_available else "no"))
    print("GPU Available  : %s" % ("yes" if gpu_available else "no"))
    print("Eager Execution: %s" % ("yes" if tf.executing_eagerly() else "no"))

    model = faster_rcnn.FasterRCNN(
        num_classes = dataset.Dataset.num_classes,
        actclassoutputs = not options.logits,
        l2 = 0.5 * options.weight_decay,
        dropout_prob = options.dropout
    )

    model.build(
        input_shape = [
            (1, None, None, 3),     # input_image: (1, height_pixels, width_pixels, 3)
            (1, None, None, 9 * 4), # anchor_map: (1, height, width, num_anchors * 4)
            (1, None, None, 9, 6),  # gt_rpn_map: (1, height, width, num_anchors, 6)
            (1, None),              # gt_box_class_idxs_map: (1, num_gt_boxes)
            (1, None, 4)            # gt_box_corners_map: (1, num_gt_boxes, 4)
        ]
    )
    optimizer = Adam(learning_rate = options.learning_rate, beta_1 = 0.9, beta_2 = 0.999)
    model.compile(optimizer = optimizer)
    
    if options.load_from:
        model.load_weights(filepath = options.load_from, by_name = True)
        print("Loaded initial weights from '%s'" % options.load_from)
    else:
        model.load_imgnet_wts()
        print("Initialized VGG-16 layers to Keras ImageNet weights")

    if options.train:
        # Perform Model Training
    elif options.eval:
        # Perform Model Evaluation
    elif options.predict:
        _predict(model = model, url = options.predict, show_image = True, output_path = None)
    elif options.predict_to_file:
        _predict(model = model, url = options.predict_to_file, show_image = False, output_path = "pred.jpg")