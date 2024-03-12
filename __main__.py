import argparse
import numpy as np
import os
import random
from tqdm import tqdm
from tensorflow.keras.optimizers import SGD
import tensorflow as tf

import warnings
warnings.filterwarnings("ignore")


from .stats import train_statistics, PRCCalc
from . import image_n_annotation_loader as dataset
from . import vgg16
from . import faster_rcnn
from . import utils
from . import anchors
from . import image as vis

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
    assert num_neg > num_neg_samples, f"{num_neg} is smaller than {num_neg_samples}."
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
    print(gt_rpn_background_indices)
    gt_rpn_minibatch =  _get_sample_rpn_minibatch(
        rpn_map = gt_rpn_map,
        object_indices = gt_rpn_object_indices,
        background_indices = gt_rpn_background_indices,
        mini_size = 8
    )

    if mode == "train":
        x = [ image_data, anchor_map, gt_rpn_minibatch, gt_box_class_idxs, gt_box_corners ]
    else: # prediction
        x = [ image_data, anchor_map ]

    return x, image_data, gt_rpn_minibatch # Returned like so for convenience

def evaluate(model,eval_data=None,num_samples = None, plot=False,print_AP=False):
    prc=PRCCalc()
    i=0
    print(f"Evaluating {eval_data.split} ...")
    for sample in tqdm(iterable=iter(eval_data), total=num_samples):
        x, _ , _ = _convert_sample_to_model_input(sample = sample, mode = "infer")
        scored_boxes_by_class_index = model.predict_on_batch(x = x, score_threshold = 0.1)# lower threshold score for evaluation
        prc.add_img_result(
            scored_boxes_by_class_index = scored_boxes_by_class_index,
            gt_boxes = sample.gt_boxes
        )
        i+=1
        if i>=num_samples:
            break
    if print_AP:
        prc.print_avg_precisions(class_index_to_name=dataset.Dataset.class_index_to_name)
    mAP=100.0*prc.compute_mean_avg_prec()
    print(f"Mean Average Precision = {mAP:1.2f}%")
    if plot:
        prc.plot_avg_precisions(class_index_to_name=dataset.Dataset.class_index_to_name)
    return mAP

def train(model):
    print("Starting Model Training....")
    print("||----------------------------------------------||")
    print(f"Epochs                    : {options.epochs}")
    print(f"Learning Rate             : {options.learning_rate}")
    print(f"Weight decay              : {options.weight_decay}")
    print(f"Dropout                   : {options.dropout}")

    training_data = dataset.Dataset(shuffling = False)
    eval_data = dataset.Dataset(augmenting = False, shuffling = False)

    if options.checkpoint_dir and not os.path.exists(options.checkpoint_dir):
        os.makedirs(options.checkpoint_dir)
    if options.save_best_to:
        best_weights_tracker = utils.BestWeightsTracker(filepath = options.save_best_to)

    for epoch in (1,1+options.epochs):
        print(f"Epoch       {epoch}/{options.epochs}")
        stats = train_statistics()
        progbar = tqdm(iterable = iter(training_data), total = training_data.num_samples, postfix = stats.progress_bar_postfix())
        for sample in progbar:
            print(sample)
            x, _, gt_rpn_minibatch = _convert_sample_to_model_input(sample = sample, mode = "train")
            losses = model.train_on_batch(x = x, y = gt_rpn_minibatch, return_dict = True)
            stats.during_training_step(losses = losses)
            progbar.set_postfix(stats.progress_bar_postfix())
        mAP = evaluate( # Mean Average Precision
                model=model,
                eval_data=eval_data,
                num_samples=10,
                plot=False,
                print_AP=False
        ) 
        if options.checkpoint_dir:
            checkpoint_file = os.path.join(options.checkpoint_dir, "checkpoint-epoch-%d-mAP-%1.1f.h5" % (epoch, mAP))
            model.save_weights(filepath = checkpoint_file, overwrite = True, save_format = "h5")
            print("Saved model checkpoint to '%s'" % checkpoint_file)

        if options.save_best_to:
            best_weights_tracker.on_epoch_end(model = model, mAP = mAP)

    if options.save_best_to:
        best_weights_tracker.restore_and_save_best_weights(model = model)

    print("Evaluating the final model...")

    evaluate(
        model = model,
        eval_data = eval_data,
        num_samples = eval_data.num_samples,
        plot = options.plot,
        print_AP = True
    )

def _predict(model,url,output_path):
    image_data, image, _ = vis.load_image(path = url)
    anchor_map = anchors.generate_anchor_map(image_shape = image_data.shape, feature_scale = 16)
    anchor_map = np.expand_dims(anchor_map,axis = 0)
    image_data = np.expand_dims(image_data,axis = 0) # Converting to Batch size of 1
    x = [ image_data, anchor_map ]
    scored_bboxes = model.predict_on_batch(x = x, threshold = 0.7)
    vis.show_detections(
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
    optimizer = SGD(learning_rate = options.learning_rate, momentum = 0.9)
    model.compile(optimizer = optimizer)
    
    if options.load_from:
        model.load_weights(filepath = options.load_from, by_name = True)
        print("Loaded initial weights from '%s'" % options.load_from)
    else:
        model.load_imgnet_wts()
        print("Initialized VGG-16 layers to Keras ImageNet weights")

    if options.train:
        train(model=model)
    elif options.eval:
        evaluate(model = model, plot = options.plot, print_AP = True)
    elif options.predict:
        _predict(model = model, url = options.predict, output_path = None)
    elif options.predict_to_file:
        _predict(model = model, url = options.predict_to_file, output_path = "pred.jpg")
