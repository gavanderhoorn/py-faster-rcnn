#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
import numpy as np
import caffe
import cv2
import argparse
import os
from classes import classes_short
import operator
import copy
import cPickle as pkl
import matplotlib.pyplot as plt

# set class labels
_classes = copy.deepcopy(classes_short)
_classes["__background__"] = 0
CLASSES = [k for k, v in sorted(_classes.items(), key=operator.itemgetter(1))]

def generateHeatMap(net, image, box, label, mask_size=(5,5), mask_stride=5):
	scores, _ = im_detect(net, image, np.array([box]))
	label = np.argmax(scores[0])

	complete_score = scores[0, label]
	print label, complete_score
	heatmap = np.zeros((image.shape[0], image.shape[1]))

	x_range = np.arange(box[0], box[2] - mask_size[1], mask_stride)
	y_range = np.arange(box[1], box[3] - mask_size[0], mask_stride)
	count = 0
	for y in y_range:
		for x in x_range:
			masked = image.copy()
			cv2.rectangle(masked, (x, y), (x + mask_size[1], y + mask_size[0]), (0, 0, 0), -1)

			scores, _    = im_detect(net, masked, np.array([box]))
			masked_score = scores[0, label]
			delta_score  = complete_score - masked_score

			heatmap[y:y + mask_size[0], x:x + mask_size[1]] += np.ones(mask_size) * delta_score
			cv2.rectangle(masked, (x, y), (x + mask_size[1], y + mask_size[0]), (0, 0, 0), -1)

			count += 1
			perc = float(count) / (len(x_range) * len(y_range))
			print perc

			#cv2.imshow("Image", masked)
			#cv2.waitKey()

	negative_heatmap = heatmap.copy()
	negative_heatmap[heatmap > 0] = 0
	negative_heatmap *= -1.0
	negative_heatmap = cv2.normalize(negative_heatmap, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

	positive_heatmap = heatmap.copy()
	positive_heatmap[heatmap < 0] = 0
	positive_heatmap = cv2.normalize(positive_heatmap, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

	heatmap = np.zeros(image.shape).astype(np.uint8)
	heatmap[:,:,1] = positive_heatmap
	heatmap[:,:,2] = negative_heatmap

	overlay = cv2.addWeighted(image, 0.3, heatmap, 0.7, 0.0)

	cv2.imshow("Heatmap", heatmap)
	cv2.imshow("Overlay", overlay)
	cv2.waitKey()

def parse_args():
	"""Parse input arguments."""
	parser = argparse.ArgumentParser(description="Faster R-CNN demo")
	parser.add_argument("input_image",   help="Input image.")
	parser.add_argument("input_objects", help="Input objects (cPickle object, list of (label, [x1, y1, x2, y2])).")
	parser.add_argument("--cpu",         help="Use CPU mode", dest="cpu_mode", action="store_true")

	args = parser.parse_args()

	return args

if __name__ == "__main__":
	cfg.TEST.HAS_RPN  = False # Disable RPN (we are supplying a box)
	cfg.TEST.BBOX_REG = False # Disable bounding box regression, we are assuming user supplied boxes

	args = parse_args()

	cv2.namedWindow("Heatmap", cv2.WINDOW_NORMAL)
	cv2.namedWindow("Overlay", cv2.WINDOW_NORMAL)

	prototxt   = os.path.join(cfg.MODELS_DIR, "VGG16", "fast_rcnn", "test.prototxt")
	caffemodel = os.path.join(cfg.DATA_DIR, "faster_rcnn_models", "apc.caffemodel")

	print prototxt
	print caffemodel

	if not os.path.isfile(caffemodel):
		raise IOError(("{:s} not found.").format(caffemodel))
	if not os.path.isfile(args.input_image):
		raise IOError(("{:s} not found.").format(args.input_image))
	if not os.path.isfile(args.input_objects):
		raise IOError(("{:s} not found.").format(args.input_objects))

	if args.cpu_mode:
		caffe.set_mode_cpu()
	else:
		caffe.set_mode_gpu()

	net = caffe.Net(prototxt, caffemodel, caffe.TEST)

	print "\n\nLoaded network {:s}".format(caffemodel)

	# Warmup on a dummy image
	im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
	for i in xrange(2):
		im_detect(net, im, np.array([[0, 0, 300, 500]]))

	im      = cv2.imread(args.input_image)
	objects = pkl.load(open(args.input_objects, "rb"))

	for label, box in objects:
		generateHeatMap(net, im, box, label)
