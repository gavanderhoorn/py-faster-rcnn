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
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
import os
import sys
from classes import classes_short
import operator
import copy

# set class labels
_classes = copy.deepcopy(classes_short)
_classes["__background__"] = 0
CLASSES = [k for k, v in sorted(_classes.items(), key=operator.itemgetter(1))]
print(CLASSES)

#CLASSES = ('__background__',
#		'aeroplane', 'bicycle', 'bird', 'boat',
#		'bottle', 'bus', 'car', 'cat', 'chair',
#		'cow', 'diningtable', 'dog', 'horse',
#		'motorbike', 'person', 'pottedplant',
#		'sheep', 'sofa', 'train', 'tvmonitor')
#		   'bottle', 'coffee', 'eraser', 'glucose', 'glue')

NETS = {'vgg16': ('VGG16',
				  'apc_bin.caffemodel'),
		'zf': ('ZF',
				  'ZF_faster_rcnn_final.caffemodel')}


def vis_detections(im, class_name, dets, thresh=0.5):
	"""Draw detected bounding boxes."""
	image = im.copy()
	inds = np.where(dets[:, -1] >= thresh)[0]
	if len(inds) == 0:
		return image

	for i in inds:
		bbox = dets[i, :4]
		score = dets[i, -1]
		#print bbox, score
		cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 4)
		cv2.putText(image, class_name + " %2.2f" % score, (int(bbox[0] + 10), int(bbox[1] + 30)), cv2.FONT_HERSHEY_SIMPLEX, 1, color=(0,0,0), thickness=3, lineType=cv2.LINE_AA)
		cv2.putText(image, class_name + " %2.2f" % score, (int(bbox[0] + 10), int(bbox[1] + 30)), cv2.FONT_HERSHEY_SIMPLEX, 1, color=(255,255,255), thickness=2, lineType=cv2.LINE_AA)


	return image

def demo(net, image_name):
	"""Detect object classes in an image using pre-computed object proposals."""

	# Load the demo image
	im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
	im = cv2.imread(im_file)

	# Detect all object classes and regress object bounds
	timer = Timer()
	timer.tic()
	scores, boxes = im_detect(net, im)
	timer.toc()
	print ('Detection took {:.3f}s for '
		   '{:d} object proposals').format(timer.total_time, boxes.shape[0])

	# Visualize detections for each class
	CONF_THRESH = 0.3
	NMS_THRESH = 0.3
	for cls_ind, cls in enumerate(CLASSES[1:]): 
		cls_ind += 1 # because we skipped background
		cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
		cls_scores = scores[:, cls_ind]
		dets = np.hstack((cls_boxes,
						  cls_scores[:, np.newaxis])).astype(np.float32)
		keep = nms(dets, NMS_THRESH)
		dets = dets[keep, :]
		im = vis_detections(im, cls, dets, thresh=CONF_THRESH)
	cv2.imshow("Image", im)
	if cv2.waitKey() == ord('q'):
		sys.exit()
def parse_args():
	"""Parse input arguments."""
	parser = argparse.ArgumentParser(description='Faster R-CNN demo')
	parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
						default=0, type=int)
	parser.add_argument('--cpu', dest='cpu_mode',
						help='Use CPU mode (overrides --gpu)',
						action='store_true')
	parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
						choices=NETS.keys(), default='vgg16')
	parser.add_argument('--caffemodel', dest='caffemodel', help='Caffemodel to use.',
						default=None)


	args = parser.parse_args()

	return args

if __name__ == '__main__':
	cfg.TEST.HAS_RPN = True  # Use RPN for proposals

	args = parse_args()

	#cv2.namedWindow("Image", cv2.WINDOW_NORMAL)

	prototxt = os.path.join(cfg.MODELS_DIR, NETS[args.demo_net][0],
							'faster_rcnn_end2end', 'test.prototxt')
	if args.caffemodel:
		caffemodel = args.caffemodel
	else:
		caffemodel = os.path.join(cfg.DATA_DIR, 'faster_rcnn_models',
								  NETS[args.demo_net][1])

	#print prototxt
	#print caffemodel

	if not os.path.isfile(caffemodel):
		raise IOError(('{:s} not found.\nDid you run ./data/script/'
					   'fetch_faster_rcnn_models.sh?').format(caffemodel))

	if args.cpu_mode:
		caffe.set_mode_cpu()
	else:
		caffe.set_mode_gpu()
		caffe.set_device(args.gpu_id)
		cfg.GPU_ID = args.gpu_id

	print ("NET creation parameters:")
	print ("prototxt: ")
	print (prototxt)
	print ("caffemodel: ")
	print (caffemodel)
	net = caffe.Net(prototxt, caffemodel, caffe.TEST)

	print ('\n\nLoaded network {:s}'.format(caffemodel))

	# Warmup on a dummy image
	im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
	for i in range(2):
		_, _= im_detect(net, im)

	im_names = [f for f in os.listdir("data/demo/") if f.endswith(".jpg") or f.endswith(".png")]
	for im_name in im_names:
		print ('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
		print ('Demo for data/demo/{}'.format(im_name))
		demo(net, im_name)

	plt.show()
