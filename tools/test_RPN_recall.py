
"""Test Recall vs RPN threshold"""

from fast_rcnn.config import cfg, get_output_dir
from fast_rcnn.bbox_transform import clip_boxes, bbox_transform_inv
import argparse
from utils.timer import Timer
import numpy as np
import cv2
import caffe
from fast_rcnn.nms_wrapper import nms
from utils.blob import im_list_to_blob
import os
from datasets.apc_eval parse_rec

def get_bb_GT(innopath, imagesetfile):
	# load annotations
	recs = {}
	for i, imagename in enumerate(imagenames):
		recs[imagename] = parse_rec(annopath.format(imagename))

	# extract gt objects of the whole image
	for imagename in imagenames:
		R = [obj for obj in recs[imagename]]
		bbox = np.array([x['bbox'] for x in R])
	
	return bbox

def get_bb_RPN():
"""Get the bounding boxes of the RPN on all images"""
	# all detections are collected into:
	#    all_boxes[class] = N x 5 array of detections in
	#    (x1, y1, x2, y2, score)
	#    N is the number of occurences os a certain class in an image

	all_boxes = [[] for _ in xrange(imdb.num_classes)]
	
	scores, boxes = im_detect(net, im, box_proposals=None) 

	# boxes: a N x (4*number_classes) ndarray where N is the number of proposals (300)
	# scores: a N x number_classes ndarray each column represents a class

	# skip j = 0, because it's the background class
	# TODO: are the box proposals different per class??
	for j in range(1, imdb.num_classes):
		cls_boxes = boxes[:, j*4:(j+1)*4] # each class has 4 columns, select the correct columns
		all_boxes[j] = cls_boxes

	return all_boxes

def get_IOU(BBGT, bb):
	# BBGT: bounding box of ground truth
	# bb: bounding box of bb proposal algorithm

	# compute overlaps
	# intersection
	ixmin = np.maximum(BBGT[:, 0], bb[0])
	iymin = np.maximum(BBGT[:, 1], bb[1])
	ixmax = np.minimum(BBGT[:, 2], bb[2])
	iymax = np.minimum(BBGT[:, 3], bb[3])
	iw = np.maximum(ixmax - ixmin + 1., 0.)
	ih = np.maximum(iymax - iymin + 1., 0.)
	inters = iw * ih

	# union
	uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
			(BBGT[:, 2] - BBGT[:, 0] + 1.) *
			(BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

	overlap = inters / uni
	#ovmax = np.max(overlaps)
	#jmax = np.argmax(overlaps)
	return overlap

def test_RPN_recall(net, imdb):
	bb_GT  = get_bb_GT()
	bb_RPN = get_bb_RPN()

	dets = 0

	# read list of test image
	with open(imagesetfile, 'r') as f:
		lines = f.readlines()
	imagenames = [x.strip() for x in lines]


	num_images = len(imdb.image_index)
	# per image
	# per bb GT 
	# check for all bb_RPN if there is enough overlap
	# if there is enough overlap then this is a 'detection' 
