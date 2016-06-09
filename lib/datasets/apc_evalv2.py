# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Modified from PASCAL VOC evaluation code written by Bharath Hariharan
# --------------------------------------------------------

import xml.etree.ElementTree as ET
import os
import cPickle
import numpy as np
import copy
import itertools
import operator
from sklearn.metrics import auc

# 20 thresholds varying from 0 to 0.95 with interval 0.05
thresh_range = np.arange(0.,1.,0.05)

def parse_rec(filename):
    """ Parse a DR APC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text),
                              int(bbox.find('ymin').text),
                              int(bbox.find('xmax').text),
                              int(bbox.find('ymax').text)]
        objects.append(obj_struct)

    return objects

def apc_auc(x, y, curve='pr'):
	"""
	Compute AUC given precision and recall at different threshholds.
	"""
	# first append sentinel values at the end
	if curve == 'pr':
		mx = np.concatenate(([1.], x, [0.]))
		my = np.concatenate(([0.], y, [1.]))
	elif curve == 'roc':
		mx = np.concatenate(([1.], x, [0.]))
		my = np.concatenate(([1.], y, [0.]))
	else:
		'Error: Invalid curve given (has to be pr or roc) in apc_evalv2.py'
	
	# compute the Area under PR Curve 
	return auc(mx, my)


def maxOverlaps(BBGT, bb):
	"""compute max overlaps between detected bb and BBGTs"""
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

	overlaps = inters / uni
	ovmax = np.max(overlaps)
	jmax = np.argmax(overlaps)

	return ovmax, jmax


def apc_evalv2(detpath,
		annopath,
		imagesetfile,
		classname,
		cachedir,
		ovthresh=0.5):
	"""rec, prec, prauc, tpr, fpr, rocauc = acp_evalv2(detpath,
								annopath,
								imagesetfile,
								classname,
								[ovthresh])

	Top level function that does a PASCAL VOC like evaluation per bbox.

	detpath: Path to detections
		detpath.format(classname) should produce the detection results file.
	annopath: Path to annotations
		annopath.format(imagename) should be the xml annotations file.
	imagesetfile: Text file containing the list of images, one image per line.
	classname: Category name (duh)
	cachedir: Directory for caching the annotations
	[ovthresh]: Overlap threshold (default = 0.5)
	"""
	# assumes detections are in detpath.format(classname)
	# assumes annotations are in annopath.format(imagename)
	# assumes imagesetfile is a text file with each line an image name
	# cachedir caches the annotations in a pickle file

	# read list of images
	with open(imagesetfile, 'r') as f:
		lines = f.readlines()
	imagenames = [x.strip() for x in lines]

	# load annots
	recs = {}
	for i, imagename in enumerate(imagenames):
		recs[imagename] = parse_rec(annopath.format(imagename))
		if i % 100 == 0:
			print 'Reading annotation for {:d}/{:d}'.format(
					i + 1, len(imagenames))


	### Calculate rec, prec, and auc for pr curve fn by comparing BBGT amd BB detected per bbox
	# extract gt objects for this class
	class_recs = {}			# ground truth for target class
	class_recs_neg = {}		# ground truth for the other classes (negatives)
	npos = 0				# number of positive objects to be detected
	nneg = 0				# number of negative objects to be rejected
	for imagename in imagenames:
		# read ground truth for current class
		R = [obj for obj in recs[imagename] if obj['name'] == classname]
		bbox = np.array([x['bbox'] for x in R])
		npos = npos + len(bbox)
		det = [False] * len(R)
		class_recs[imagename] = {'bbox': bbox,
					'det': det}
		# read ground truth for the other classes
		R_neg = [obj for obj in recs[imagename] if obj['name'] != classname]
		bbox_neg = np.array([x['bbox'] for x in R_neg])
		nneg = nneg + len(bbox_neg)
		class_recs_neg[imagename] = {'bbox': bbox_neg}

	# read detections
	detfile = detpath.format(classname)
	with open(detfile, 'r') as f:
		lines = f.readlines()

	splitlines = [x.strip().split(' ') for x in lines]
	# Filenames are split on ' '. In the image filenames there is already a space, therefore use the 2nd space as split
	image_ids = [x[0] for x in splitlines]
	confidence = np.array([float(x[1]) for x in splitlines])
	BB = np.array([[float(z) for z in x[2:]] for x in splitlines])
	
	# compute for each threshold from 0 to 0.95 with interval 0.05
	tps = np.zeros(len(thresh_range))		# correct detections
	fps = np.zeros(len(thresh_range))		# target objects are wrongly rejected
	tns = np.zeros(len(thresh_range))		# untarget objects are correctly rejected
	for t, thresh in enumerate(thresh_range):
		# filter by confidence
		keep_ind = np.where(confidence > thresh)[0]
		if len(keep_ind) > 0:	# if there is detections left
			_BB = BB[keep_ind, :]
			_image_ids = [image_ids[x] for x in keep_ind]
			_class_recs = copy.deepcopy(class_recs)
			_class_recs_neg = copy.deepcopy(class_recs_neg)

			# go down dets and mark TPs, FPs and TNs per bbox
			nd = len(_image_ids)
			tn = nneg
			tp = 0
			fp = 0
			for d in xrange(0, nd):
				R = _class_recs[_image_ids[d]]
				bb = _BB[d, :].astype(float)
				ovmax = -np.inf
				BBGT = R['bbox'].astype(float)
				
				if BBGT.size > 0:
					ovmax, jmax = maxOverlaps(BBGT, bb)
		
				if ovmax > ovthresh:
					if not R['det'][jmax]:
						tp += 1
						R['det'][jmax] = 1
					else:
						fp += 1
				else:
					fp += 1
				# if untargeted objects are wrongly accepted
				R_neg = _class_recs_neg[_image_ids[d]]
				ovmax_neg = -np.inf
				BBGT_neg = R_neg['bbox'].astype(float)
				if BBGT_neg.size > 0:
					ovmax_neg, _ = maxOverlaps(BBGT_neg, bb)
				#TODO what if target object and untarget object are overlaped? IGNORE
				if ovmax_neg > ovthresh and ovmax < ovthresh:
					tn -= 1

		else: # no detection left after filtering
			tp = 0
			fp = -1
			tn = nneg
		# add tp, fp and tn for t-th thresh
		tps[t] = tp
		fps[t] = fp
		tns[t] = tn

	# compute fn and recall
	fns = npos - tps		# number of objects not detected
	if npos > 0:
		rec = tps / float(npos)
	else:
		rec = np.ones(len(tps))
	# compute tpr and fpr per bbox
	tpr = rec
	fpr = [fp / np.maximum(float(fp + tn), np.finfo(np.float64).eps) if fp != -1 else 0. for (fp,tn) in zip(fps,tns)]
	
	# avoid divide by zero in case the first detection matches a difficult
	# ground truth, prec = 1 and recall = 0 if no detection kept
	prec = [tp / np.maximum(float(tp + fp), np.finfo(np.float64).eps) if fp != -1 else 1. for (tp,fp) in zip(tps,fps)]
	prauc = apc_auc(rec, prec, 'pr')
	rocauc = apc_auc(fpr, tpr, 'roc')

	return rec, prec, prauc, tpr, fpr, rocauc








def apc_evalv3(detpath,
		annopath,
		imagesetfile,
		classname,
		cachedir,
		ovthresh=0.5):
	"""tp, fp, tn, fn = acp_evalv3(detpath,
						annopath,
						imagesetfile,
						classname,
						cachedir,
						[ovthresh])

	Top level function that evaluates detection accuracy per class.

	detpath: Path to detections
		detpath.format(classname) should produce the detection results file.
	annopath: Path to annotations
		annopath.format(imagename) should be the xml annotations file.
	imagesetfile: Text file containing the list of images, one image per line.
	classname: Category name (duh)
	cachedir: Directory for caching the annotations
	[ovthresh]: Overlap threshold (default = 0.5)
	"""
	# assumes detections are in detpath.format(classname)
	# assumes annotations are in annopath.format(imagename)
	# assumes imagesetfile is a text file with each line an image name
	# cachedir caches the annotations in a pickle file

	# read list of images
	with open(imagesetfile, 'r') as f:
		lines = f.readlines()
	imagenames = [x.strip() for x in lines]

	# load annots
	recs = {}
	for i, imagename in enumerate(imagenames):
		recs[imagename] = parse_rec(annopath.format(imagename))
		if i % 100 == 0:
			print 'Reading annotation for {:d}/{:d}'.format(
					i + 1, len(imagenames))

	# extract gt objects for this class
	class_recs = {}
	npos = 0
	neg_imgs = []
	for imagename in imagenames:
		# TODO select obj only if it is obtainable
		R = [obj for obj in recs[imagename] if obj['name'] == classname]
		bbox = np.array([x['bbox'] for x in R])
		det = [False] * len(R)
		class_recs[imagename] = {'bbox': bbox,
					'det': det}
		# count each image only once for number of positives
		if len(R) > 0:
			npos += 1
		else:
			neg_imgs.append(imagename)

	# read dets
	detfile = detpath.format(classname)
	with open(detfile, 'r') as f:
		lines = f.readlines()

	splitlines = [x.strip().split(' ') for x in lines]
	# Filenames are split on ' '. In the image filenames there is already a space, therefore use the 2nd space as split
	image_ids	= np.array([x[0] for x in splitlines])
	confidence	= np.array([float(x[1]) for x in splitlines])
	BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

	# compute FALSE NEGATIVES, extended later
	neg_dets	= set(imagenames) - set(image_ids)
	tn			= len(set(neg_dets) & set(neg_imgs))
	
	# zip image_ids, condidence and BBe
	all_boxes = zip(image_ids, confidence, BB)

	# group by image_id and pick the highest confidence
	tp = 0
	fp = 0
	for key,group in itertools.groupby(all_boxes, operator.itemgetter(0)):
		# sort by confidence
		proposals	= list(group)
		proposals	= sorted(proposals, key=lambda x: x[1])
		# pick the highest confidence
		best_det	= proposals[-1]

		# retrieve detection bb and gound truth BBGT
		_image_id	= best_det[0]
		_confidence	= best_det[1]
		_bb			= best_det[2].astype(float)
		_R			= class_recs[_image_id]
		_BBGT		= _R['bbox'].astype(float)

		# filter by a threshold #TODO replace by class specific threshold
		if best_det[1] < 0.5 and _BBGT.size == 0:
			# add as TRUE NEGATIVE
			tn += 1
			continue
		elif best_det[1] >= 0.5 and _BBGT.size == 0:
			# add as FALSE NAGATIVE
			fp += 1
			continue
		elif best_det[1] < 0.5:
			# consider as FALSE NEGATIVE
			continue

		# calculate TRUE POSITIVES and FALSE POSITIVES
		ovmax = -np.inf
		ovmax, jmax = maxOverlaps(_BBGT, _bb)

		if ovmax > ovthresh:
			tp += 1
		else:
			fp += 1

	# compute FASLSE POSTIVES
	fn = npos - tp

	# compute precision, recall, tpr and fpr
	if npos > 0:
		rec = tp / float(npos)
	else:
		rec = 1
	# compute tpr and fpr
	tpr = rec
	fpr = fp / np.maximum(float(fp + tn), np.finfo(np.float64).eps)
	
	# avoid divide by zero in case the first detection matches a difficult
	# ground truth, prec = 1 and recall = 0 if no detection kept
	prec = tp / np.maximum(float(tp + fp), np.finfo(np.float64).eps)

	print "(tp, fp, tn, fn) is ({},{},{},{})".format(tp, fp, tn, fn)

	return tp, fp, tn, fn


