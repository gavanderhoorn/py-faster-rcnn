# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Bharath Hariharan
# --------------------------------------------------------

import xml.etree.ElementTree as ET
import os
import cPickle
import numpy as np
import copy
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
	"""rec, prec, ap = acp_evalv2(detpath,
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
				#TODO what if target object and untarget object are overlaped?
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








# Deprecated performance calculation per image
def apc_evalv2_bak(detpath,
		annopath,
		imagesetfile,
		classname,
		cachedir,
		ovthresh=0.5):
	"""rec, prec, ap = acp_evalv2(detpath,
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
	class_recs = {}
	npos = 0
	negative_gt = [] # negative ground truth per image
	for imagename in imagenames:
		R = [obj for obj in recs[imagename] if obj['name'] == classname]
		if R  == []:
			negative_gt.append(imagename)
		bbox = np.array([x['bbox'] for x in R])
		npos = npos + len(bbox)
		det = [False] * len(R)
		class_recs[imagename] = {'bbox': bbox,
					'det': det}

	# read dets
	detfile = detpath.format(classname)
	with open(detfile, 'r') as f:
		lines = f.readlines()

	splitlines = [x.strip().split(' ') for x in lines]
	# Filenames are split on ' '. In the image filenames there is already a space, therefore use the 2nd space as split
	image_ids = [x[0] for x in splitlines]
	confidence = np.array([float(x[1]) for x in splitlines])
	BB = np.array([[float(z) for z in x[2:]] for x in splitlines])
	
	# filter by confidence
	thresh_range = np.arange(0.,1.,0.05)
	tps = np.zeros(len(thresh_range))
	fps = np.zeros(len(thresh_range))
	tns = np.zeros(len(thresh_range))
	tps_im = np.zeros(len(thresh_range))
	fps_im = np.zeros(len(thresh_range))
	fns_im = np.zeros(len(thresh_range))
	for t, thresh in enumerate(thresh_range):
		keep_ind = np.where(confidence > thresh)[0]
		if len(keep_ind) > 0:
			_BB = BB[keep_ind, :]
			_image_ids = [image_ids[x] for x in keep_ind]
			_class_recs = copy.deepcopy(class_recs)

			# find true negative predictions and count tn
			negative_preds = set(imagenames) - set(_image_ids) # negative predictions per image
			true_negatives = set(negative_gt).intersection(negative_preds)
			tn = len(true_negatives)

			# go down dets and mark TPs and FPs per bbox and tp_im, fp_im per image
			#nd = len(_image_ids)
			tp = 0.
			fp = 0.
			tp_im = 0.
			fp_im = 0.
			fn_im = 0.
			positive_imgs = set(_image_ids)
			for image_id in positive_imgs:
				ds = np.where(np.array(_image_ids)==image_id)[0]
				for d in ds:
					R = _class_recs[_image_ids[d]]
					bb = _BB[d, :].astype(float)
					ovmax = -np.inf
					BBGT = R['bbox'].astype(float)
					
					if BBGT.size > 0:
						ovmax, jmax = maxOverlaps(BBGT, bb)
			
					if ovmax > ovthresh:
						if not R['det'][jmax]:
							tp += 1.
							R['det'][jmax] = 1
						else:
							fp += 1.
					else:
						fp += 1.
				# tp_im is number of correctly detected images
				det = _class_recs[image_id]['det']
				if len(ds) > sum(det):
					fp_im += 1.
				elif len(ds) == sum(det) == len(det):
					tp_im += 1.
				elif sum(det) < len(det):
					fn_im += 1.

		else: # no detection left after filtering
			tp = 0.
			fp = -1.
			tp_im = 0.
			fp_im = -1.
		# add tp, fp and tn for t-th thresh
		tps[t] = tp
		fps[t] = fp
		tns[t] = tn
		# add tp_im, fp_im, tn_im and fn_im for t-th thresh
		tps_im[t] = tp_im
		fps_im[t] = fp_im
		fns_im[t] = fn_im

	# compute fn and precision recall
	fns = npos - tps
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

	# comput tpr and fpr per image
	tpr_im = [ tp_im / np.maximum(float(tp_im + fp_im), np.finfo(np.float64).eps) if fp_im != -1 else 0. for (tp_im,fp_im) in zip(tps_im,fps_im)]
	fpr_im = [fp_im / np.maximum(float(fp_im + tn), np.finfo(np.float64).eps) if fp_im != -1 else 1. for (fp_im,tn) in zip(fps_im,tns)]
	rocauc_im = apc_auc(fpr_im, tpr_im, 'roc')


	return rec, prec, prauc, tpr_im, fpr_im, rocauc_im





# Deprecated tn measures
# Problem: easily more 1000 true negatives
def computeTN_bak(detpath,
		annopath,
		imagesetfile,
		classname,
		ovthresh=0.5):
	"""A function computing True Negative by comparing 
	the background detected and GroundTruth"""

	# first load gt
	# read list of images
	with open(imagesetfile, 'r') as f:
		lines = f.readlines()
	imagenames = [x.strip() for x in lines]

	# load annots
	recs = {}
	for i, imagename in enumerate(imagenames):
		recs[imagename] = parse_rec(annopath.format(imagename))

	### Using background to calculate true negatives and false negatives
	if classname == '__background__':
		class_recs = {}
		nneg = 0
		for imagename in imagenames:
			R = [obj for obj in recs[imagename]]		# Read all objects in this image
			bbox = np.array([x['bbox'] for x in R])
			nneg = nneg + len(bbox)
			det = [False] * len(R)
			class_recs[imagename] = {'bbox': bbox,
					'det': det}
	
			# read dets
		detfile = detpath.format(classname)
		with open(detfile, 'r') as f:
			lines = f.readlines()
	
		splitlines = [x.strip().split(' ') for x in lines]
		# Filenames are split on ' '. In the image filenames there is already a space, therefore use the 2nd space as split
		image_ids = [x[0] for x in splitlines]
		confidence = np.array([float(x[1]) for x in splitlines])
		BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

		# Sort by confidence
		if len(confidence) > 0:
			sorted_ind = np.argsort(-confidence)
			sorted_scores = np.sort(-confidence)
			BB = BB[sorted_ind, :]
			image_ids = [image_ids[x] for x in sorted_ind]
	
			# go down dets and mark TNs and FNs
			nd = len(image_ids)
			tn = np.zeros(nd)
			for d in range(nd):
				R = class_recs[image_ids[d]]
				bb = BB[d, :].astype(float)
				ovmax = -np.inf
				BBGT = R['bbox'].astype(float)
				
				ovmax, jmax = maxOverlaps(BBGT, bb)
		
				if ovmax < ovthresh:
					if not R['det'][jmax]:
						tn[d] = 1.
						R['det'][jmax] = 1
			tn = np.cumsum(tn)
		else:
			tn = [0]
		
		return tn[-1]
