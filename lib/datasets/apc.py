# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import os
from datasets.imdb import imdb
import datasets.ds_utils as ds_utils
import xml.etree.ElementTree as ET
import numpy as np
import scipy.sparse
import scipy.io as sio
import utils.cython_bbox
import _pickle as cPickle
import subprocess
import uuid
from apc_eval import apc_eval
from apc_evalv2 import apc_evalv2
from fast_rcnn.config import cfg
from classes import classes, classes_short
import operator
import copy

class apc(imdb):
	def __init__(self, image_set, year, devkit_path=None):
		imdb.__init__(self, 'apc_' + year + '_' + image_set)
		self._year = year
		self._image_set = image_set
		self._devkit_path = self._get_default_path() if devkit_path is None \
				else devkit_path
		self._data_path = os.path.join(self._devkit_path, 'APC' + self._year)

		# set class labels
		_classes = copy.deepcopy(classes)
		_classes["__background__"] = 0
		self._classes = [k for k, v in sorted(_classes.items(), key=operator.itemgetter(1))]

		# set short labels
		_classes_short = copy.deepcopy(classes_short)
		_classes_short["__background__"] = 0
		self._classes_short = [k for k, v in sorted(_classes_short.items(), key=operator.itemgetter(1))]

		# set class to indices dict
		self._class_to_ind = _classes
		print self._classes

		self._image_ext = '.png'
		self._image_index = self._load_image_set_index()
		# Default to roidb handler
		self._roidb_handler = self.selective_search_roidb
		self._salt = str(uuid.uuid4())
		self._comp_id = 'comp4'

		# PASCAL specific config options
		self.config = {'cleanup'     : True,
				'use_salt'    : True,
				'use_diff'    : False,
				'python_eval' : True,
				'rpn_file'    : None,
				'min_size'    : 2}

		assert os.path.exists(self._devkit_path), \
				'DRapc path does not exist: {}'.format(self._devkit_path)
		assert os.path.exists(self._data_path), \
				'Path does not exist: {}'.format(self._data_path)

	def image_path_at(self, i):
		"""
		Return the absolute path to image i in the image sequence.
		"""
		return self.image_path_from_index(self._image_index[i])

	def image_path_from_index(self, index):
		"""
		Construct an image path from the image's "index" identifier.
		"""
		image_path = os.path.join(self._data_path, 'Images',
								  index + self._image_ext)
		assert os.path.exists(image_path), \
				'Path does not exist: {}'.format(image_path)
		return image_path

	def _load_image_set_index(self):
		"""
		Load the indexes listed in this dataset's image set file.
		"""
		# Example path to image set file:
		# self._devkit_path + /DRapc2016/APC2016/ImageSets/Main/val.txt
		image_set_file = os.path.join(self._data_path, 'ImageSets', 'Main',
									  self._image_set + '.txt')
		assert os.path.exists(image_set_file), \
				'Path does not exist: {}'.format(image_set_file)
		with open(image_set_file) as f:
			image_index = [x.strip() for x in f.readlines()]
		return image_index

	def _get_default_path(self):
		"""
		Return the default path where APC is expected to be installed.
		"""
		return os.path.join(cfg.DATA_DIR, 'DRapc2016')

	def gt_roidb(self):
		"""
		Return the database of ground-truth regions of interest.
		This function loads/saves from/to a cache file to speed up future calls.
		"""
		# DISABLE CACHING
		#cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
		#if os.path.exists(cache_file):
		#	with open(cache_file, 'rb') as fid:
		#		roidb = cPickle.load(fid)
		#	print '{} gt roidb loaded from {}'.format(self.name, cache_file)
		#	return roidb

		gt_roidb = [self._load_pascal_annotation(index)
				for index in self.image_index]
		#with open(cache_file, 'wb') as fid:
		#	cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
		#print 'wrote gt roidb to {}'.format(cache_file)

		return gt_roidb

	def selective_search_roidb(self):
		"""
		Return the database of selective search regions of interest.
		Ground-truth ROIs are also included.
		This function loads/saves from/to a cache file to speed up future calls.
		"""
		cache_file = os.path.join(self.cache_path,
								  self.name + '_selective_search_roidb.pkl')

		if os.path.exists(cache_file):
			with open(cache_file, 'rb') as fid:
				roidb = cPickle.load(fid)
			print '{} ss roidb loaded from {}'.format(self.name, cache_file)
			return roidb

		if int(self._year) == 2007 or self._image_set != 'test':
			gt_roidb = self.gt_roidb()
			ss_roidb = self._load_selective_search_roidb(gt_roidb)
			roidb = imdb.merge_roidbs(gt_roidb, ss_roidb)
		else:
			roidb = self._load_selective_search_roidb(None)
		with open(cache_file, 'wb') as fid:
			cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
		print 'wrote ss roidb to {}'.format(cache_file)

		return roidb

	def rpn_roidb(self):
		if int(self._year) == 2007 or self._image_set != 'test':
			gt_roidb = self.gt_roidb()
			rpn_roidb = self._load_rpn_roidb(gt_roidb)
			roidb = imdb.merge_roidbs(gt_roidb, rpn_roidb)
		else:
			roidb = self._load_rpn_roidb(None)

		return roidb

	def _load_rpn_roidb(self, gt_roidb):
		filename = self.config['rpn_file']
		print 'loading {}'.format(filename)
		assert os.path.exists(filename), \
				'rpn data not found at: {}'.format(filename)
		with open(filename, 'rb') as f:
			box_list = cPickle.load(f)
		return self.create_roidb_from_box_list(box_list, gt_roidb)

	def _load_selective_search_roidb(self, gt_roidb):
		filename = os.path.abspath(os.path.join(cfg.DATA_DIR,
			'selective_search_data',
			self.name + '.mat'))
		assert os.path.exists(filename), \
				'Selective search data not found at: {}'.format(filename)
		raw_data = sio.loadmat(filename)['boxes'].ravel()

		box_list = []
		for i in xrange(raw_data.shape[0]):
			boxes = raw_data[i][:, (1, 0, 3, 2)] - 1
			keep = ds_utils.unique_boxes(boxes)
			boxes = boxes[keep, :]
			keep = ds_utils.filter_small_boxes(boxes, self.config['min_size'])
			boxes = boxes[keep, :]
			box_list.append(boxes)

		return self.create_roidb_from_box_list(box_list, gt_roidb)

	def _load_pascal_annotation(self, index):
		"""
		Load image and bounding boxes info from XML file in the PASCAL VOC
		format.
		"""
		filename = os.path.join(self._data_path, 'Annotations', index + '.xml')
		tree = ET.parse(filename)
		objs = tree.findall('object')
		if not self.config['use_diff']:
			# Exclude the samples labeled as difficult
			non_diff_objs = [
				obj for obj in objs if int(obj.find('difficult').text) == 0]
			# if len(non_diff_objs) != len(objs):
			#     print 'Removed {} difficult objects'.format(
			#         len(objs) - len(non_diff_objs))
			objs = non_diff_objs
		num_objs = len(objs)

		boxes = np.zeros((num_objs, 4), dtype=np.uint16)
		gt_classes = np.zeros((num_objs), dtype=np.int32)
		overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
		# "Seg" area for pascal is just the box area
		seg_areas = np.zeros((num_objs), dtype=np.float32)

		# Load object bounding boxes into a data frame.
		for ix, obj in enumerate(objs):
			bbox = obj.find('bndbox')
			# Make pixel indexes 0-based
			x1 = float(bbox.find('xmin').text)
			y1 = float(bbox.find('ymin').text)
			x2 = float(bbox.find('xmax').text)
			y2 = float(bbox.find('ymax').text)
			cls = self._class_to_ind[obj.find('name').text.lower().strip()]
			boxes[ix, :] = [x1, y1, x2, y2]
			gt_classes[ix] = cls
			overlaps[ix, cls] = 1.0
			seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

		overlaps = scipy.sparse.csr_matrix(overlaps)

		return {'boxes' : boxes,
				'gt_classes': gt_classes,
				'gt_overlaps' : overlaps,
				'flipped' : False,
				'seg_areas' : seg_areas}

	def _get_comp_id(self):
		comp_id = (self._comp_id + '_' + self._salt if self.config['use_salt']
				else self._comp_id)
		return comp_id

	def _get_apc_results_file_template(self):
		# DRapc/results/APC2016/Main/<comp_id>_det_test_aeroplane.txt
		filename = self._get_comp_id() + '_det_' + self._image_set + '_{:s}.txt'
		path = os.path.join(
				self._devkit_path,
				'results',
				'APC' + self._year,
				'Main',
				filename)
		return path

	def _write_apc_results_file(self, all_boxes):
		for cls_ind, cls in enumerate(self.classes):
			# Not skip background
			#if cls == '__background__':
			#	continue
			print 'Writing {} APC results file'.format(cls)
			filename = self._get_apc_results_file_template().format(cls)
			with open(filename, 'wt') as f:
				for im_ind, index in enumerate(self.image_index):
					dets = all_boxes[im_ind][cls_ind]
					if dets == []:
						continue
					# the VOCdevkit expects 1-based indices
					for k in xrange(dets.shape[0]):
						f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
								format(index, dets[k, -1],
									dets[k, 0] + 1, dets[k, 1] + 1,
									dets[k, 2] + 1, dets[k, 3] + 1))

	def _do_python_eval(self, output_dir = 'output'):
		annopath = os.path.join(
			self._devkit_path,
			'APC' + self._year,
			'Annotations',
			'{:s}.xml')
		imagesetfile = os.path.join(
			self._devkit_path,
			'APC' + self._year,
			'ImageSets',
			'Main',
			self._image_set + '.txt')
		cachedir = os.path.join(self._devkit_path, 'annotations_cache')
		performance = {}
		performance['aps'] = []
		performance['recs'] = []
		performance['precs'] = []
		performance['prauc'] = []
		performance['tprs'] = []
		performance['fprs'] = []
		performance['rocauc'] = []
		print "\n"
		if not os.path.isdir(output_dir):
			os.mkdir(output_dir)
		for i, cls in enumerate(self._classes):
			filename = self._get_apc_results_file_template().format(cls)
			# skip background class
			if cls == '__background__':
				continue
			# Deprecated AP
			_, _, ap = apc_eval(
					filename, annopath, imagesetfile, \
							cls, cachedir, ovthresh=0.5)

			# TODO: Remove this! This should skip not yet learned objects
			if ap == 0:
				tpr, fpr, rocauc, rec, prec, prauc = [0]*20, [0]*20, 0, [0]*20, [0]*20, 0
			# Preferred PR measure and ROC measure
			else:
				rec, prec, prauc, tpr, fpr, rocauc = apc_evalv2(
						filename, annopath, imagesetfile, \
								cls, cachedir, ovthresh=0.5)

			performance['aps'] += [ap]
			performance['tprs'] += [tpr]
			performance['fprs'] += [fpr]
			performance['rocauc'] += [rocauc]
			performance['recs'] += [rec]
			performance['precs'] += [prec]
			performance['prauc'] += [prauc]

			if ap != 0: #TODO For DEBUG, remove this
				print('AP for {} = {:.4f}'.format(cls, ap))
				print('PR curve AUC for {:s} = {}'.format(cls, prauc))
				print('ROC curve AUC for {:s} = {}'.format(cls, rocauc))
			with open(os.path.join(output_dir, cls + '_pr.pkl'), 'w') as f:
				cPickle.dump({'ap': ap, 'rec': rec, 'prec': prec, 'prauc': prauc, \
					'tpr': tpr, 'fpr': fpr, 'rocauc': rocauc}, f)
		print('\nMean AP = {:.4f}'.format(np.mean(performance['aps'])))
		print('~~~~~~~~')
		print('Results:')
		for ap in performance['aps']:
			print('{:.3f}'.format(ap))
		print('{:.3f}'.format(np.mean(performance['aps'])))
		print('~~~~~~~~')
		return performance

	def evaluate_detections(self, all_boxes, output_dir):
		print "\n"
		self._write_apc_results_file(all_boxes)
		if self.config['python_eval']:
			performance = self._do_python_eval(output_dir)
		if self.config['cleanup']:
			for cls in self._classes:
				if cls == '__background__':
					continue
				filename = self._get_apc_results_file_template().format(cls)
				os.remove(filename)
		return performance

	def competition_mode(self, on):
		if on:
			self.config['use_salt'] = False
			self.config['cleanup'] = False
		else:
			self.config['use_salt'] = True
			self.config['cleanup'] = True

if __name__ == '__main__':
	from datasets.apc import apc
	d = apc('trainval', '2016')
	d = apc('trainval', '2016')
	res = d.roidb
	from IPython import embed; embed()
