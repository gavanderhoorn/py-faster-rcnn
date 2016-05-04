#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Test a Fast R-CNN network on an image database."""

import _init_paths
from fast_rcnn.test import test_net
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from datasets.factory import get_imdb
from test_iters import make_figure
import caffe
import argparse
import pprint
import time, os, sys
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.metrics import auc
from classes import inverse_classes, inverse_classes_short
import cPickle
import getpass
USER = getpass.getuser()

def parse_args():
	"""
	Parse input arguments
	"""
	parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
	parser.add_argument('--gpu', dest='gpu_id', help='GPU id to use',
						default=0, type=int)
	parser.add_argument('--def', dest='prototxt',
						help='prototxt file defining the network',
						default=None, type=str)
	parser.add_argument('--net', dest='caffemodel',
						help='model to test',
						default=None, type=str)
	parser.add_argument('--cfg', dest='cfg_file',
						help='optional config file', default=None, type=str)
	parser.add_argument('--wait', dest='wait',
						help='wait until net file exists',
						default=True, type=bool)
	parser.add_argument('--imdb', dest='imdb_name',
						help='dataset to test',
						default='apc_test_validated', type=str)
	parser.add_argument('--comp', dest='comp_mode', help='competition mode',
						action='store_true')
	parser.add_argument('--set', dest='set_cfgs',
						help='set config keys', default=None,
						nargs=argparse.REMAINDER)
	parser.add_argument('--vis', dest='vis', help='visualize detections',
						action='store_true')
	parser.add_argument('--num_dets', dest='max_per_image',
						help='max number of detections per image',
						default=100, type=int)
	parser.add_argument('--eval_iters', dest='eval_iters',
						help='test multiple models to evaluate the effect of iterations',
						default=False, type=bool)

	if len(sys.argv) == 1:
		parser.print_help()
		print "No arguments given, exit"
		sys.exit(1)

	args = parser.parse_args()
	return args

def ROC_Vis(classes, tprss, fprss):
	tprss = np.array(tprss)
	fprss = np.array(fprss)
	if len(classes)-1 == tprss.shape[0]:
		num_figs = len(classes) - 1
		num_horizontal = 5
		num_vertical = int(math.ceil(num_figs/float(num_horizontal)))
		fig, axs = plt.subplots(num_vertical, num_horizontal, figsize=(15, 24), facecolor='w', edgecolor='k')
		fig.subplots_adjust(hspace = .5, wspace=.5)
		optimal_idc, optimal_threshs = determine_threshold(classes, fprss, tprss, mode='roc')

		axs = axs.ravel()
		for i in xrange(0,num_figs):
			fpr = np.concatenate(([1.], fprss[i,:], [0.]))
			tpr = np.concatenate(([1.], tprss[i,:], [0.]))
			axs[i].plot(fpr, tpr, 'b-', linewidth=3)
			axs[i].set_title(inverse_classes_short[i+1])
			axs[i].set_xlabel('FP rate')
			axs[i].set_xlim(0.,1.)
			axs[i].set_ylabel('TP rate')
			axs[i].set_ylim(0.,1.)
			axs[i].spines['top'].set_color('none')
			axs[i].spines['right'].set_color('none')
			axs[i].xaxis.set_ticks_position('bottom')
			axs[i].yaxis.set_ticks_position('left')
			# highlight best rec-prec pair
			cls = classes[i+1]
			axs[i].scatter(fprss[i, optimal_idc[cls]],tprss[i, optimal_idc[cls]], s=100, c='r', label='optimal thresh: {}'.format(optimal_threshs[cls]))
			axs[i].legend(loc='lower right', numpoints=1, fontsize=8)

		plt.savefig('/home/{}/workspace/ROC.png'.format(USER))
	else:
		print 'Error: Number of classses {} are not consistent with number of classes in tp array and fp arrays {}'.format(len(classes)-1,tprss.shape[0])

def PR_Vis(classes, recss, precss):
	recss = np.array(recss)
	precss = np.array(precss)
	if len(classes)-1 == recss.shape[0]:
		num_figs = len(classes) - 1
		num_horizontal = 5
		num_vertical = int(math.ceil(num_figs/float(num_horizontal)))
		fig, axs = plt.subplots(num_vertical, num_horizontal, figsize=(15, 24), facecolor='w', edgecolor='k')
		fig.subplots_adjust(hspace = .5, wspace = .5)
		optimal_idc, optimal_threshs = determine_threshold(classes, recss, precss, mode='pr')

		axs = axs.ravel()
		for i in xrange(0,num_figs):
			rec = np.concatenate(([1.], recss[i,:], [0.]))
			prec = np.concatenate(([0.], precss[i,:], [1.]))
			axs[i].plot(rec, prec, 'b-', linewidth=3)
			axs[i].set_title(inverse_classes_short[i+1])
			axs[i].set_xlabel('Recall')
			axs[i].set_xlim(0.,1.)
			axs[i].set_ylabel('Precision')
			axs[i].set_ylim(0.,1.)
			axs[i].spines['top'].set_color('none')
			axs[i].spines['right'].set_color('none')
			axs[i].xaxis.set_ticks_position('bottom')
			axs[i].yaxis.set_ticks_position('left')
			# highlight best rec-prec pair
			cls = classes[i+1]
			ot = axs[i].scatter(recss[i, optimal_idc[cls]],precss[i, optimal_idc[cls]], s=100, c='r', label='optimal thresh: {}'.format(optimal_threshs[cls]))
			axs[i].legend(loc='lower right', numpoints=1, fontsize=8)
		plt.savefig('/home/{}/workspace/PR.png'.format(USER))
	else:
		print 'Error: Number of classses {} are not consistent with number of classes in recall array and precision arrays {}'.format(len(classes)-1,recss.shape[0])

def determine_threshold(classes, xss, yss, mode='pr'):
	xss = np.array(xss)
	yss = np.array(yss)
	thresholds = np.arange(0.,1., 0.05)
	idc = {}
	optimal_threshs = {}
	if mode == 'pr':
		"criterion: maximum f1_score"
		scores = 2 * (yss * xss) / \
				np.maximum((yss + xss), np.finfo(np.float64).eps)

	else:
		"criterion: maximum Youden index"
		scores = yss - xss
	# pick median if there are multiple thresholds
	for i in xrange(0, len(classes)-1):
		cls = classes[i+1]
		if np.sum(scores[i,:]) == 0:
			idx = 0.
			idc[cls] = idx
			optimal_threshs[cls] = thresholds[idx]
		else:
			#print (mode, cls, scores[i,:])
			thresh_ids = np.where(scores[i,:] == np.max(scores[i,:]))
			idx = thresh_ids[0][len(thresh_ids[0]) / 2]
			idc[cls] = idx
			optimal_threshs[cls] = thresholds[idx]
			print '{} Threshold for class {} is {}'.format(mode, cls, thresholds[idx])
	return idc, optimal_threshs

def readCache(pkl_dir = 'output'):
	performance = {}
	performance['aps'] = []
	performance['recs'] = []
	performance['precs'] = []
	performance['prauc'] = []
	performance['tprs'] = []
	performance['fprs'] = []
	performance['rocauc'] = []
	classes = ['__background__']
	for i,cls in inverse_classes.items():
		classes.append(cls)
		with open (os.path.join(pkl_dir, cls + '_pr.pkl'), 'rb') as f:
			obj = cPickle.load(f)
			performance['aps']= [obj['ap']]
			performance['tprs'] += [obj['tpr']]
			performance['fprs'] += [obj['fpr']]
			performance['rocauc'] += [obj['rocauc']]
			performance['recs'] += [obj['rec']]
			performance['precs'] += [obj['prec']]
			performance['prauc'] += [obj['prauc']]
	return (classes, performance)



if __name__ == '__main__':
	DEBUG = 0
	args = parse_args()

	print('Called with args:')
	print(args)

	if args.cfg_file is not None:
		cfg_from_file(args.cfg_file)
	if args.set_cfgs is not None:
		cfg_from_list(args.set_cfgs)

	cfg.GPU_ID = args.gpu_id

	print('Using config:')
	pprint.pprint(cfg)

	#while not os.path.exists(args.caffemodel) and args.wait:
	#    print('Waiting for {} to exist...'.format(args.caffemodel))
	#    time.sleep(10)

	caffe.set_mode_gpu()
	caffe.set_device(args.gpu_id)
	print "TEST NET creation parameters:"
	print "prototxt: "
#	print args.prototxt

	imdb = get_imdb(args.imdb_name)
	print "IMDB: "
	print imdb
	imdb.competition_mode(args.comp_mode)
	if not cfg.TEST.HAS_RPN:
		imdb.set_proposal_method(cfg.TEST.PROPOSAL_METHOD)

	print "Model path:"
#	print args.caffemodel

	# do one detection and save the detections.pkl

	net = caffe.Net(args.prototxt, args.caffemodel, caffe.TEST)
	net.name = os.path.splitext(os.path.basename(args.caffemodel))[0]

	output_dir = get_output_dir(imdb, net)
	if DEBUG:
		(classes, performance) = readCache(output_dir)
	else:
		(classes, performance) = test_net(net, imdb, max_per_image=args.max_per_image, thresh=0.)

	recss = performance['recs']
	precss = performance['precs']
	tprss = performance['tprs']
	fprss = performance['fprs']


	ROC_Vis(classes, tprss, fprss)
	PR_Vis(classes, recss, precss)	


