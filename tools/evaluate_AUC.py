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
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list
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

def AUC_Calculator(classes, tprss, fprss):
	pass

def ROC_Vis(classes, tprss, fprss):
	tprss = np.array(tprss)
	fprss = np.array(fprss)
	if len(classes)-1 == tprss.shape[0]:
		num_figs = len(classes) - 1
		num_horizontal = 5
		num_vertical = int(math.ceil(num_figs/float(num_horizontal)))
		fig, axs = plt.subplots(num_vertical, num_horizontal, figsize=(15, 6), facecolor='w', edgecolor='k')
		fig.subplots_adjust(hspace = .5, wspace=.5)

		axs = axs.ravel()	
		for i in xrange(0,num_figs):
			fpr = np.concatenate(([1.], fprss[i,:], [0.]))
			tpr = np.concatenate(([1.], tprss[i,:], [0.]))
			axs[i].plot(fpr, tpr, 'bo-')
			axs[i].set_title(classes[i+1])
			axs[i].set_xlabel('FP rate')
			axs[i].set_xlim(0.,1.)
			axs[i].set_ylabel('TP rate')
			axs[i].set_ylim(0.,1.)
		plt.show()
			
	else:
		print 'Error: Number of classses {} are not consistent with number of classes in tp array and fp arrays {}'.format(len(classes)-1,tprss.shape[1])

def PR_Vis(classes, recss, precss):
	recss = np.array(recss)
	precss = np.array(precss)
	if len(classes)-1 == recss.shape[0]:
		num_figs = len(classes) - 1
		num_horizontal = 5
		num_vertical = int(math.ceil(num_figs/float(num_horizontal)))
		fig, axs = plt.subplots(num_vertical, num_horizontal, figsize=(15, 6), facecolor='w', edgecolor='k')
		fig.subplots_adjust(hspace = .5, wspace=.5)

		axs = axs.ravel()	
		for i in xrange(0,num_figs):
			rec = np.concatenate(([1.], recss[i,:], [0.]))
			prec = np.concatenate(([0.], precss[i,:], [1.]))
			axs[i].plot(rec, prec, 'b-')
			axs[i].set_title(classes[i+1])
			axs[i].set_xlabel('Recall')
			axs[i].set_xlim(0.,1.)
			axs[i].set_ylabel('Precision')
			axs[i].set_ylim(0.,1.)
		plt.show()
			
	else:
		print 'Error: Number of classses {} are not consistent with number of classes in recall array and precision arrays {}'.format(len(classes)-1,recss.shape[1])


if __name__ == '__main__':
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
	print args.prototxt

	imdb = get_imdb(args.imdb_name)
	print "IMDB: "
	print imdb
	imdb.competition_mode(args.comp_mode)
	if not cfg.TEST.HAS_RPN:
		imdb.set_proposal_method(cfg.TEST.PROPOSAL_METHOD)

	print "Model path:"
	print args.caffemodel

	# do one detection and save the detections.pkl

	net = caffe.Net(args.prototxt, args.caffemodel, caffe.TEST)
	net.name = os.path.splitext(os.path.basename(args.caffemodel))[0]
	(classes, performance) = test_net(net, imdb, max_per_image=args.max_per_image, thresh=0.)

	recss = performance['recs']
	precss = performance['precs']
	tprss = performance['tprs']
	fprss = performance['fprs']

	ROC_Vis(classes, tprss, fprss)
	PR_Vis(classes, recss, precss)	


