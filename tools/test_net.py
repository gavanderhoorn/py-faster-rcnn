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
	parser.add_argument('--net', dest='caffe_model_path',
						help='path to caffe model or path to directory if --eval_iters = True',
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
	parser.add_argument('--thresh',dest='thresh_detect',
						help='Threshold for detection',
						type=float,
						default=0.3)

	if len(sys.argv) == 1:
		parser.print_help()
		print "No arguments given, exit"
		sys.exit(1)

	args = parser.parse_args()
	return args

def get_iterations_from_filename(filename):
	iterations = int(os.path.splitext(os.path.basename(filename))[0].split("_")[-1])
	return iterations

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

	# TODO: does not work for when a single model is given
	print "Model path:"
	print args.caffe_model_path
	iterations = []
	if(args.eval_iters == True):
		# Get list of models in directory and SORT on the number of iterations (last part of filenames, preceded by '_') 
		model_paths = [os.path.join(args.caffe_model_path, f) for f in os.listdir(args.caffe_model_path) if f.endswith("caffemodel")]
		model_paths = sorted(model_paths, key=get_iterations_from_filename)
	else:
		# Path of model is given by user, use this
		model_paths = [args.caffe_model_path]

	print "Model path(s)"
	print model_paths

	performance_list	= []
	iterations_list		= []
	
	#print sorted(os.listdir(args.caffe_model_path))
	
	for model_file in model_paths:
		print "caffemodel:"
		print model_file
		print "modelpath:"
		print os.path.join(args.caffe_model_path,model_file)
		print "Iterations"
		iterations = os.path.splitext(os.path.basename(model_file))[0].split("_")
		print iterations[-1]
		iterations_list += [iterations[-1]]
		print iterations_list
		net = caffe.Net(args.prototxt, model_file, caffe.TEST)
		net.name = os.path.splitext(os.path.basename(model_file))[0]
		(classes, performance) = test_net(net, imdb, max_per_image=args.max_per_image, thresh=args.thresh_detect, vis=args.vis)
		performance_list += [performance['aps']]
	
	if(args.eval_iters == True):
		pass
		# make plot
		make_figure(classes, iterations_list, performance_list)
