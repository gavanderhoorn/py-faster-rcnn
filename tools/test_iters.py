# This script evaluates the performance of the same model trained with different iterations (snapshots during training)
# The result is a plot of the precision for each class for different numbers of iterations.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib
from fast_rcnn.config import cfg
import os 

def make_figure(classes, iterations, performance_list):
	print classes
	print performance_list
	print iterations

	n_classes = len(classes)
	
	font = {'family' : 'sans-serif',
		'weight' : 'normal',
		'size'   : 22}

	print cfg.DATA_DIR

	matplotlib.rc('font', **font)
	performance_array = np.transpose(np.array(performance_list))
	print performance_array
	fig = plt.figure(figsize=(20,10),facecolor='white')
	ax = plt.subplot(111)
	n_precisions = performance_array.shape[0]

	colors = plt.cm.cool(np.linspace(0,1,n_classes))
	for c in range(n_precisions):
		if c < n_precisions - 1:
			class_name = classes[c+1]
		else:
			# last precision is the average precision (AP)
			class_name = 'AP'
		
		print class_name
		print type(performance_array)
		print performance_array.shape
		print c
		performance_class = performance_array[c,:]
		print performance_class
		if c < n_precisions - 1:
			ax.plot(performance_class,label=class_name,color=colors[c])
		else:
			ax.plot(performance_class,label=class_name,lw=3,color=colors[c])
		plt.hold(True)
	
	# Shrink current axis by 70%
	box = ax.get_position()
	ax.set_position([box.x0, box.y0+box.height*0.1, box.width * 0.8, box.height*0.9])
	ax.legend(loc='upper center', bbox_to_anchor=(1.2,1),borderaxespad=0.)
	ax.set_title('Number of iterations vs model precision')
	labels = iterations
	ax.set_xticks(range(len(iterations)))
	ax.set_xticklabels(labels,rotation=45)
	ax.set_xlabel('Iterations')
	ax.set_ylabel('Precision (AP)')
	ax.set_ylim([0, 1])
	image_output_path = os.path.join(cfg.DATA_DIR,'DRapc2016','testOutput','iterations_precision.eps')
	plt.savefig(image_output_path, format='eps', dpi=1000, facecolor=fig.get_facecolor())
	plt.show()
