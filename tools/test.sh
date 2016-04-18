#!/bin/sh
# Must be run from dir 'py-faster-rcnn', via: './tools/test.sh'
#./tools/test_net.py --gpu 0 --def models/apc/VGG16/faster_rcnn_end2end/test.prototxt --model_path /home/$USER/py-faster-rcnn/output/faster_rcnn_end2end/apc_2016_trainval/ --imdb apc_test_validated --cfg experiments/cfgs/faster_rcnn_end2end.yml --vis --eval_iters True
./tools/test_net.py --gpu 0 --def models/apc/VGG16/faster_rcnn_end2end/test.prototxt --net /home/$USER/py-faster-rcnn/output/faster_rcnn_end2end/apc_old_models/vgg16_faster_rcnn_iter_10000.caffemodel --imdb apc_test_validated --cfg experiments/cfgs/faster_rcnn_end2end.yml --thresh 0.05
#./tools/test_net.py --gpu 0 --def models/apc/VGG16/faster_rcnn_end2end/test.prototxt --model_path /home/$USER/py-faster-rcnn/output/faster_rcnn_end2end/vgg16_faster_rcnn_iter_10000.caffemodel --imdb apc_test_validated --cfg experiments/cfgs/faster_rcnn_end2end.yml --eval_iters True --thresh 0.05
