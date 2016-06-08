#!/bin/bash
# Usage:
# ./experiments/scripts/faster_rcnn_end2end.sh GPU NET DATASET [options args to {train,test}_net.py]
# DATASET is either pascal_voc or coco.
#
# Example:
# ./experiments/scripts/faster_rcnn_end2end_test.sh 0 VGG16 apc vgg16_faster_rcnn_iter_1.caffemodel \
#   --set EXP_DIR foobar RNG_SEED 42 TRAIN.SCALES "[400, 500, 600, 700]"

set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=$1
NET=$2
NET_lc=${NET,,}
DATASET=$3
NET_FINAL=$4

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:4:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

case $DATASET in
  apc)
	TEST_IMDB="apc_2016_test"
	PT_DIR="apc"
	;;
  apcbin)
	TEST_IMDB="apc_bin_test"
	PT_DIR="apc"
	;;
  apctote)
	TEST_IMDB="apc_tote_test"
	PT_DIR="apc"
	;;
  *)
    echo "No dataset given"
    exit
    ;;
esac


time ./tools/test_net.py --gpu ${GPU_ID} \
  --def models/${PT_DIR}/${NET}/faster_rcnn_end2end/test.prototxt \
  --net ${NET_FINAL} \
  --imdb ${TEST_IMDB} \
  --cfg experiments/cfgs/faster_rcnn_end2end.yml \
  --vis \
  ${EXTRA_ARGS}
