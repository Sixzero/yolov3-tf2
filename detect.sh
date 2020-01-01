#!/usr/bin/env bash

FLAG=(--tiny)
python detect.py \
 --image /Users/tamashavlik/data/tw_plays/screen/7bce0d2b5af81ea2714e1bd6e0908993.jpg \
 --weights ./checkpoints/yolov3-tiny_best.tf \
 --classes ../yolov3-tf2/train/voc2012/classes.txt "${FLAG}" $*
