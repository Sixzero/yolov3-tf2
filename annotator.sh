#!/usr/bin/env bash

FLAG=(--tiny)
python3 annotator.py \
 --image ../YOLOv3_TensorFlow/annotations/vott-json-export/0cbb25d2f16debf4c2e7c2862b6e777f.jpg \
 --weights ./checkpoints/yolov3_best.tf \
 --classes ../YOLOv3_TensorFlow/data/my_data/data.names "${FLAG}" $*
