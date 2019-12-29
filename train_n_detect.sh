#!/usr/bin/env bash
FLAG=(--tiny)
python train.py --batch_size 8 \
--dataset '../YOLOv3_TensorFlow/annotations/tw_plays_tft2-TFRecords-export/*.tfrecord' \
--val_dataset '../YOLOv3_TensorFlow/annotations/tw_plays_tft2-TFRecords-export/*.tfrecord' \
 --classes ../YOLOv3_TensorFlow/data/my_data/data.names --epochs 1000 \
 --mode fit --batch_size 16 \
 --transfer darknet \
 --learning_rate 0.001 "${FLAG}" $*


#--dataset '../YOLOv3_TensorFlow/annotations/tw_plays_tft2-TFRecords-export/*.tfrecord' \
#--val_dataset '../YOLOv3_TensorFlow/annotations/tw_plays_tft2-TFRecords-export/*.tfrecord' \

python detect.py \
 --image ../YOLOv3_TensorFlow/annotations/vott-json-export/0cbb25d2f16debf4c2e7c2862b6e777f.jpg \
 --weights ./checkpoints/yolov3_best.tf \
 --classes ../YOLOv3_TensorFlow/data/my_data/data.names "${FLAG}" $*
