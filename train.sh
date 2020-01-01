
python train.py --batch_size 1  \
    --val_dataset ./train/testing/*.tfrecord \
    --dataset ./train/testing/*.tfrecord \
    --classes ./train/voc2012/classes.txt  \
    --epochs 100 --mode fit --transfer darknet --tiny
# python3 train.py --batch_size 8 --dataset /Users/tamashavlik/repo/test/records/Screens-TFRecords-export/1ff88ed89985e015974f519f5e9e6a2e.tfrecord --epochs 100 --mode eager_tf --transfer fine_tune


#    --dataset /Users/tamashavlik/repo/test/records/Screens-TFRecords-export/*.tfrecord \
#    --val_dataset /Users/tamashavlik/repo/test/records/Screens-TFRecords-export/*.tfrecord \

