import cv2
import numpy as np
import tensorflow as tf
import time
from absl import app, flags, logging
from absl.flags import FLAGS

from yolov3_tf2.dataset import transform_images, load_tfrecord_dataset
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.utils import draw_outputs

flags.DEFINE_string('classes', './data/coco.names', 'path to classes file')
flags.DEFINE_string('weights', './checkpoints/yolov3.tf',
                    'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('image', './data/girl.png', 'path to input image')
flags.DEFINE_string('tfrecord', None, 'tfrecord instead of image')
flags.DEFINE_string('output_dir', './results/', 'path to output image')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

yolo = None


def get_yolo():
    global yolo
    if yolo is None:
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        if len(physical_devices) > 0:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print('FLAGS.tiny', FLAGS.tiny)
        if FLAGS.tiny:
            yolo = YoloV3Tiny(classes=FLAGS.num_classes)
        else:
            yolo = YoloV3(classes=FLAGS.num_classes)
        yolo.load_weights(FLAGS.weights).expect_partial()
        logging.info('weights loaded')
    return yolo


def run_on_img(img):
    yolo = get_yolo()

    img = tf.expand_dims(img, 0)
    img = transform_images(img, FLAGS.size)

    t1 = time.time()
    boxes, scores, classes, nums = yolo(img)
    t2 = time.time()
    logging.info('time: {}'.format(t2 - t1))
    return boxes, scores, classes, nums


def interpret(res, class_names):
    boxes, scores, classes, nums = res
    logging.info('detections:')
    for i in range(nums[0]):
        logging.info('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
                                           np.array(scores[0][i]),
                                           np.array(boxes[0][i])))


def save_img(img_raw, res, class_names, fname):
    boxes, scores, classes, nums = res
    img = cv2.cvtColor(img_raw.numpy(), cv2.COLOR_RGB2BGR)
    img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
    file_out = os.path.join(FLAGS.output_dir, fname.split("/")[-1] if fname else 'output.jpg')
    cv2.imwrite(file_out, img)
    logging.info('output saved to: {}'.format(file_out))


def main(_argv):
    if FLAGS.tfrecord:
        dataset = load_tfrecord_dataset(
            FLAGS.tfrecord, FLAGS.classes, FLAGS.size)
        dataset = dataset.shuffle(512)
        img_raw, _label = next(iter(dataset.take(1)))
        images = [img_raw]
    else:
        import glob
        files = glob.glob('/'.join(FLAGS.image.split('/')[:-1]) + '/*.jpg')
        print('FLAGS.image.split()[:-1]', FLAGS.image.split('/')[:-1])
        print('files', files)
        images = [tf.image.decode_image(open(f, 'rb').read(), channels=3) for f in files]

    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    logging.info('classes loaded')
    for i, img_raw in enumerate(images[:20]):
        res = run_on_img(img_raw)
        interpret(res, class_names)
        save_img(img_raw, res, class_names, files[i])


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
