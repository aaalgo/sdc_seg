#!/usr/bin/env python3
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
sys.path.append('../aardvark')
import subprocess as sp
import numpy as np
from glob import glob
import cv2
from tqdm import tqdm
import tensorflow as tf
from tensorflow.python.framework import meta_graph
from mold import Scaling as Mold

class Model:
    def __init__ (self, path, name='xxx'):
        mg = meta_graph.read_meta_graph_file(path + '.meta')
        self.images = tf.placeholder(tf.float32, shape=(None, None, None, 3), name="images")
        is_training = tf.constant(False)
        self.probs, = tf.import_graph_def(mg.graph_def, name=name,
                    input_map={'images:0': self.images, 'is_training:0': is_training},
                    return_elements=['probs:0'])
        self.saver = tf.train.Saver(saver_def=mg.saver_def, name=name)
        self.loader = lambda sess: self.saver.restore(sess, path)
        pass
    pass

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('model', None, '')
flags.DEFINE_integer('stride', 1, '')
flags.DEFINE_float('th', 0.5, '')
flags.DEFINE_integer('fps', 30, '')

OUTPUT = 'output'

def overlay (image, prob, color):
    image[:, :, color][prob] += 75
    pass

def main (_):
    model = Model(FLAGS.model)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        model.loader(sess)

        mold = Mold(16, 1.0)

        video_input = cv2.VideoCapture('test.mov')
        video_output = None
        C = 0
        while video_input.grab():
            flag, image = video_input.retrieve()
            if not flag:
                break
            if image.shape[0] > image.shape[1]:
                # patch the BDD .mov files
                image = np.fliplr(image)
                image = np.transpose(image, (1, 0, 2))
                pass

            probs = sess.run(model.probs, feed_dict={model.images: mold.batch_image(image)})

            frame = cv2.resize(image, None, fx=0.5, fy=0.5).astype(np.float32)
            frame *= 0.7
            probs = mold.unbatch_prob(frame, probs)
            overlay(frame, probs[:, :, 1] > 0.5, 0)
            overlay(frame, probs[:, :, 2] > 0.5, 1)
            overlay(frame, probs[:, :, 3] > 0.5, 2)

            frame = np.clip(frame, 0, 255)

            if video_output is None:
                H, W = frame.shape[:2]
                video_output = cv2.VideoWriter('%s.avi' % OUTPUT, cv2.VideoWriter_fourcc(*'MJPG'), FLAGS.fps, (W, H))
            video_output.write(frame.astype(np.uint8))
            print('%d' % C)
            C += 1
            pass
        video_output.release()
        # convert to MP4 so it can be served on web
        sp.check_call('ffmpeg -i %s.avi -y -c:v libx264 -preset slow -crf 22 -pix_fmt yuv420p %s.mp4' % (OUTPUT, OUTPUT), shell=True)
    pass

if __name__ == '__main__':
    tf.app.run()

