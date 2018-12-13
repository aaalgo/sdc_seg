#!/usr/bin/env python3
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
sys.path.append('../aardvark')
import numpy as np
from glob import glob
import cv2
from tqdm import tqdm
import tensorflow as tf
from tensorflow.python.framework import meta_graph
from mold import Scaling as Mold
from gallery import Gallery
#from skimage.morphology import remove_small_objects

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

tasks = glob('data/cityscape/leftImg8bit/test/*/*.png')[:20]

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

        mold = Mold(16, 0.5)

        gal = Gallery('output')
        for path in tasks:
            image = cv2.imread(path, cv2.IMREAD_COLOR)
            probs = sess.run(model.probs, feed_dict={model.images: mold.batch_image(image)})

            output = cv2.resize(image, None, fx=0.25, fy=0.25).astype(np.float32)
            probs = mold.unbatch_prob(output, probs)

            output *= 0.7
            overlay(output, probs[:, :, 1] > 0.5, 0)
            overlay(output, probs[:, :, 2] > 0.5, 1)
            overlay(output, probs[:, :, 3] > 0.5, 2)
            output = np.clip(output, 0, 255)
            print(output.shape, probs.shape)
            cv2.imwrite(gal.next(), output)
            pass
        gal.flush()
    pass

if __name__ == '__main__':
    tf.app.run()

