#!/usr/bin/env python3
import sys
import os
sys.path.append('../picpac/build/lib.linux-x86_64-3.5')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.python.framework import meta_graph
import picpac
from gallery import Gallery

class Model:
    def __init__ (self, X, path, name):
        mg = meta_graph.read_meta_graph_file(path + '.meta')
        is_training = tf.constant(False)
        self.logits, = \
                tf.import_graph_def(mg.graph_def, name=name,
                    input_map={'images:0': X, 'is_training:0': is_training},
                    return_elements=['logits:0'])
        self.saver = tf.train.Saver(saver_def=mg.saver_def, name=name)
        self.loader = lambda sess: self.saver.restore(sess, path)
        pass
    pass

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('model', None, '')
flags.DEFINE_integer('width', 144, '')
flags.DEFINE_integer('height', 48, '')

flags.DEFINE_string('db', 'scratch/val.db', '')
flags.DEFINE_integer('max', 50, '')


def logits2text (logits):
    logits = np.reshape(logits, [6, 36]) # 6个字，每个字10+26种可能
    label = []
    for i in range(logits.shape[0]):
        v = np.argmax(logits[i])
        if v < 10:
            label.append(chr(ord('0') + v))
        else:
            label.append(chr(ord('A') + v - 10))
    return ''.join(label)

def main (_):
    X = tf.placeholder(tf.float32, shape=(None, None, None, 1), name="images")
    model = Model(X, FLAGS.model, 'xxx')
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        model.loader(sess)
        # 模型至此导入完毕
        # 注意如果要读入图片预测，需要进行下述预处理
        # 1. 转成灰度图, 比如用cv2.imread(..., cv2.IMREAD_GRAYSCALE)
        # 2. resize成固定大小，和train.sh中的参数匹配
        # 3. 比如灰度图是image, 那么对应的batch = image[np.newaxis, :, :, np.newaxis], 就可以送入tf.session预测了

        gal = Gallery('output', cols=2, ext='.jpg')
        CC = 0
        stream = picpac.ImageStream({'db': FLAGS.db, 'loop': False, 'channels': 1, 'threads': 1, 'shuffle': False,
            'transforms': [{"type": "resize", "width": FLAGS.width, "height": FLAGS.height}]})
        for meta, batch in stream:
                if CC > FLAGS.max:
                    break
                print(meta.ids)
                image = batch[0]
                logits = sess.run(model.logits, feed_dict={X: batch})
                # 把logits转换成字符串
                label = logits2text(logits)
                '''END INFERENCE'''
                save_prediction_image(gal, image, label)
                CC += 1
        gal.flush()
    pass

# 保存预测结果图片
def save_prediction_image (gal, image, label):
    image2 = np.zeros((image.shape[0]+40, image.shape[1]), dtype=np.float32)
    image2[40:, :] = image[:, :, 0]
    image = image2
    cv2.putText(image, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255), 2)
    cv2.imwrite(gal.next(), image)
    pass

if __name__ == '__main__':
    tf.app.run()

