import time
import cv2

class Timer(object):
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff


from keras.layers import Input, Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Flatten, BatchNormalization, Permute, TimeDistributed, Dense, Bidirectional, GRU
from keras.models import Model

import numpy as np
from PIL import Image
import keras.backend  as K

from imp import reload
from keras_ocr import densenet

import os
from keras.layers import Lambda
from keras.optimizers import SGD
import imutils
import tensorflow as tf
import keras.backend.tensorflow_backend as K
from matplotlib import pyplot as plt


def get_session(gpu_fraction=0.8):
    '''''Assume that you have 6GB of GPU memory and want to allocate ~2GB'''

    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


K.set_session(get_session())

with open('char_std_5990.txt', encoding='utf-8') as f:
    chars = f.read().split('\n')

# caffe_ocr中把0作为blank，但是tf 的CTC  the last class is reserved to the blank label.
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/util/ctc/ctc_loss_calculator.h
nclass = len(chars)
print('nclass:', len(chars))
id_to_char = {i: j for i, j in enumerate(chars)}

model_path = '/home/adam/workspace/github/okra/keras_ocr/weights-densent-32-0.9846.hdf5'
input = Input(shape=(32, None, 1), name='the_input')
y_pred = densenet.dense_cnn(input, nclass)
basemodel = Model(inputs=input, outputs=y_pred)
basemodel.load_weights(model_path)
t = Timer()


def predict(image_path):
    image = cv2.imread(image_path, 0)
    image = imutils.resize(image, height=32)
    image_height, image_width = image.shape[:2]
    image = image.astype(np.float32) / 255.0 - 0.5
    image = image.reshape(32, image_width, 1)
    # X = np.array([X])
    input = np.expand_dims(image, axis=0)

    t.tic()
    y_pred = basemodel.predict(input)
    t.toc()
    print("times,", t.diff)
    # argmax = np.argmax(y_pred, axis=2)[0]

    y_pred = y_pred[:, :, :]
    out = K.get_value(K.ctc_decode(y_pred, input_length=np.ones(y_pred.shape[0]) * y_pred.shape[1], )[0][0])[:, :]
    out = ''.join([id_to_char[x] for x in out[0]])

    return out, image


test_image = '/home/adam/Pictures/vat_dates/1100162350_12093275_20180427_299606_2.jpg'
result, image = predict(test_image)
print(result)
cv2.imshow('test_image', image)
cv2.waitKey(0)
# plt.imshow(image, cmap='gray')
