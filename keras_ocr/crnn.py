from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Reshape, Masking, Lambda, Permute
from keras.layers import Input, Dense, Flatten
from keras.preprocessing.sequence import pad_sequences
from keras.layers.recurrent import GRU, LSTM
from keras.layers.wrappers import Bidirectional
from keras.models import Model
from keras import backend as K
from keras.preprocessing import image
from keras.optimizers import Adam, SGD, Adadelta
from keras import losses
from keras.layers.wrappers import TimeDistributed
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.utils import plot_model
from matplotlib import pyplot as plt
import numpy as np
import os
import json
import tensorflow as tf
import keras.backend.tensorflow_backend as K


class CRNN():
    @staticmethod
    def build(input_shape=(32, None, 1), rnn_unit=256, num_classes=5990, max_string_len=30):
        input = Input(shape=input_shape, name='the_input')
        m = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', name='conv1')(input)
        m = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1')(m)
        m = Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same', name='conv2')(m)
        m = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool2')(m)
        m = Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same', name='conv3')(m)
        m = Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same', name='conv4')(m)
        m = MaxPooling2D(pool_size=(2, 1), strides=(2, 2), padding='valid', name='pool3')(m)
        m = Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same', name='conv5')(m)
        m = BatchNormalization(axis=3)(m)
        m = Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same', name='conv6')(m)
        m = BatchNormalization(axis=3)(m)
        m = MaxPooling2D(pool_size=(2, 1), strides=(2, 2), padding='valid', name='pool4')(m)
        m = Conv2D(512, kernel_size=(2, 2), activation='relu', padding='valid', name='conv7')(m)
        m = Permute((2, 1, 3), name='permute')(m)
        m = TimeDistributed(Flatten(), name='timedistrib')(m)
        m = Bidirectional(GRU(rnn_unit, return_sequences=True, implementation=2), name='blstm1')(m)
        m = Bidirectional(GRU(rnn_unit, return_sequences=True, implementation=2), name='blstm2')(m)
        y_pred = Dense(num_classes, name='blstm2_out', activation='softmax')(m)
        basemodel = Model(inputs=input, outputs=y_pred)
        basemodel.summary()
        label = Input(name='label', shape=[max_string_len], dtype='int64')
        input_length = Input(name='input_length', shape=[1], dtype='int64')
        label_length = Input(name='label_length', shape=[1], dtype='int64')
        loss_out = Lambda(ctc_lambda_func, output_shape=(1,),
                          name='ctc')([label, y_pred, input_length, label_length])
        model = Model(input=[input, label, input_length, label_length], output=[loss_out])
        return model


def ctc_lambda_func(args):
    # https://www.tensorflow.org/api_docs/python/tf/keras/backend/ctc_batch_cost
    y_true, y_pred, input_length, label_length = args
    return K.ctc_batch_cost(y_true, y_pred, input_length, label_length)

def generator():


crnn = CRNN.build()
crnn.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adadelta')
