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
from keras_ocr.hdf5 import HDF5DatasetGenerator
import keras


class CRNN():
    @staticmethod
    def build(input_shape=(32, 248, 1), rnn_unit=256, num_classes=14, max_string_len=11):
        input = Input(shape=input_shape, name='the_input')
        m = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', name='conv1')(input)
        m = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1')(m)
        m = Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same', name='conv2')(m)
        m = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool2')(m)
        m = Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same', name='conv3')(m)
        m = Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same', name='conv4')(m)
        m = MaxPooling2D(pool_size=(2, 1), strides=(2, 1), padding='valid', name='pool3')(m)
        m = Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same', name='conv5')(m)
        m = BatchNormalization(axis=3)(m)
        m = Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same', name='conv6')(m)
        m = BatchNormalization(axis=3)(m)
        m = MaxPooling2D(pool_size=(2, 1), strides=(2, 1), padding='valid', name='pool4')(m)
        m = Conv2D(512, kernel_size=(2, 2), activation='relu', padding='valid', name='conv7')(m)
        m = Permute((2, 1, 3), name='permute')(m)
        m = TimeDistributed(Flatten(), name='timedistrib')(m)
        m = Bidirectional(GRU(rnn_unit, return_sequences=True, implementation=2), name='blstm1')(m)
        m = Bidirectional(GRU(rnn_unit, return_sequences=True, implementation=2), name='blstm2')(m)
        y_pred = Dense(num_classes, name='blstm2_out', activation='softmax')(m)
        basemodel = Model(inputs=input, outputs=y_pred)
        label = Input(name='label', shape=[max_string_len], dtype='int64')
        seq_length = Input(name='seq_length', shape=[1], dtype='int64')
        label_length = Input(name='label_length', shape=[1], dtype='int64')
        loss_out = Lambda(ctc_lambda_func, output_shape=(1,),
                          name='ctc')([label, y_pred, seq_length, label_length])
        model = Model(input=[input, label, seq_length, label_length], output=[loss_out])
        model.summary()
        return model


def ctc_lambda_func(args):
    # https://www.tensorflow.org/api_docs/python/tf/keras/backend/ctc_batch_cost
    y_true, y_pred, input_length, label_length = args
    return K.ctc_batch_cost(y_true, y_pred, input_length, label_length)


# def gen(batch_size=128):
#     X = np.zeros((batch_size, width, height, 3), dtype=np.uint8)
#     y = np.zeros((batch_size, n_len), dtype=np.uint8)
#     while True:
#         generator = ImageCaptcha(width=width, height=height)
#         for i in range(batch_size):
#             random_str = ''.join([random.choice(characters) for j in range(4)])
#             X[i] = np.array(generator.generate_image(random_str)).transpose(1, 0, 2)
#             y[i] = [characters.find(x) for x in random_str]
#         yield [X, y, np.ones(batch_size) * int(conv_shape[1] - 2),
#                np.ones(batch_size) * n_len], np.ones(batch_size)

gen = HDF5DatasetGenerator('vat_dates.hdf5', batch_size=32).generator

callbacks = [
    # Interrupts training when improvement stops
    keras.callbacks.EarlyStopping(
        # Monitors the model’s validation accuracy
        monitor='acc',
        # Interrupts training when accuracy has stopped
        # improving for more than one epoch (that is, two epochs)
        patience=10,
    ),
    # Saves the current weights after every epoch
    keras.callbacks.ModelCheckpoint(
        # Path to the destination model file
        filepath='vat_model.h5',
        # These two arguments mean you won’t overwrite the
        # model file unless val_loss has improved, which allows
        # you to keep the best model seen during training.
        monitor='val_loss',
        save_best_only=True
    )
]
crnn = CRNN.build(max_string_len=11)
crnn.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adadelta')
callbacks = []
while True:
    crnn.fit_generator(gen(), steps_per_epoch=1000, callbacks=callbacks, validation_data=gen(), validation_steps=10)
