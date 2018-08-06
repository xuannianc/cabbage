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
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, Callback
from keras.utils import plot_model
from matplotlib import pyplot as plt
import numpy as np
import os
import json
import tensorflow as tf
import keras.backend.tensorflow_backend as K
from keras_ocr.hdf5 import HDF5DatasetGenerator
import keras
import h5py


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
        base_model = Model(inputs=input, outputs=y_pred)
        label = Input(name='label', shape=[max_string_len], dtype='int64')
        seq_length = Input(name='seq_length', shape=[1], dtype='int64')
        label_length = Input(name='label_length', shape=[1], dtype='int64')
        loss_out = Lambda(ctc_lambda_func, output_shape=(1,),
                          name='ctc')([label, y_pred, seq_length, label_length])
        model = Model(input=[input, label, seq_length, label_length], output=[loss_out])
        model.summary()
        return base_model, model


def ctc_lambda_func(args):
    # https://www.tensorflow.org/api_docs/python/tf/keras/backend/ctc_batch_cost
    y_true, y_pred, input_length, label_length = args
    return K.ctc_batch_cost(y_true, y_pred, input_length, label_length)


def evaluate(base_model, batch_num=10):
    batch_acc = 0
    generator = gen()
    for i in range(batch_num):
        [X_test, y_test, _, _], _ = next(generator)
        y_pred = base_model.predict(X_test)
        shape = y_pred[:, :, :].shape
        ctc_decode = K.ctc_decode(y_pred[:, :, :],
                                  input_length=np.ones(shape[0]) * shape[1])[0][0]
        out = K.get_value(ctc_decode)[:, :11]
        if out.shape[1] == 11:
            batch_acc += ((y_test == out).sum(axis=1) == 11).mean()
    return batch_acc / batch_num


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

# gen = HDF5DatasetGenerator('vat_dates.hdf5', batch_size=32).generator
gen = HDF5DatasetGenerator('vat_dates_0806_697.hdf5', batch_size=32).generator
base_model, crnn = CRNN.build(max_string_len=11)


class Evaluate(Callback):
    def __init__(self):
        self.accs = []

    def on_epoch_end(self, epoch, logs=None):
        acc = evaluate(base_model) * 100
        self.accs.append(acc)
        print('acc={}%'.format(acc))


evaluator = Evaluate()
callbacks = [
    # Interrupts training when improvement stops
    keras.callbacks.EarlyStopping(
        # Monitors the model’s validation accuracy
        # monitor='acc',
        # Interrupts training when accuracy has stopped
        # improving for more than one epoch (that is, two epochs)
        patience=10,
    ),
    # Saves the current weights after every epoch
    keras.callbacks.ModelCheckpoint(
        # Path to the destination model file
        filepath='vat_model_0806_697.hdf5',
        # These two arguments mean you won’t overwrite the
        # model file unless val_loss has improved, which allows
        # you to keep the best model seen during training.
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    ),
    #    keras.callbacks.TensorBoard(
    #        log_dir='log',
    #        histogram_freq=1
    #    ),
    evaluator
]

# clipnorm seems to speeds up convergence
sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
crnn.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd)
# crnn.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adadelta', metrics=['accuracy'])
H = crnn.fit_generator(gen(), steps_per_epoch=1000,
                       callbacks=callbacks,
                       epochs=100,
                       # validation_data=(
                       # [db['data'][:30], db['labels'][:30], np.ones(30) * (248 // 4 - 2), np.ones(30) * 11],
                       # np.ones(30)))
                       validation_data=gen(),
                       validation_steps=30)

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 100), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 100), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 100), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, 100), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()
