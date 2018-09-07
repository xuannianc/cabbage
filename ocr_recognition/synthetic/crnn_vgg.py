from keras.optimizers import Adam, SGD, Adadelta
import keras.backend.tensorflow_backend as K
from keras import layers
from keras import models
from keras import callbacks
from ocr_recognition.synthetic.hdf5_2 import HDF5DatasetGenerator
from ocr_recognition.synthetic.callback import *
from ocr_recognition.synthetic.config import *


def ctc_lambda_func(args):
    # https://www.tensorflow.org/api_docs/python/tf/keras/backend/ctc_batch_cost
    y_true, y_pred, input_length, label_length = args
    return K.ctc_batch_cost(y_true, y_pred, input_length, label_length)


def crnn(input_shape=(32, 280, 1), rnn_unit=256, num_classes=5991, max_string_len=10):
    input = layers.Input(shape=input_shape, name='input')
    m = layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', name='conv1')(input)
    m = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1')(m)
    m = layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same', name='conv2')(m)
    m = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool2')(m)
    m = layers.Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same', name='conv3')(m)
    m = layers.Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same', name='conv4')(m)
    m = layers.MaxPooling2D(pool_size=(2, 1), strides=(2, 1), padding='valid', name='pool3')(m)
    m = layers.Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same', name='conv5')(m)
    m = layers.BatchNormalization(axis=3)(m)
    m = layers.Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same', name='conv6')(m)
    m = layers.BatchNormalization(axis=3)(m)
    m = layers.MaxPooling2D(pool_size=(2, 1), strides=(2, 1), padding='valid', name='pool4')(m)
    m = layers.Conv2D(512, kernel_size=(2, 2), activation='relu', padding='valid', name='conv7')(m)
    m = layers.Permute((2, 1, 3), name='permute')(m)
    m = layers.wrappers.TimeDistributed(layers.Flatten(), name='flatten')(m)
    m = layers.wrappers.Bidirectional(layers.LSTM(rnn_unit, return_sequences=True, implementation=2), name='blstm1')(m)
    m = layers.wrappers.Bidirectional(layers.LSTM(rnn_unit, return_sequences=True, implementation=2), name='blstm2')(m)
    y_pred = layers.wrappers.TimeDistributed(layers.Dense(num_classes, activation='softmax'), name='base_model_output')(m)
    base_model = models.Model(inputs=input, outputs=y_pred)
    label = layers.Input(name='label', shape=[max_string_len], dtype='int64')
    # 序列的长度,此模型中为 69
    seq_length = layers.Input(name='seq_length', shape=[1], dtype='int64')
    label_length = layers.Input(name='label_length', shape=[1], dtype='int64')
    loss_out = layers.Lambda(ctc_lambda_func, output_shape=(1,),
                      name='ctc')([label, y_pred, seq_length, label_length])
    model = models.Model(inputs=[input, label, seq_length, label_length], outputs=[loss_out])
    model.summary()
    return base_model, model


base_model, model = crnn()
# clipnorm seems to speeds up convergence
sgd = SGD(lr=0.1, decay=1e-3, momentum=0.9, nesterov=True, clipnorm=5)
model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd)
# model = models.load_model('models/synthetic_model_0829_3000000_1.3227_1.0404.hdf5',
#                  custom_objects={'<lambda>': lambda y_true, y_pred: y_pred})
gen = HDF5DatasetGenerator(TRAIN_DB_PATH, batch_size=300).generator
val_gen = HDF5DatasetGenerator(VALIDATION_DB_PATH, batch_size=100).generator

# callbacks
# training_monitor = TrainingMonitor(figure_path='synthetic_0829_3000000.jpg', json_path='synthetic_0829_3000000.json', start_at=5)
# accuracy_evaluator = AccuracyEvaluator(TEST_DB_PATH, batch_size=100)
learning_rate_updator = LearningRateUpdator(init_lr=0.01)
callbacks = [
    # Interrupts training when improvement stops
    callbacks.EarlyStopping(
        # Monitors the model’s validation accuracy
        monitor='val_loss',
        # Interrupts training when accuracy has stopped
        # improving for more than one epoch (that is, two epochs)
        patience=10,
    ),
    # Saves the current weights after every epoch
    callbacks.ModelCheckpoint(
        # Path to the destination model file
        filepath='models/synthetic_model_0829_3000000.hdf5',
        # These two arguments mean you won’t overwrite the
        # model file unless val_loss has improved, which allows
        # you to keep the best model seen during training.
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    ),
    # training_monitor,
    # accuracy_evaluator
    learning_rate_updator
]
model.fit_generator(gen(), steps_per_epoch=3000000 // 300,
                    callbacks=callbacks,
                    epochs=100,
                    validation_data=val_gen(),
                    validation_steps=279600 // 100)
