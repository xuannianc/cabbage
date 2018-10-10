from keras.models import Model, load_model
from keras.layers import Input, Dense, Lambda
import keras.backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from ocr_recognition.common.callback import TrainingMonitor, LearningRateUpdator, AccuracyEvaluator
from ocr_recognition.check_code.hdf5 import HDF5DatasetGenerator
from ocr_recognition.check_code.config import *
from keras.optimizers import RMSprop, SGD


def ctc_lambda_func(args):
    # https://www.tensorflow.org/api_docs/python/tf/keras/backend/ctc_batch_cost
    y_true, y_pred, input_length, label_length = args
    return K.ctc_batch_cost(y_true, y_pred, input_length, label_length)


def inspect_layers(model):
    for (i, layer) in enumerate(model.layers):
        print("[INFO] {}\t{}\t{}".format(i, layer.__class__.__name__, layer.name))


def get_new_model(crnn_model_path, num_classes, max_string_len):
    crnn = load_model(crnn_model_path,
                      custom_objects={'<lambda>': lambda y_true, y_pred: y_pred})
    x = crnn.get_layer('blstm2').output
    y_pred = Dense(num_classes, name='blstm2_out', activation='softmax')(x)
    label = Input(name='label', shape=[max_string_len], dtype='int64')
    seq_length = Input(name='seq_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,),
                      name='ctc')([label, y_pred, seq_length, label_length])
    model = Model(inputs=[crnn.input[0], label, seq_length, label_length], outputs=[loss_out])
    # inspect_layers(crnn)
    # freeze Dense 前面的所有层
    for layer in model.layers:
        if layer.name == 'blstm2':
            break
        else:
            layer.trainable = False
    model.summary()
    return model


model = get_new_model('../synthetic/models/synthetic_model_0810_3000000_0.0765_0.1086.hdf5', NUM_CLASSES, MAX_STRING_LEN)
# opt = RMSprop(lr=0.01)
opt = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=opt)
gen = HDF5DatasetGenerator(TRAIN_DB_PATH, batch_size=50, seq_len=124, label_len=23).generator
val_gen = HDF5DatasetGenerator(VALIDATION_DB_PATH, batch_size=10, seq_len=124, label_len=23).generator

# callbacks
training_monitor = TrainingMonitor(figure_path='check_code_ft_0910_1000.jpg', json_path='check_code_ft_0910_1000.json',
                                   start_at=0)
# accuracy_evaluator = AccuracyEvaluator(TEST_DB_PATH, batch_size=100)
learning_rate_updator = LearningRateUpdator(init_lr=0.01)
callbacks = [
    # Interrupts training when improvement stops
    EarlyStopping(
        # Monitors the model’s validation accuracy
        monitor='val_loss',
        # Interrupts training when accuracy has stopped
        # improving for more than one epoch (that is, two epochs)
        patience=10,
    ),
    # Saves the current weights after every epoch
    ModelCheckpoint(
        # Path to the destination model file
        filepath='models/check_code_ft_model_0910_1000.hdf5',
        # These two arguments mean you won’t overwrite the
        # model file unless val_loss has improved, which allows
        # you to keep the best model seen during training.
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    ),
    training_monitor,
    # accuracy_evaluator
    # learning_rate_updator
]
model.fit_generator(gen(), steps_per_epoch=1000,
                    callbacks=callbacks,
                    epochs=100,
                    validation_data=val_gen(),
                    validation_steps=200)
