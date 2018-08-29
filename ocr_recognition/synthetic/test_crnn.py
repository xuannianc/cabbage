from keras.models import load_model, Model
from keras.layers import Input, Dense, Flatten
import cv2
import imutils
from keras import backend as K
import numpy as np
import glob
import os.path as osp

DATASET_DIR = '/home/adam/.keras/datasets/synthetic_chinese_string'
with open('char_std_5990.txt') as f:
    chars = f.read().split('\n')

crnn = load_model('synthetic_model_0808_100000_0.695.hdf5', custom_objects={'<lambda>': lambda y_true, y_pred: y_pred})
# crnn = load_model('vat_model.hdf5', custom_objects={'<lambda>': lambda y_true, y_pred: y_pred})
output = crnn.get_layer('blstm2_out').output
base_model = Model(input=crnn.input[0], output=output)
with open(osp.join(DATASET_DIR, 'test.txt')) as f:
    lines = f.read().split('\n')
    image_files = [line.split(' ')[0] for line in lines]
    image_labels =  [line.split(' ')[1:] for line in lines]
for image_file,image_label in zip(image_files, image_labels):
    image = cv2.imread(osp.join(DATASET_DIR, 'test', image_file), 0)
    image = image.reshape((32, 280, 1))
    input = np.expand_dims(image, axis=0)
    y_pred = base_model.predict(input)
    shape = y_pred[:, :, :].shape
    ctc_decode = K.ctc_decode(y_pred[:, :, :],
                              input_length=np.ones(shape[0]) * shape[1])[0][0]
    out = K.get_value(ctc_decode)
    # out = K.get_value(ctc_decode)[:, :11]
    print('image_path={}'.format(image_file))
    print('scan_result={}'.format(out))
    print('image_label={}'.format(image_label))
    cv2.imshow('image', image)
    cv2.waitKey(0)
