from keras.models import load_model, Model
from keras.layers import Input, Dense, Flatten
import cv2
import imutils
from keras import backend as K
import numpy as np
import glob
from imutils import paths
import os.path as osp
from keras_ocr.synthetic.config import DATASET_DIR
from util import extract_text
from PIL import Image, ImageDraw, ImageFont
import time

TEST_TXT_PATH = osp.join(DATASET_DIR, 'test.txt')
with open('char_std_5990.txt') as f:
    chars = f.read().split('\n')
crnn = load_model('synthetic_model_0810_3000000_0.0765_0.1086.hdf5', custom_objects={'<lambda>': lambda y_true, y_pred: y_pred})
output = crnn.get_layer('blstm2_out').output
base_model = Model(inputs=crnn.input[0], outputs=output)
num_true = 0
num_false = 0
with open(TEST_TXT_PATH) as f:
    for line in f.read().split('\n'):
        splits = line.split(' ')
        image_file = splits[0]
        y_true = [int(value) for value in splits[1:]]
        image = cv2.imread(osp.join(DATASET_DIR, 'test', image_file), 0)
        image = imutils.resize(image, height=32)
        image_height, image_width = image.shape[:2]
        image = image.reshape((image_height, image_width, 1))
        input = np.expand_dims(image, axis=0)
        y_pred = base_model.predict(input)
        shape = y_pred[:, :, :].shape
        ctc_decode = K.ctc_decode(y_pred[:, :, :],
                                  input_length=np.ones(shape[0]) * shape[1])[0][0]
        y_pred = K.get_value(ctc_decode)[0].tolist()
        print('y_true={}'.format(y_true))
        print('y_pred={}'.format(y_pred))
        if y_pred == y_true:
            num_true += 1
        else:
            num_false += 1
print('accuracy={}'.format(num_true * 1.0 / (num_true + num_false)))

