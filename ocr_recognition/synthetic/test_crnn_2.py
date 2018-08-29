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

DATASET_DIR = '/home/adam/Pictures/vat_test'
with open('char_std_5990.txt') as f:
    chars = f.read().split('\n')
crnn = load_model('synthetic_model_0810_3000000_0.0765_0.1086.hdf5', custom_objects={'<lambda>': lambda y_true, y_pred: y_pred})
# crnn = load_model('vat_model.hdf5', custom_objects={'<lambda>': lambda y_true, y_pred: y_pred})
output = crnn.get_layer('blstm2_out').output
base_model = Model(inputs=crnn.input[0], outputs=output)
for image_path in paths.list_images(osp.join(DATASET_DIR, 'test')):
    image = cv2.imread(image_path, 0)
    image = imutils.resize(image, height=32)
    image_height, image_width = image.shape[:2]
    image = image.reshape((image_height, image_width, 1))
    input = np.expand_dims(image, axis=0)
    y_pred = base_model.predict(input)
    shape = y_pred[:, :, :].shape
    ctc_decode = K.ctc_decode(y_pred[:, :, :],
                              input_length=np.ones(shape[0]) * shape[1])[0][0]
    out = K.get_value(ctc_decode)
    # out = K.get_value(ctc_decode)[:, :11]
    print('image_path={}'.format(image_path))
    result1 = ''.join([chars[idx - 1] for idx in out[0]])
    print('my_result={}'.format(result1))
    # cv2.imshow('image', image)
    # cv2.waitKey(0)
    image = cv2.imread(image_path)
    result2 = extract_text(image)
    print('tesseract_result={}'.format(result2))
    height, width = image.shape[:2]
    image = np.insert(image, [height] * 300, values=255, axis=0)
    image = np.insert(image, [width] * 200, values=255, axis=1)
    image_PIL = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # font = ImageFont.truetype('NotoSansCJK-Black.ttc', 20)
    font = ImageFont.truetype('simsun.ttc', 20)
    # 字体颜色
    fill_color = (255, 0, 0)
    # 文字输出位置
    position1 = (10, 100)
    position2 = (10, 200)
    draw = ImageDraw.Draw(image_PIL)
    draw.text(position1, "okra:{}".format(result1), font=font, fill=fill_color)
    draw.text(position2, "tesseract:{}".format(result2), font=font, fill=fill_color)
    # 转换回OpenCV格式
    image = cv2.cvtColor(np.asarray(image_PIL), cv2.COLOR_RGB2BGR)
    cv2.imshow('image', image)
    cv2.waitKey(0)
