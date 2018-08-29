from keras.models import load_model, Model
from keras.layers import Input, Dense, Flatten
import cv2
import imutils
from keras import backend as K
import numpy as np
import glob

crnn = load_model('vat_model_0807_1076.hdf5', custom_objects={'<lambda>': lambda y_true, y_pred: y_pred})
# crnn = load_model('vat_model.hdf5', custom_objects={'<lambda>': lambda y_true, y_pred: y_pred})
output = crnn.get_layer('blstm2_out').output
base_model = Model(input=crnn.input[0], output=output)
for image_path in glob.glob('/home/adam/Pictures/vat_other/2017/11/7/*_2.jpg'):
    image = cv2.imread(image_path, 0)
    image = imutils.resize(image, height=32)
    height, width = image.shape[:2]
    padding_width = (248 - width) // 2
    image = cv2.copyMakeBorder(image, 0, 0, padding_width, padding_width,
                               cv2.BORDER_REPLICATE)
    image = cv2.resize(image, (248, 32))
    image = image.reshape((32, 248, 1))
    input = np.expand_dims(image, axis=0)
    y_pred = base_model.predict(input)
    shape = y_pred[:, :, :].shape
    ctc_decode = K.ctc_decode(y_pred[:, :, :],
                              input_length=np.ones(shape[0]) * shape[1])[0][0]
    out = K.get_value(ctc_decode)
    # out = K.get_value(ctc_decode)[:, :11]
    print('image_path={}'.format(image_path))
    print('scan_result={}'.format(out))
