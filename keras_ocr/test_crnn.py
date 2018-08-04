from keras.models import load_model, Model
from keras.layers import Input, Dense, Flatten
import cv2
import imutils
from keras import backend as K
import numpy as np

crnn = load_model('vat_model.h5', custom_objects={'<lambda>': lambda y_true, y_pred: y_pred})
output = crnn.get_layer('blstm2_out').output
base_model = Model(input=crnn.input[0], output=output)
image = cv2.imread('/home/adam/Pictures/vat/vat_dates/1100162350_12093275_20180427_299606_2.jpg', 0)
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
out = K.get_value(ctc_decode)[:, :11]
print(out)
