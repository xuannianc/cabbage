from keras.models import load_model
import cv2
import imutils
import numpy as np

model = load_model('models/hdf5/mnist_lenet_official.hdf5')
test_imgage = cv2.imread('datasets/digits/0/000001.jpg', 0)
roi_h28 = imutils.resize(test_imgage, height=28)
roi_h28_width = roi_h28.shape[1]
roi_h28_width_margin = (28 - roi_h28_width) // 2
roi_margin = cv2.copyMakeBorder(roi_h28, 0, 0, roi_h28_width_margin, roi_h28_width_margin,
                                cv2.BORDER_REPLICATE)
roi = cv2.resize(roi_margin, (28, 28))
roi = roi.reshape((28, 28, 1))
roi = np.expand_dims(roi, axis=0)
result = model.predict(roi)
print(result)
