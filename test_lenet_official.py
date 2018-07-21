from keras.models import load_model
import cv2
from imutils import paths
import imutils
import numpy as np
from sklearn.metrics import classification_report
from dataset_loaders import SimpleDatasetLoader
from sklearn.preprocessing import LabelBinarizer

model = load_model('models/hdf5/mnist_lenet_official.hdf5')
print(model.summary())
sdl = SimpleDatasetLoader()
image_paths = list(paths.list_images('./datasets/digits'))
x_test, y_test = sdl.load(image_paths)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
x_test = x_test.astype('float32')
x_test /= 255
print(x_test.shape)
print(y_test.shape)
le = LabelBinarizer()
y_test = le.fit_transform(y_test)
print('class_labels:{}'.format(le.classes_))
print(y_test.shape)
# evaluate the network
print("[INFO] evaluating network...")
# predictions = model.predict(x_test, batch_size=32)
# print(predictions.shape)
# print(classification_report(y_test.argmax(axis=1),
#                             predictions.argmax(axis=1),
#                             target_names=[str(x) for x in le.classes_]))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
