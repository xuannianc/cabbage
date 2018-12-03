from imutils import paths
import imutils
import cv2
from ocr_recognition.vat_number.hdf5 import HDF5DatasetWriter
import os
import numpy as np
import os.path as osp
from sklearn.preprocessing import LabelBinarizer
from dataset_loaders import SimpleDatasetLoader
from sklearn.model_selection import train_test_split
import logging
import sys

logger = logging.getLogger('annotate')
logger.setLevel(logging.DEBUG)  # default log level
format = logging.Formatter("%(asctime)s %(name)-8s %(levelname)-8s %(lineno)-4d %(message)s")  # output format
sh = logging.StreamHandler(stream=sys.stdout)  # output to standard output
sh.setFormatter(format)
logger.addHandler(sh)

vat_number_train_writer = HDF5DatasetWriter(data_dims=(4600, 28, 28, 1),
                                            label_dims=(4600, 10),
                                            output_path='data/vat_number_train_1015_4600.hdf5')
vat_number_val_writer = HDF5DatasetWriter(data_dims=(1500, 28, 28, 1),
                                          label_dims=(1500, 10),
                                          output_path='data/vat_number_val_1015_1500.hdf5')

sdl = SimpleDatasetLoader()
image_paths = list(paths.list_images('../../datasets/digits'))
(data, labels) = sdl.load(image_paths, verbose=1000)
data = data.astype("float") / 255.0
(x_train, x_val, y_train, y_val) = train_test_split(data, labels, test_size=0.25, random_state=42)
logger.debug('{} {} {} {}'.format(x_train.shape, y_train.shape, x_val.shape, y_val.shape))
y_train = LabelBinarizer().fit_transform(y_train)
y_val = LabelBinarizer().fit_transform(y_val)
logger.debug('{} {} {} {}'.format(x_train.shape, y_train.shape, x_val.shape, y_val.shape))
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_val = x_val.reshape(x_val.shape[0], 28, 28, 1)
logger.debug('{} {} {} {}'.format(x_train.shape, y_train.shape, x_val.shape, y_val.shape))
x_train = x_train.tolist()[:4600]
y_train = y_train.tolist()[:4600]
x_val = x_val.tolist()[:1500]
y_val = y_val.tolist()[:1500]
vat_number_train_writer.add(x_train, y_train)
vat_number_val_writer.add(x_val, y_val)
