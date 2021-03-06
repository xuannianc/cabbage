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
import random

logger = logging.getLogger('annotate')
logger.setLevel(logging.DEBUG)  # default log level
format = logging.Formatter("%(asctime)s %(name)-8s %(levelname)-8s %(lineno)-4d %(message)s")  # output format
sh = logging.StreamHandler(stream=sys.stdout)  # output to standard output
sh.setFormatter(format)
logger.addHandler(sh)

vat_number_train_writer = HDF5DatasetWriter(data_dims=(6000, 28, 28, 1),
                                            label_dims=(6000, 10),
                                            output_path='data/vat_number_train_1130_6000.hdf5')
vat_number_val_writer = HDF5DatasetWriter(data_dims=(2000, 28, 28, 1),
                                          label_dims=(2000, 10),
                                          output_path='data/vat_number_val_1130_2000.hdf5')

sdl = SimpleDatasetLoader()
train_image_paths = list(paths.list_images('../../datasets/train_digits'))
random.shuffle(train_image_paths)
(x_train, y_train) = sdl.load(train_image_paths, verbose=1000)
x_train = x_train.astype("float") / 255.0
val_image_paths = list(paths.list_images('../../datasets/val_digits'))
random.shuffle(val_image_paths)
val_image_paths = val_image_paths[:2000]
(x_val, y_val) = sdl.load(val_image_paths, verbose=1000)
x_val = x_val.astype("float") / 255.0
logger.debug('{} {} {} {}'.format(x_train.shape, y_train.shape, x_val.shape, y_val.shape))
y_train = LabelBinarizer().fit_transform(y_train)
y_val = LabelBinarizer().fit_transform(y_val)
logger.debug('{} {} {} {}'.format(x_train.shape, y_train.shape, x_val.shape, y_val.shape))
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_val = x_val.reshape(x_val.shape[0], 28, 28, 1)
logger.debug('{} {} {} {}'.format(x_train.shape, y_train.shape, x_val.shape, y_val.shape))
x_train = x_train.tolist()
y_train = y_train.tolist()
x_val = x_val.tolist()
y_val = y_val.tolist()
vat_number_train_writer.add(x_train, y_train)
vat_number_val_writer.add(x_val, y_val)
vat_number_train_writer.close()
vat_number_val_writer.close()
