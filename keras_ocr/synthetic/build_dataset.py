from imutils import paths
import imutils
import cv2
from keras_ocr.hdf5 import HDF5DatasetWriter
import os
import numpy as np
import os.path as osp

hdf5_writer = HDF5DatasetWriter(data_dims=(279606, 32, 280, 1),
                                label_dims=(279606, 10),
                                output_path='synthetic_validation_0808_279606.hdf5')
DATASET_DIR = '/home/adam/.keras/datasets/synthetic_chinese_string'
idx = 0
with open(osp.join(DATASET_DIR, 'train.txt')) as f:
    for line in f.read().split('\n')[3000000:]:
        image_file = line.split(' ')[0]
        image_file_path = osp.join(DATASET_DIR, 'train', image_file)
        if not osp.exists(image_file_path):
            print('imaga_file_path={} does not exist'.format(image_file_path))
            print(idx)
            continue
        image = cv2.imread(image_file_path, 0)
        if image is None:
            print('imaga_file_path={} is empty'.format(image_file_path))
            print(idx)
            continue
        height, width = image.shape[:2]
        if width != 280 or height != 32:
            print('{} is of different size:height={},width={}'.format(image_file, height, width))
            continue
        image = image.reshape((32, 280, 1))
        label = [int(word_idx) for word_idx in line.split(' ')[1:]]
        if len(label) != 10:
            print('{} label is of different length:length={}'.format(image_file, len(label)))
        hdf5_writer.add([image], [label])
        idx += 1
    print(idx)
    hdf5_writer.close()
