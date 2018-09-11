import cv2
from ocr_recognition.check_code.hdf5 import HDF5DatasetWriter
from ocr_recognition.check_code.config import *
import os.path as osp
import imutils

output_path = 'data/validation_0910_200.hdf5'
if osp.exists(output_path):
    print('{} already exist'.format(output_path))
    exit(-1)
hdf5_writer = HDF5DatasetWriter(data_dims=(200, 32, 500, 1),
                                label_dims=(200, 23),
                                output_path=output_path)
num_images = 0
with open('check_code_labels.txt') as f:
    for idx, line in enumerate(f.read().split('\n')[1000:1200]):
        try:
            image_file = line.split(',')[0]
            image_file_path = osp.join(TRAIN_DIR, image_file)
            if not osp.exists(image_file_path):
                print('imaga_file_path={} does not exist'.format(image_file_path))
                print(idx)
                continue
            image = cv2.imread(image_file_path, 0)
            if image is None:
                print('imaga_file_path={} is empty'.format(image_file_path))
                print(idx)
                continue
            image = imutils.resize(image, height=32)
            height, width = image.shape[:2]
            padding_width = (500 - width) // 2
            image = cv2.copyMakeBorder(image, 0, 0, padding_width, 500 - width - padding_width,
                                       cv2.BORDER_REPLICATE)
            image = image.reshape((32, 500, 1))
            image = image.astype('float32') / 255.0
            label = [int(word_idx) for word_idx in line.split(',')[1:]]
            if len(label) != 23:
                print('{} label is of different length:length={}'.format(image_file, len(label)))
                continue
            num_images += 1
            hdf5_writer.add([image], [label])
        except Exception as e:
            print(idx)
    print('num_images={}'.format(num_images))
    hdf5_writer.close()
