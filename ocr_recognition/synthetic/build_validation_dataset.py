import cv2
from ocr_recognition.synthetic.hdf5_2 import HDF5DatasetWriter
from ocr_recognition.synthetic.config import *
import os.path as osp

output_path = 'data/synthetic_validation_0829_279600.hdf5'
if osp.exists(output_path):
    print('{} already exists'.format(output_path))
    exit(-1)
hdf5_writer = HDF5DatasetWriter(data_dims=(279600, 32, 280, 1),
                                label_dims=(279600, 10),
                                output_path=output_path)
num_images = 0
with open(osp.join(DATASET_DIR, 'train.txt')) as f:
    for idx, line in enumerate(f.read().split('\n')[3000000:3279600]):
        image_file = line.split(' ')[0]
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
        height, width = image.shape[:2]
        if width != 280 or height != 32:
            print('{} is of different size:height={},width={}'.format(image_file, height, width))
            continue
        image = image.reshape((32, 280, 1))
        image = image.astype('float32') / 255.0
        label = [int(word_idx) for word_idx in line.split(' ')[1:]]
        if len(label) != 10:
            print('{} label is of different length:length={}'.format(image_file, len(label)))
            continue
        num_images += 1
        hdf5_writer.add([image], [label])
    print('num_images={}'.format(num_images))
    hdf5_writer.close()
