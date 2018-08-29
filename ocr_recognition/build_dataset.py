from imutils import paths
import imutils
import cv2
from keras_ocr.hdf5 import HDF5DatasetWriter
import os
import numpy as np

hdf5_writer = HDF5DatasetWriter(data_dims=(1076, 32, 248, 1),
                                label_dims=(1076, 11),
                                output_path='vat_dates_0807_1076.hdf5')
for image_path in paths.list_images('/home/adam/Pictures/vat/vat_dates/'):
    image_file = os.path.split(image_path)[1]
    image = cv2.imread(image_path, 0)
    image = imutils.resize(image, height=32)
    height, width = image.shape[:2]
    if width > 248:
        continue
    padding_width = (248 - width) // 2
    image = cv2.copyMakeBorder(image, 0, 0, padding_width, padding_width,
                               cv2.BORDER_REPLICATE)
    image = cv2.resize(image, (248, 32))
    image = image.reshape((32, 248, 1))
    label = []
    vat_date = image_file.split('_')[2]
    # [2,0,1,7,1,2,1,4]
    for c in vat_date:
        label.append(int(c))
    # [2,0,1,7,10,1,2,1,4]
    label.insert(4, 10)
    # [2,0,1,7,10,1,2,11,1,4]
    label.insert(7, 11)
    # [2,0,1,7,10,1,2,11,1,4,12]
    label.append(12)
    hdf5_writer.add([image], [label])
hdf5_writer.close()
