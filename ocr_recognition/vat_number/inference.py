from keras.models import load_model
import imutils
from imutils import paths
import cv2
import numpy as np
import os
from pprint import pprint
import os.path as osp
import shutil
import json
import random


# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# vat_num_model = load_model('models/vat_number_1015_4600_1500_0.0525_0.9923_0.0145_0.9980.hdf5')


def verify_train_data():
    misunderstand_digits = {}
    for image_path in paths.list_images('/home/adam/workspace/github/cabbage/datasets/digits'):
        label = image_path.split('/')[-2]
        image = cv2.imread(image_path, 0)
        image = image.reshape(28, 28, 1)
        input = np.expand_dims(image, axis=0)
        pred = vat_num_model.predict(input).argmax(axis=1).tolist()[0]
        if label != str(pred):
            if misunderstand_digits.get(label):
                misunderstand_digits[label].append((image_path, pred))
            else:
                misunderstand_digits[label] = []
                misunderstand_digits[label].append((image_path, pred))
    pprint(misunderstand_digits)


# verify_train_data()
def move_out_wrong_data():
    backup_dir = '/home/adam/workspace/github/cabbage/datasets/wrong'
    for digit in range(10):
        for image_path in paths.list_images('/home/adam/Pictures/vat/misunderstand/number/3/' + str(digit)):
            print('Handing image_path {} starts'.format(image_path))
            image_file = image_path.split('/')[-1]
            label = image_path.split('/')[-2]
            image = cv2.imread(image_path)
            cv2.namedWindow(label, cv2.WINDOW_NORMAL)
            cv2.imshow(label, image)
            key = cv2.waitKey(0)
            if 48 <= key <= 57:
                new_image_dir = osp.join(backup_dir, str(key - 48))
                new_image_path = osp.join(new_image_dir, image_file)
                print('move {} to {}'.format(image_path, new_image_path))
                shutil.move(image_path, new_image_path)
            print('Handing image_path {} ends'.format(image_path))


# move_out_wrong_data()

def add_data_to_dataset():
    counts = json.load(open('counts.json'))
    dataset_dir = '/home/adam/workspace/github/cabbage/datasets/digits'
    for digit in range(10):
        for image_path in paths.list_images('/home/adam/Pictures/vat/misunderstand/number/3/' + str(digit)):
            print('Handing image_path {} starts'.format(image_path))
            label = image_path.split('/')[-2]
            count = counts.get(label)
            digit_dir = osp.join(dataset_dir, label)
            new_image_path = osp.join(digit_dir, "{}.jpg".format(str(count + 1).zfill(6)))
            print('move {} to {}'.format(image_path, new_image_path))
            shutil.move(image_path, new_image_path)
            counts[label] = count + 1
            json.dump(counts, open('counts.json', 'w'))
    json.dump(counts, open('counts.json', 'w'))


# add_data_to_dataset()

def generate_train_digits():
    train_digits_dir = '/home/adam/workspace/github/cabbage/datasets/train_digits'
    val_digits_dir = '/home/adam/workspace/github/cabbage/datasets/val_digits'
    if not osp.exists(train_digits_dir):
        os.mkdir(train_digits_dir)
    else:
        shutil.rmtree(train_digits_dir)
        os.mkdir(train_digits_dir)
    if not osp.exists(val_digits_dir):
        os.mkdir(val_digits_dir)
    else:
        shutil.rmtree(val_digits_dir)
        os.mkdir(val_digits_dir)
    dataset_dir = '/home/adam/workspace/github/cabbage/datasets/digits'
    for digit in range(10):
        image_files = os.listdir(osp.join(dataset_dir, str(digit)))
        random.shuffle(image_files)
        train_image_files = image_files[:600]
        val_image_files = image_files[600:]
        train_digit_dir = osp.join(train_digits_dir, str(digit))
        val_digit_dir = osp.join(val_digits_dir, str(digit))
        if not osp.exists(train_digit_dir):
            os.mkdir(train_digit_dir)
        if not osp.exists(val_digit_dir):
            os.mkdir(val_digit_dir)
        for image_file in train_image_files:
            shutil.copy(osp.join(dataset_dir, str(digit), image_file), osp.join(train_digit_dir, image_file))
        for image_file in val_image_files:
            shutil.copy(osp.join(dataset_dir, str(digit), image_file), osp.join(val_digit_dir, image_file))

generate_train_digits()
