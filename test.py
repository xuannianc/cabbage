import numpy as np
import imutils
from imutils import paths
import cv2
import os

# samples = ['The cat sat on the mat.', 'The dog ate my homework.']
# token_index = {}
# for sample in samples:
#     for word in sample.split():
#         if word not in token_index:
#             token_index[word] = len(token_index) + 1
# max_length = 10
# results = np.zeros(shape=(len(samples),
#                           max_length,
#                           max(token_index.values()) + 1))
# for i, sample in enumerate(samples):
#     for j, word in list(enumerate(sample.split()))[:max_length]:
#         index = token_index.get(word)
#         results[i, j, index] = 1
# print(results)
# print(token_index)


DATASET_DIR = '/home/adam/.keras/datasets/synthetic_chinese_string'
# image = cv2.imread(os.path.join(DATASET_DIR, 'test', '20456343_4045240981.jpg'))
# print(image.shape)
# cv2.imshow('xxx', image)
# cv2.waitKey(0)

# import shutil
#
# with open(os.path.join(DATASET_DIR, 'test.txt')) as f:
#     lines = f.read().split('\n')
#     print(len(lines))
#     for line in lines:
#         test_image_file = line.split(' ')[0]
#         shutil.move(os.path.join(DATASET_DIR,'images',test_image_file), os.path.join(DATASET_DIR, 'test'))
#         print('{} is moved'.format(test_image_file))

print(len(os.listdir(os.path.join(DATASET_DIR, 'test'))))
print(len(os.listdir(os.path.join(DATASET_DIR, 'train'))))

# for image_file_path in paths.list_images(os.path.join(DATASET_DIR, 'test')):
#    image = cv2.imread(image_file_path)
#    cv2.imshow('{}'.format(os.path.split(image_file_path)[-1]), image)
#    cv2.waitKey(0)
