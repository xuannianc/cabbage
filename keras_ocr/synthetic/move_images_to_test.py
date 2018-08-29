from keras_ocr.synthetic.config import DATASET_DIR, TEST_DIR
import os.path as osp
import shutil
import os

images_dir = osp.join(DATASET_DIR, 'images')
test_txt_path = osp.join(DATASET_DIR, 'test.txt')

with open(test_txt_path) as f:
    lines = f.read().split('\n')
    for line in lines:
        test_file = line.split(' ')[0]
        test_file_path = osp.join(images_dir, test_file)
        shutil.move(test_file_path, TEST_DIR)

print(len(os.listdir(TEST_DIR)))
print(len(os.listdir(images_dir)))
