from keras_ocr.synthetic.config import *
import os.path as osp
import shutil
import os
import cv2


def move_images_to_test_dir():
    print('image_dir_old={}'.format(len(os.listdir(IMAGE_DIR))))
    with open(TEST_TXT_PATH) as f:
        lines = f.read().split('\n')
        # lines 最后一个元素为 '', 所以要忽略
        for line in lines[:-1]:
            test_file = line.split(' ')[0]
            test_file_path = osp.join(IMAGE_DIR, test_file)
            shutil.move(test_file_path, TEST_DIR)
    print('test_dir={}'.format(len(os.listdir(TEST_DIR))))
    print('image_dir_new={}'.format(len(os.listdir(IMAGE_DIR))))


def find_spare_file_in_train_dir():
    """
    images 在去除掉 364400 个文件到 test 里面, 还剩 3279607, 重命名为 train.
    比 train.txt 里面多一个文件
    :return:
    """
    with open(TRAIN_TXT_PATH) as f:
        lines = f.read().split('\n')
        # lines 最后一个元素为 '', 所以要忽略
        files_in_txt = [line.split(' ')[0] for line in lines[:-1]]
        print('old file_in_txt len={}'.format(len(files_in_txt)))
        files_in_dir = [os.path.split(file_path)[1] for file_path in os.listdir(TRAIN_DIR)]
        print('old file_in_dir len={}'.format(len(files_in_dir)))
        spare_file = set(files_in_dir) - set(files_in_txt)
        print(spare_file[0])
    spare_file_path = osp.join(TRAIN_DIR, spare_file[0])
    spare_image = cv2.imread(spare_file_path)
    cv2.imshow('spare_image', spare_image)
    cv2.waitKey(0)
    # 删除
    files_in_dir = [os.path.split(file_path)[1] for file_path in os.listdir(TRAIN_DIR)]
    print('new file_in_dir len={}'.format(len(files_in_dir)))


# find_spare_file_in_train_dir()
