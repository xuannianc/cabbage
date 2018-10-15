import imutils
from imutils import paths
import os
import glob
import cv2
import os.path as osp


# for image_file_path in paths.list_images('/home/adam/Pictures/vat_test'):
#     image_file_dir, image_file = os.path.split(image_file_path)
#     if not os.path.exists(os.path.join('/home/adam/Pictures/vat', image_file)):
#         print(image_file_path)
def find_vat_without_vat_date(vat_dir, vat_date_dir):
    print('********** nonexist vat date file **********')
    for vat_file_path in glob.glob(vat_dir + '/*.jpg'):
        vat_file = os.path.split(vat_file_path)[1]
        vat_file_name = os.path.splitext(vat_file)[0]
        vat_date_file_name = vat_file_name + '_2'
        vat_date_file = vat_date_file_name + '.jpg'
        vat_date_file_path = os.path.join(vat_date_dir, vat_date_file)
        if not os.path.exists(vat_date_file_path):
            print(vat_file)
    print('********** nonexist vat date file **********')


def find_vat_date_without_vat(vat_dir, vat_date_dir):
    print('********** nonexist vat file **********')
    for vat_date_file_path in glob.glob(vat_date_dir + '/*.jpg'):
        vat_date_file = os.path.split(vat_date_file_path)[1]
        vat_date_file_name = os.path.splitext(vat_date_file)[0]
        vat_file_name = vat_date_file_name[:-2]
        vat_file = vat_file_name + '.jpg'
        vat_file_path = os.path.join(vat_dir, vat_file)
        if not os.path.exists(vat_file_path):
            print(vat_date_file)
    print('********** nonexist vat file **********')


def find_vat_without_check_code(vat_dir, check_code_dir):
    print('********** nonexist vat check code file **********')
    vat_without_check_code = []
    for vat_file_path in glob.glob(vat_dir + '/*.jpg'):
        # 查找没有对应 check_code 的 vat
        vat_file = os.path.split(vat_file_path)[1]
        vat_file_name = os.path.splitext(vat_file)[0]
        vat_check_code_file_name = vat_file_name + '_4'
        vat_check_code_file = vat_check_code_file_name + '.jpg'
        vat_check_code_file_path = os.path.join(check_code_dir, vat_check_code_file)
        if len(vat_file_name.split('_')) == 3:
            # 专票
            continue
        elif len(vat_file_name.split('_')) == 4:
            # 普票
            if not os.path.exists(vat_check_code_file_path):
                vat_without_check_code.append(vat_file)
                image = cv2.imread(osp.join(vat_dir, vat_file))
                cv2.namedWindow('{}'.format(vat_file), cv2.WINDOW_NORMAL)
                cv2.imshow('{}'.format(vat_file), image)
                cv2.waitKey(0)
        else:
            print('{} 名字不规范'.format(vat_file_name))
    print('\n'.join(sorted(vat_without_check_code)))
    print('********** nonexist vat check code file **********')


def find_check_code_without_vat(vat_dir, check_code_dir):
    print('********** nonexist vat file **********')
    for vat_check_code_file_path in glob.glob(check_code_dir + '/*.jpg'):
        vat_check_code_file = os.path.split(vat_check_code_file_path)[1]
        vat_check_code_file_name = os.path.splitext(vat_check_code_file)[0]
        # 去掉最后的 _4
        vat_file_name = vat_check_code_file_name[:-2]
        vat_file = vat_file_name + '.jpg'
        vat_file_path = os.path.join(vat_dir, vat_file)
        if not os.path.exists(vat_file_path):
            print(vat_check_code_file)
    print('********** nonexist vat file **********')


# 查找存在于子目录中但是不存在于总目录的 vat 文件
def find_vat_in_one_dir_but_not_in_another_dir(one_dir, another_dir):
    print('********** nonexist vat file **********')
    nonexist_files = []
    for one_vat_file_path in glob.glob(one_dir + '/*.jpg'):
        one_vat_file = os.path.split(one_vat_file_path)[1]
        another_vat_file_path = os.path.join(another_dir, one_vat_file)
        if not os.path.exists(another_vat_file_path):
            nonexist_files.append(one_vat_file)
    print('nonexist_num={}'.format(len(nonexist_files)))
    print('\n'.join(sorted(nonexist_files)))
    print('********** nonexist vat file **********')


vat_dir = '/home/adam/Pictures/201801/vat'
# find_vat_without_check_code(vat_dir, vat_dir + '/check_code')
find_check_code_without_vat(vat_dir, vat_dir + '/check_code')
# find_vat_without_vat_date(vat_dir, vat_dir + '/vat_date')
# find_vat_date_without_vat(vat_dir, vat_dir + '/vat_date')

one_dir = '/home/adam/Pictures/vat_other/2017/vat'
another_dir = '/home/adam/Pictures/vat'
# find_vat_in_one_dir_but_not_in_another_dir(one_dir, another_dir)
find_vat_in_one_dir_but_not_in_another_dir(one_dir, another_dir)
