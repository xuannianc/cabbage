import pytesseract
from imutils import paths
import os.path as osp
import cv2

check_code_dir = '/home/adam/Pictures/201801/vat/check_code'
with open('check_code_labels.txt', 'w') as f:
    check_code_files = []
    for check_code_path in paths.list_images(check_code_dir):
        check_code_file = osp.split(check_code_path)[1]
        check_code_files.append(check_code_file)
    for check_code_file in sorted(check_code_files):
        image = cv2.imread(osp.join(check_code_dir, check_code_file))
        text = pytesseract.image_to_string(image, lang='chi_sim')
        print('text={}'.format(text))
        print('check_code_file={}'.format(check_code_file))
        check_code = ''.join(text.split(' '))[3:]
        label = '{},10,11,12,'.format(check_code_file) + ','.join(check_code) + '\n'
        print('check_code={}'.format(check_code))
        print('label={}'.format(label))
        f.write(label)
        # cv2.imshow('image', image)
        # cv2.waitKey(0)
