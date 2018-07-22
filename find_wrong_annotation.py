import os
import imutils
from imutils import paths
import glob

BASE_DIR = '/home/adam/Pictures/vat'


def find_missing(reverse=False):
    # vat --> vat_number
    if not reverse:
        for vat_file_path in glob.glob(BASE_DIR + '/*.jpg'):
            vat_file = os.path.split(vat_file_path)[1]
            vat_file_name, vat_file_ext = os.path.splitext(vat_file)
            vat_number_file_path = os.path.join(BASE_DIR, 'vat_numbers', vat_file_name + '_1' + vat_file_ext)
            if not os.path.exists(vat_number_file_path):
                print(vat_file)
    # vat_number --> vat
    else:
        for vat_number_file_path in glob.glob(BASE_DIR + '/vat_numbers/*.jpg'):
            vat_number_file = os.path.split(vat_number_file_path)[1]
            vat_number_file_name, vat_number_file_ext = os.path.splitext(vat_number_file)
            vat_file_path = os.path.join(BASE_DIR, vat_number_file_name[:-2] + vat_number_file_ext)
            if not os.path.exists(vat_file_path):
                print(vat_number_file)


find_missing(reverse=True)
