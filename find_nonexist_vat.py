import imutils
from imutils import paths
import os
import glob

# for image_file_path in paths.list_images('/home/adam/Pictures/vat_test'):
#     image_file_dir, image_file = os.path.split(image_file_path)
#     if not os.path.exists(os.path.join('/home/adam/Pictures/vat', image_file)):
#         print(image_file_path)

for vat_file_path in glob.glob('/home/adam/Pictures/vat/*.jpg'):
    vat_file = os.path.split(vat_file_path)[1]
    vat_file_name = os.path.splitext(vat_file)[0]
    vat_date_file_name = vat_file_name + '_2'
    vat_date_file = vat_date_file_name + '.jpg'
    vat_date_file_path = os.path.join('/home/adam/Pictures/vat/vat_dates', vat_date_file)
    if not os.path.exists(vat_date_file_path):
        print(vat_file)