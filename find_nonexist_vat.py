import imutils
from imutils import paths
import os

for image_file_path in paths.list_images('/home/adam/Pictures/vat_test'):
    image_file_dir, image_file = os.path.split(image_file_path)
    if not os.path.exists(os.path.join('/home/adam/Pictures/vat', image_file)):
        print(image_file_path)
        