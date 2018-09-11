from imutils import paths
import imutils
import cv2
from ocr_recognition.check_code.config import *

# widths = []
# for image_path in paths.list_images(TRAIN_DIR):
#     image = cv2.imread(image_path, 0)
#     image = imutils.resize(image, height=32)
#     height, width = image.shape[:2]
#     # padding_width = (248 - width) // 2
#     # image = cv2.copyMakeBorder(image, 0, 0, padding_width, padding_width,
#     #                            cv2.BORDER_REPLICATE)
#     # image = cv2.resize(image, (248, 32))
#     # cv2.imshow('image', image)
#     # cv2.waitKey(0)
#     if width < 249:
#         widths.append(width)
# print(sorted(widths))
# print(len(widths))
# print('min={}'.format(min(widths)))
# # 248
# print('max={}'.format(max(widths)))


def get_max_width():
    widths = []
    for image_path in paths.list_images(TRAIN_DIR):
        image = cv2.imread(image_path, 0)
        image = imutils.resize(image, height=32)
        height, width = image.shape[:2]
        widths.append(width)
    print('sorted={}'.format(sorted(widths)))
    print('len={}'.format(len(widths)))
    # 282
    print('min={}'.format(min(widths)))
    # 499
    print('max={}'.format(max(widths)))

get_max_width()

