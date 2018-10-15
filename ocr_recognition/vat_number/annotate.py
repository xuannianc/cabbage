# import the necessary packages
from imutils import paths
import argparse
import imutils
import cv2
import os
import logging
import json
import glob
import sys
import numpy as np

# logger.basicConfig(level=logger.DEBUG)
logger = logging.getLogger('annotate')
logger.setLevel(logging.DEBUG)  # default log level
format = logging.Formatter("%(asctime)s %(name)-8s %(levelname)-8s %(lineno)-4d %(message)s")  # output format
sh = logging.StreamHandler(stream=sys.stdout)  # output to standard output
sh.setFormatter(format)
logger.addHandler(sh)

image_paths = glob.glob('/home/adam/Pictures/vat_other/2017/vat/vat_number/*.jpg')[451:]
num_images = len(image_paths)
# image_paths = paths.list_images('/home/adam/Pictures/vat/vat_numbers/')
# image_paths = list(paths.list_images('./invoice_numbers'))
# image_paths = ['./invoice_numbers/201709261041055_1.jpg']
# image_path = './invoice_numbers/2017092610412394_1.jpg'
# print(image_paths.index(image_path))
#
counts = json.load(open('counts.json'))
logger.info('init_counts={}'.format(counts))
# loop over the image paths
for (i, image_path) in enumerate(image_paths):
    # display an update to the user
    logger.info("Processing image {}/{} starts".format(i + 1, num_images))
    try:
        # load the image and convert it to grayscale, then pad the
        # image to ensure digits caught on the border of the image are retained
        image = cv2.imread(image_path)
        image_width, image_height = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.namedWindow('gray', cv2.WINDOW_NORMAL)
        cv2.imshow('gray', gray)
        cv2.waitKey(0)
        # threshold the image to reveal the digits
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

        # find contours in the image, keeping only the four largest
        # ones
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:8]
        cnts = sorted(cnts, key=lambda c: cv2.boundingRect(c)[0])
        # cv2.drawContours(gray, cnts, -1, (0, 255, 0), 3)
        # cv2.imshow('image_with_cnts', gray)
        # cv2.waitKey(0)
        # loop over the contours
        for j, c in enumerate(cnts):
            # compute the bounding box for the contour then extract
            # the digit
            (x, y, w, h) = cv2.boundingRect(c)
            if x < 5:
                roi = gray[:, : x + w + 5]
            else:
                roi = gray[:, x - 5:x + w + 5]
            # display the character, making it larger enough for us to see, then wait for a keypress
            roi_height, roi_width = roi.shape[:2]
            # cv2.namedWindow('raw_roi', cv2.WINDOW_NORMAL)
            # cv2.imshow("raw_roi", roi)
            # cv2.waitKey(0)
            if roi_width > roi_height:
                logger.error('width bigger than height:image_path={},idx={}-{}'.format(image_path, i, j))
                cv2.imwrite("/home/adam/workspace/github/okra/datasets/errors/error-{}-{}.jpg".format(i, j), roi)
                continue
            # 按 height=28 等比例缩放
            roi = imutils.resize(roi, height=28)
            roi_width = roi.shape[1]
            roi_width_pad = (28 - roi_width) // 2
            roi = np.pad(roi, [(0, 0), (roi_width_pad, 28 - roi_width - roi_width_pad)], mode='constant',
                         constant_values=255)
            # roi_margin = cv2.copyMakeBorder(roi_h28, 0, 0, roi_h28_width_margin, roi_h28_width_margin,
            #                                 cv2.BORDER_REPLICATE)
            cv2.namedWindow('roi', cv2.WINDOW_NORMAL)
            cv2.imshow("roi", roi)
            # 返回的是字符的 ascii
            key = cv2.waitKey(0)
            key = chr(key)
            # if the ’‘’ key is pressed, then ignore the character
            if key not in '0123456789':
                logger.info("ignoring character")
                continue
            # grab the key that was pressed and construct the path
            # the output directory
            digit_dir_path = os.path.sep.join(['datasets/digits', key])
            # if the output directory does not exist, create it
            if not os.path.exists(digit_dir_path):
                os.makedirs(digit_dir_path)
            # write the labeled character to file
            count = counts.get(key)
            count += 1
            p = os.path.sep.join([digit_dir_path, "{}.jpg".format(str(count).zfill(6))])
            cv2.imwrite(p, roi)
            # increment the count for the current key
            counts[key] = count
        logger.info("Processing {}/{} image ends with counts: {}".format(i + 1, num_images, counts))
        json.dump(counts, open('counts.json', 'w'))
    # we are trying to control-c out of the script, so break from the
    # loop (you still need to press a key for the active window to trigger this)
    except KeyboardInterrupt:
        logger.info("Manually leaving script when handling {}/{} image: {}".format(i + 1, num_images, image_path))
        break
    # an unknown error has occurred for this particular image
    except Exception as e:
        logger.exception('Handling failed on {}/{} image: {}'.format(i + 1, num_images, image_path))
logger.info('terminal_counts={}'.format(counts))
json.dump(counts, open('counts.json', 'w'))
