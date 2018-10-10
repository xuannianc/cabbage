# import the necessary packages
from imutils import paths
import argparse
import imutils
import cv2
import os
import logging
import json
import glob

logging.basicConfig(level=logging.DEBUG)

image_paths = glob.glob('/home/adam/Pictures/vat_other/2017/vat/vat_number/*.jpg')
num_image = len(image_paths)
# image_paths = paths.list_images('/home/adam/Pictures/vat/vat_numbers/')
# image_paths = list(paths.list_images('./invoice_numbers'))
# image_paths = ['./invoice_numbers/201709261041055_1.jpg']
# image_path = './invoice_numbers/2017092610412394_1.jpg'
# print(image_paths.index(image_path))
#
counts = json.load(open('counts.json'))
# loop over the image paths
for (i, image_path) in enumerate(image_paths):
    # display an update to the user
    logging.info("processing image {}/{}".format(i + 1, num_image))
    try:
        # load the image and convert it to grayscale, then pad the
        # image to ensure digits caught on the border of the image
        # are retained
        image = cv2.imread(image_path)
        image_width, image_height = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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
            cv2.namedWindow('raw_roi', cv2.WINDOW_NORMAL)
            cv2.imshow("raw_roi", roi)
            key = cv2.waitKey(0)
            if roi_width > roi_height:
                logging.error('width bigger than height:image_path={},idx={}-{}'.format(image_path, i, j))
                cv2.imwrite("/home/adam/workspace/github/okra/datasets/errors/error-{}-{}.jpg".format(i, j), roi)
                continue
            roi_h28 = imutils.resize(roi, height=28)
            roi_h28_width = roi_h28.shape[1]
            roi_h28_width_margin = (28 - roi_h28_width) // 2
            roi_margin = cv2.copyMakeBorder(roi_h28, 0, 0, roi_h28_width_margin, roi_h28_width_margin,
                                            cv2.BORDER_REPLICATE)
            roi = cv2.resize(roi_margin, (28, 28))
            cv2.namedWindow('roi', cv2.WINDOW_NORMAL)
            cv2.imshow("roi", roi)
            key = cv2.waitKey(0)
            key = chr(key)
            # if the ’‘’ key is pressed, then ignore the character
            if key not in '0123456789':
                print("[INFO] ignoring character")
                continue
            # grab the key that was pressed and construct the path
            # the output directory
            digit_dir_path = os.path.sep.join(['datasets/digits', key])
            # if the output directory does not exist, create it
            if not os.path.exists(digit_dir_path):
                os.makedirs(digit_dir_path)
            # write the labeled character to file
            count = counts.get(key)
            p = os.path.sep.join([digit_dir_path, "{}.jpg".format(str(count).zfill(6))])
            cv2.imwrite(p, roi)
            # increment the count for the current key
            counts[key] = count + 1
    # we are trying to control-c out of the script, so break from the
    # loop (you still need to press a key for the active window to
    # trigger this)
    except KeyboardInterrupt:
        logging.info("manually leaving script")
        break
    # an unknown error has occurred for this particular image
    except Exception as e:
        logging.exception('handling {}/{} image fail:{}'.format(i, len(image_path), image_path))
print('counts={}'.format(counts))
json.dump(counts, open('counts.json', 'w'))
