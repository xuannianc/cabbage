# import the necessary packages
from imutils import paths
import argparse
import imutils
import cv2
import os
import logging

logging.basicConfig(level=logging.DEBUG)
# construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--input", required=True,
#                 help="path to input directory of images")
# ap.add_argument("-a", "--annot", required=True,
#                 help="path to output directory of annotations")
# args = vars(ap.parse_args())
# grab the image paths then initialize the dictionary of character
# counts
# imagePaths = list(paths.list_images(args["input"]))
image_paths = list(paths.list_images('./invoice_numbers'))
# image_paths = ['./invoice_numbers/201709261041055_1.jpg']
# image_path = './invoice_numbers/2017092610412394_1.jpg'
# print(image_paths.index(image_path))
#

counts = {}
# loop over the image paths
for (i, image_path) in enumerate(image_paths):
    # display an update to the user
    logging.info("processing image {}/{}".format(i + 1,
                                                 len(image_paths)))
    try:
        # load the image and convert it to grayscale, then pad the
        # image to ensure digits caught on the border of the image
        # are retained
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('gray1', gray)
        gray = cv2.copyMakeBorder(gray, 0, 0, 8, 8,
                                  cv2.BORDER_REPLICATE)
        cv2.imshow('gray2', gray)
        cv2.waitKey(0)
        # threshold the image to reveal the digits
        thresh = cv2.threshold(gray, 0, 255,
                               cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

        # find contours in the image, keeping only the four largest
        # ones
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:8]
        # cv2.drawContours(gray, cnts, -1, (0, 255, 0), 3)
        # cv2.imshow('image_with_cnts', gray)
        # cv2.waitKey(0)
        # loop over the contours
        for j, c in enumerate(cnts):
            # compute the bounding box for the contour then extract
            # the digit
            (x, y, w, h) = cv2.boundingRect(c)
            roi = gray[y:y + h, x - 5:x + w + 5]
            # display the character, making it larger enough for us
            # to see, then wait for a keypress
            roi_height, roi_width = roi.shape[:2]
            # cv2.imshow("raw_ROI", roi)
            # key = cv2.waitKey(0)
            if roi_width > roi_height:
                logging.error('{}-{}-{}'.format(i, j, image_path))
                cv2.imwrite("./datasets/errors/error-{}-{}-{}".format(i, j, image_path), roi)
                continue
            roi_h28 = imutils.resize(roi, height=28)
            roi_h28_width = roi_h28.shape[1]
            roi_h28_width_margin = (28 - roi_h28_width) // 2
            roi_margin = cv2.copyMakeBorder(roi_h28, 0, 0, roi_h28_width_margin, roi_h28_width_margin,
                                            cv2.BORDER_REPLICATE)
            roi = cv2.resize(roi_margin, (28, 28))
            cv2.imshow("ROI", roi)
            key = cv2.waitKey(0)
            key = chr(key)
            # if the ’‘’ key is pressed, then ignore the character
            if key not in '0123456789':
                print("[INFO] ignoring character")
                continue
            # grab the key that was pressed and construct the path
            # the output directory
            dirPath = os.path.sep.join(['datasets/digits', key])
            # if the output directory does not exist, create it
            if not os.path.exists(dirPath):
                os.makedirs(dirPath)
            # write the labeled character to file
            count = counts.get(key, 1)
            p = os.path.sep.join([dirPath, "{}.jpg".format(
                str(count).zfill(6))])
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
        logging.exception('handling image {}-{} fail'.format(i, image_path))
