# import the necessary packages
from imutils import paths
import argparse
import imutils
import cv2
import os
import logging
import json
import glob
from keras.models import load_model
import numpy as np

logging.basicConfig(level=logging.DEBUG)

image_paths = glob.glob('/home/adam/Pictures/aa-ocr-test/*.png')
# image_paths = ['/home/adam/Pictures/vat/vat_numbers/3700164320_26498391_20170915_1.jpg',
#                '/home/adam/Pictures/vat/vat_numbers/1200174320_05570276_20180324_1.jpg',
#                '/home/adam/Pictures/vat/vat_numbers/1200174320_01942058_20180228_1.jpg',
#                '/home/adam/Pictures/vat/vat_numbers/3700164320_01277576_20170626_1.jpg',
#                '/home/adam/Pictures/vat/vat_numbers/4403172320_22308889_20180228_1.jpg',
#                '/home/adam/Pictures/vat/vat_numbers/5000172320_19102331_20171114_1.jpg',
#                '/home/adam/Pictures/vat/vat_numbers/3600164130_02509918_20171214_1.jpg',
#                '/home/adam/Pictures/vat/vat_numbers/5100163320_13217064_20171113_1.jpg',
#                '/home/adam/Pictures/vat/vat_numbers/1200164350_02309743_20180210_1.jpg']
model = load_model('models/best_weights_noaug_0.9896_0.0959.hdf5')
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
        image_width, image_height = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imshow('source image', image)
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
        cnts = sorted(cnts, key=lambda c: cv2.boundingRect(c)[0])
        # cv2.drawContours(image, cnts, -1, (0, 255, 0), 3)
        # cv2.imshow('image_with_cnts', image)
        # cv2.waitKey(0)
        # loop over the contours
        preds = []
        for j, c in enumerate(cnts):
            # compute the bounding box for the contour then extract
            # the digit
            (x, y, w, h) = cv2.boundingRect(c)
            print(x, y, w, h)
            if x < 2:
                roi = gray[y - 2: y + h + 2, : x + w + 5]
            else:
                roi = gray[y - 2: y + h + 2, x - 2:x + w + 2]
            # display the character, making it larger enough for us
            # to see, then wait for a keypress
            roi_height, roi_width = roi.shape[:2]
            print('raw_roi_height={},raw_roi_width={}'.format(roi_height, roi_width))
            cv2.imshow("raw_ROI", roi)
            key = cv2.waitKey(0)
            if roi_width > roi_height:
                logging.error('width bigger than height:image_path={},idx={}-{}'.format(image_path, i, j))
                cv2.imwrite("./datasets/errors/error-{}-{}.jpg".format(i, j), roi)
                continue
            roi_h28 = imutils.resize(roi, height=28)
            roi_h28_width = roi_h28.shape[1]
            roi_h28_width_margin = (28 - roi_h28_width) // 2
            roi_margin = cv2.copyMakeBorder(roi_h28, 0, 0, roi_h28_width_margin, roi_h28_width_margin,
                                            cv2.BORDER_REPLICATE)
            roi = cv2.resize(roi_margin, (28, 28))
            roi = roi.reshape(28, 28, 1)
            # cv2.imshow("roi", roi)
            # key = cv2.waitKey(0)
            cv2.imshow("roi", roi)
            key = cv2.waitKey(0)
            roi = np.expand_dims(roi, axis=0)
            pred = model.predict(roi).argmax(axis=1).tolist()[0]
            preds.append(str(pred))
        result = ''.join(preds)
        image_file_name = os.path.split(image_path)[1]
        vat_number = image_file_name.split('_')[1]
        if vat_number == result:
            success_count = counts.get('success', 0)
            counts['success'] = success_count + 1
        else:
            fail_count = counts.get('fail', 0)
            counts['fail'] = fail_count + 1
            fail_image_paths = counts.get('fail_image_paths', [])
            fail_image_paths.append(image_path)
            counts['fail_image_paths'] = fail_image_paths
        cv2.putText(image, result, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 2)
        cv2.imshow('image_with_results', image)
        cv2.waitKey(0)
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
# json.dump(counts, open('counts.json', 'w'))
