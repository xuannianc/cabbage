import cv2
import pytesseract
from imutils import paths
import os
import json

image_paths = list(paths.list_images('./invoice_numbers'))
results = {}
for image_path in image_paths:
    image_name = os.path.split(image_path)[1]
    image = cv2.imread(image_path)
    image = cv2.copyMakeBorder(image, 8, 8, 8, 8,
                               cv2.BORDER_REPLICATE)
    content = pytesseract.image_to_string(image, config='--psm 7 --oem 0 -c tessedit_char_whitelist=0123456789')
    print(content)
    cv2.imshow('image', image)
    cv2.waitKey(0)
    results[image_name] = content
print('results={}'.format(results))
json.dump(results, open('scan_in_results.json', 'w'))
