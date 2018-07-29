from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
import time
import matplotlib.pyplot as plt
import cv2
import numpy as np
import imutils
from imutils import paths

labels_to_names = {0: 'code', 1: 'number', 2: 'date', 3: 'check_code', 4: 'buyer', 5: 'seller'}
model = models.load_model('/home/adam/.keras/models/retinanet/vat_0729232039.h5', backbone_name='resnet50')
# image = cv2.imread('/home/adam/Pictures/vat/1100162350_12093275_20180427.jpg')
for image_file_path in paths.list_images('/home/adam/Pictures/vat_test'):
    image = read_image_bgr(image_file_path)
    # image = read_image_bgr('/home/adam/Pictures/1100181130_16283226_20180402.jpg')
    # copy to draw on
    draw = image.copy()
    # preprocess image for network
    image = preprocess_image(image)
    image, scale = resize_image(image)
    image = np.expand_dims(image, axis=0)
    boxes, scores, labels = model.predict_on_batch(image)
    # correct for image scale
    boxes /= scale
    # visualize detections
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        # scores are sorted so we can break
        if score < 0.5:
            break
        print(label)
        start_x = int(box[0])
        start_y = int(box[1])
        end_x = int(box[2])
        end_y = int(box[3])
        color = label_color(label)
        # draw the prediction on the output image
        name = labels_to_names[label]
        text = "{}: {:.2f}".format(name, score)
        cv2.rectangle(draw, (start_x, start_y), (end_x, end_y), color, 4)
        y = start_y - 10 if start_y - 10 > 10 else start_y + 10
        cv2.putText(draw, text, (start_x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1)
    # show the output image
    cv2.namedWindow('Output', cv2.WINDOW_NORMAL)
    cv2.imshow("Output", draw)
    cv2.waitKey(0)
