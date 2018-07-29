from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
import time
import matplotlib.pyplot as plt
import cv2
import numpy as np
labels_to_names = {0: 'code', 1: 'number', 2: 'date', 3: 'check_code', 4: 'buyer', 5: 'seller'}
# image = read_image_bgr('/home/adam/Pictures/vat/1100162350_12093275_20180427.jpg')
# image = read_image_bgr('/home/adam/Pictures/vat/1100162350_12093276_20180504.jpg')
image = read_image_bgr('/home/adam/Pictures/1100181130_16283226_20180402.jpg')
# copy to draw on
draw = image.copy()
draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
# preprocess image for network
image = preprocess_image(image)
image, scale = resize_image(image)
input = np.expand_dims(image, axis=0)
model = models.load_model('/home/adam/.keras/models/retinanet/vat.h5', backbone_name='resnet50')
boxes, scores, labels = model.predict_on_batch(input)
print(labels)
print(scores)

# process image
start = time.time()
boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
print("processing time: ", time.time() - start)

# correct for image scale
boxes /= scale

# visualize detections
for box, score, label in zip(boxes[0], scores[0], labels[0]):
    # scores are sorted so we can break
    if score < 0.5:
        break
    print('label={},score={}'.format(label,score))
    color = label_color(label)

    b = box.astype(int)
    draw_box(draw, b, color=color)

    caption = "{} {:.3f}".format(labels_to_names[label], score)
    draw_caption(draw, b, caption)

plt.figure(figsize=(15, 15))
plt.axis('off')
plt.imshow(draw)
plt.show()