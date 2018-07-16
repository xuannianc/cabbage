import cv2
from util import draw_lines

image = cv2.imread('201709251020208_1.jpg')
indexes = [30 * i for i in range(8)]
draw_lines(image, indexes, axis=0, width=1)
