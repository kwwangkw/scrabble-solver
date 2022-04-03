import pytesseract
import cv2 
import common
import numpy as np

img = cv2.imread('bro.png')
# imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

hImg, wImg, _ = img.shape

ave_value = (np.max(img) + np.min(img))/2

contrast = (img > ave_value) * 255

common.save_img(contrast, 'output0.jpg')

contrast = contrast.astype("uint8")

boxes = pytesseract.image_to_boxes(contrast)

# print(contrast.dtype)
# print(img.dtype)

# print(boxes)
for b in boxes.splitlines():
    b = b.split(' ')
    # print(b)
    x, y, w, h = int(b[1]), int(b[2]), int(b[3]), int(b[4])
    cv2.rectangle(contrast, (x, hImg - y), (w, hImg - h), (50, 50, 255), 1)
    cv2.putText(contrast, b[0], (x, hImg - y + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (50, 205, 50), 1)

common.save_img(contrast, 'output1.jpg')
