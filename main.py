import pytesseract 
from pytesseract import Output
import cv2 
import common
import numpy as np
import matplotlib.pyplot as plt
import math
import helper
import config
import find_corners
import torch
from torch import nn
from torchvision import transforms
from PIL import Image
import pickle

#img = cv2.imread('coffee_sample.png')


# pic = 'hello.jpeg'
# img = cv2.imread(pic)

pic = 'hello.jpeg'
#pic = 'scrab2.jpg'
img = cv2.imread(pic)

#img = cv2.resize(img, (1200, 1200))
# imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

hImg, wImg, _ = img.shape

ave_value = (np.max(img) + np.min(img))/2

contrast = (img > ave_value) * 255

contrast = contrast.astype("uint8")

#d = pytesseract.image_to_data(img, lang='eng', config=custom_config, output_type=Output.DICT)
full_width = 0
full_height = 0
print(str(full_width) + ", ", str(full_height))
x_top = 0
x_bot = 0
y_top = 0
y_bot = 0

img_string = 'image'
pts =  np.zeros((4, 2), dtype = "float32")

pts = find_corners.find_corners(img)
img_wrect = img.copy()
img_wrect = helper.draw_rect(img, pts)


cv2.imshow(img_string, img_wrect)

helper.drag_corners(img, img_wrect, img_string, pts)
while (True):
    k = cv2.waitKey(10)
    if k == 32:
        break

print("Rect Done [X]")

#pts = np.asarray(config.global_coord, dtype = "float32")
warped = helper.four_point_transform(img, pts) # Their code

warp_h, warp_w, _ = warped.shape


print("Loading letter recognition...")

wi = math.ceil(warp_w/15)
hi = math.ceil(warp_h/15)
print(str(wi) + ", ", str(hi))

charar = np.chararray((15, 15))
print("warp dimensions: ", warped.shape)
cp_warped = warped.copy()
max_dim = max(wi, hi)
cp_warped = cv2.resize(cp_warped, (max_dim*15, max_dim*15))
cv2.imshow("warp", cp_warped)
print("warp dimensions: ", cp_warped.shape)
print("(B)")
print("[Press [space] to continue]")

while (True):
    k = cv2.waitKey(10)
    if k == 32:
        cv2.destroyWindow("warp")
        break


charar = np.chararray((15, 15))
#print("warp dimensions: ", warped.shape)
cp_warped = warped.copy()
cp_warped = cv2.resize(cp_warped, (wi*15, hi*15))
#model = pickle.load(open('letter_model_state.sav', 'rb'))

for i in range(15):
    #print("i: ", i)
    for j in range(15):

        catch = False
        
        cv2.rectangle(cp_warped, (wi*j, hi*i), (wi*(j+1), hi*(i+1)), (200, 50, 255), 1)
        #cv2.putText(cp_warped, str(j+1), (wi*j, hi*i+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 50, 50), 2)   
        cv2.imshow('segmented', cp_warped)
        cv2.waitKey(10)
        #common.save_img(cp_warped, 'output2.jpg')
        roi = cp_warped[hi*i:hi*(i+1), wi*j:wi*(j+1)]


        roi = cv2.resize(roi, (wi*2, hi*2))

        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        gray, img_bin = cv2.threshold(gray,128,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        gray = cv2.bitwise_not(cv2.bitwise_not(img_bin))
        kernel = np.ones((2, 1), np.uint8)
        roi = cv2.erode(gray, kernel, iterations=1)
        roi = cv2.dilate(roi, kernel, iterations=1)
        
        # state_dict = torch.load("letter_model_state.sav", map_location=torch.device('cpu'))
        # char = helper.get_prediction(roi, model)

        letter_config = '--psm 10 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVXWYZ'
        
        db = pytesseract.image_to_data(roi, lang='eng', config=letter_config, output_type=Output.DICT)
        

        char = '_'
        try:
            char = db['text'][4]
        except:
            char = '_'

        charar[i][j] = char
        #print(char)
        # if (i == 0 and j ==  8):
        #     print(db['text'])
        #     plt.imshow(roi)
        #     plt.show()
        # if (i == 0 and j ==  2):
        #     print(db['text'][4])
        #     plt.imshow(roi)
        #     plt.show()
        # if (i == 0 and j == 3):
        #     print(db['text'][4])
        #     plt.imshow(roi)
        #     plt.show()


        #    common.save_img(roi, 'roi.jpg')

print(charar)

