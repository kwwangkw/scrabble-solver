import pytesseract 
from pytesseract import Output
import cv2 
import common
import numpy as np
import matplotlib.pyplot as plt
import math
import helper
import config

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

common.save_img(contrast, 'output0.jpg')

contrast = contrast.astype("uint8")

#plt.savefig('output2.png')
custom_config = '--psm 1 --oem 3 -c tessedit_char_blacklist=0123456789'
#d = pytesseract.image_to_boxes(contrast, lang='eng', config=custom_config)

# print(contrast.dtype)
# print(img.dtype)

#dnp = np.asarray(d)

d = pytesseract.image_to_data(img, lang='eng', config=custom_config, output_type=Output.DICT)
full_width = 0
full_height = 0
print(str(full_width) + ", ", str(full_height))
x_top = 0
x_bot = 0
y_top = 0
y_bot = 0

img_string = 'image'
cv2.imshow(img_string, img)

full_height,full_width,_ = img.shape
#helper.manual_warp(img, full_width, full_height)               ### TRACKBAR
helper.click_corners(img, img_string, full_width, full_height)  ### CORNERS
while (True):
    k = cv2.waitKey(10)
    if k == 32:
        break

print("Rect Done [X]")

n_boxes = len(d['level'])

pts = np.asarray(config.global_coord, dtype = "float32")
warped = helper.four_point_transform(img, pts) # Their code

warp_h, warp_w, _ = warped.shape

cv2.imshow("warp", warped)
print("warp dimensions: ", warped.shape)
print("(B)")
print("[Press [space] to continue]")

while (True):
    k = cv2.waitKey(10)
    if k == 32:
        break

print("Loading letter recognition...")

wi = math.ceil(warp_w/15)
hi = math.ceil(warp_h/15)
print(str(wi) + ", ", str(hi))

charar = np.chararray((15, 15))
print("warp dimensions: ", warped.shape)
cp_warped = warped.copy()
cp_warped = cv2.resize(cp_warped, (wi*15, hi*15))
for i in range(15):
    print("i: ", i)
    for j in range(15):
        print(j)
        print(wi*(i+1), hi*(j+1))
        catch = False
        
        cv2.rectangle(cp_warped, (wi*j, hi*i), (wi*(j+1), hi*(i+1)), (200, 50, 255), 1)
        #cv2.putText(cp_warped, str(j+1), (wi*j, hi*i+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 50, 50), 2)   
        cv2.imshow("final", cp_warped)
        cv2.waitKey(10)
        common.save_img(cp_warped, 'output2.jpg')
        roi = cp_warped[hi*i:hi*(i+1), wi*j:wi*(j+1)]
        # except:
        #     cv2.rectangle(cp_warped, (hi*j, wi*i ), (warp_h-hi*j, warp_w-wi*i), (200, 50, 255), 1)
        #     cv2.imshow("final", cp_warped)
        #     cv2.waitKey(10)
        #     common.save_img(cp_warped, 'output2.jpg')
        #     roi = cp_warped[wi*i:warp_w-wi*i, hi*j:warp_h-hi*j]
            
        #     catch = True

        # if (j == 13):
        #     cv2.imshow("roi", roi)
        #     cv2.waitKey(0)
        #     print("CATCH: ", catch)
        # if (j == 14):
        #     cv2.imshow("roi", roi)
        #     cv2.waitKey(0)

        roi = cv2.resize(roi, (wi*2, hi*2))

        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        gray, img_bin = cv2.threshold(gray,128,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        gray = cv2.bitwise_not(cv2.bitwise_not(img_bin))
        kernel = np.ones((2, 1), np.uint8)
        roi = cv2.erode(gray, kernel, iterations=1)
        roi = cv2.dilate(roi, kernel, iterations=1)

        letter_config = '--psm 10 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVXWYZ'
        #db = pytesseract.image_to_data(roi, config= letter_config, output_type=Output.DICT)
        db = pytesseract.image_to_data(roi, lang='eng', config=letter_config, output_type=Output.DICT)
        

        #db = pytesseract.image_to_data(roi, lang='eng', config=custom_config, output_type=Output.DICT)
        #print(d)
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
# plt.imshow(img)
# plt.show()
# for element in d:
    


common.save_img(cp_warped, 'output1.jpg')
