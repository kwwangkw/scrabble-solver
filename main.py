import pytesseract 
from pytesseract import Output
import cv2 
import common
import numpy as np
import matplotlib.pyplot as plt
import math

#img = cv2.imread('coffee_sample.png')
img = cv2.imread('bro.png')
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

full_height,full_width,_ = img.shape

# n_boxes = len(d['level'])
# print(n_boxes)

# file1 = open("text_out.txt","w")
# file1.writelines(d)
# file1.close()



# # Board is 15x15
# for i in range(n_boxes):
#     (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
#     #cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
#     # cv2.putText(img, str(x), (x+20,y+40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100,100,255))
#     # cv2.putText(img, str(y), (x+20,y+55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100,100,255))
#     # cv2.putText(img, str(w), (x+20,y+70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100,100,255))
#     # cv2.putText(img, str(h), (x+20,y+85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100,100,255))
#     if (w > full_width):
#         full_width = w
#     if (h > full_height):
#         full_height = h


print(str(full_width) + ", ", str(full_height))


wi = math.ceil(full_width/15)
hi = math.ceil(full_height/15)
print(str(wi) + ", ", str(hi))

charar = np.chararray((15, 15))

for i in range(15):
    for j in range(15):
        cv2.rectangle(img, (wi*i, hi*j), (wi*(i+1), hi*(j+1)), (50, 50, 255), 2)
        roi = img[wi*i:wi*(i+1), hi*j:hi*(j+1)]
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
            char = db['text'][4][0]
        except:
            char = '_'

        charar[i][j] = char
        #print(char)
        # if (i == 0 and j == 5):
        #     #print(out_below)
        #     print(db['text'][4][0])
        #     # plt.imshow(roi)
        #     # plt.show()
        # if (i == 0 and j == 6):
        #     #print(out_below)
        #     print(db['text'][4][0])
        #     # plt.imshow(roi)
        #     # plt.show()


        #    common.save_img(roi, 'roi.jpg')

print(charar)
# plt.imshow(img)
# plt.show()
# for element in d:
    


common.save_img(img, 'output1.jpg')
