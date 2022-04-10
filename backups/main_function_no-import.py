import pytesseract 
from pytesseract import Output
import cv2 
import common
import numpy as np
import matplotlib.pyplot as plt
import math
import helper

#img = cv2.imread('coffee_sample.png')
pic = 'hello.jpeg'
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
x_top = 0
x_bot = 0
y_top = 0
y_bot = 0

full_height,full_width,_ = img.shape

def nothing(x):
    pass
helper = False
if (helper):

    cv2.namedWindow("trackbar", cv2.WINDOW_NORMAL)
    cv2.createTrackbar("Upper_X", "trackbar", x_top, full_width, nothing)
    cv2.createTrackbar("Lower_X", "trackbar", x_bot, full_width, nothing)
    cv2.createTrackbar("Upper_Y", "trackbar", y_top, full_height, nothing)
    cv2.createTrackbar("Lower_Y", "trackbar", y_bot, full_height, nothing)



    A_max = 2000
    B_max = 2000
    C_max = 5000

    A_scale = 10000
    B_scale = 10
    C_scale = 100000000

    A1 = int(A_max/2) 
    A2 = int(A_max/2)  
    A3 = int(A_max/2)
    A4 = int(A_max/2)
    B1 = int(B_max/2)
    B2 = int(B_max/2)
    C1 = int(C_max/2)
    C2 = int(C_max/2)


    cv2.createTrackbar("A1 (X-stretch)", "trackbar", A1, A_max, nothing)
    cv2.createTrackbar("A2 (Bottom-slant)", "trackbar", A2, A_max, nothing)
    cv2.createTrackbar("A3 (Left-slant)", "trackbar", A3, A_max, nothing)
    cv2.createTrackbar("A4 (Y-stretch)", "trackbar", A4, A_max, nothing)
    cv2.createTrackbar("B1 (X-shift)", "trackbar", B1, B_max, nothing)
    cv2.createTrackbar("B2 (Y-shift)", "trackbar", B2, B_max, nothing)
    cv2.createTrackbar("C1 (Left-lift)", "trackbar", C1, C_max, nothing)
    cv2.createTrackbar("C2 (Bottom-lift)", "trackbar", C2, C_max, nothing)



    while(True):
        img_cp = img.copy()
        

        A1 = int(cv2.getTrackbarPos("A1 (X-stretch)", "trackbar"))
        A2 = int(cv2.getTrackbarPos("A2 (Bottom-slant)", "trackbar"))
        A3 = int(cv2.getTrackbarPos("A3 (Left-slant)", "trackbar"))
        A4 = int(cv2.getTrackbarPos("A4 (Y-stretch)", "trackbar"))
        B1 = int(cv2.getTrackbarPos("B1 (X-shift)", "trackbar"))
        B2 = int(cv2.getTrackbarPos("B2 (Y-shift)", "trackbar"))
        C1 = int(cv2.getTrackbarPos("C1 (Left-lift)", "trackbar"))
        C2 = int(cv2.getTrackbarPos("C2 (Bottom-lift)", "trackbar"))

        
        #M = np.float32([[A1, A2, B1], [A3, A4, -B2], [C1, -C2, 1.00000]])
        #M = np.float32([[A1/100, A2/100, B1], [A3/100, A4/100, -B2], [C1/100000, -C2/100000, 1.00000]])
        
        M_iden = np.float32([[1, 0.0, 0.0], [0.0, 1, 0.0], [0.0, 0.0, 1.00000]])
        
        M_fixed = np.float32([[1.0+(A1-(A_max/2))/A_scale, (A2-(A_max/2))/A_scale, (B1-(B_max/2))/B_scale], 
        [(A3-(A_max/2))/A_scale, 1.0+(A4-(A_max/2))/A_scale, (B2-(B_max/2))/B_scale], 
        [(C1-(C_max/2))/C_scale, (C2-(C_max/2))/C_scale, 1.00000]])

        #print((B1-(B_max/2))/B_scale)
        print((B2-(B_max/2)/B_scale))

        img_cp = cv2.warpPerspective(img_cp, M_fixed, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)

        x_top = int(cv2.getTrackbarPos("Upper_X", "trackbar"))
        x_bot = int(cv2.getTrackbarPos("Lower_X", "trackbar"))
        y_top = int(cv2.getTrackbarPos("Upper_Y", "trackbar"))
        y_bot = int(cv2.getTrackbarPos("Lower_X", "trackbar"))
        cv2.line(img_cp, (x_top, 0), (x_top, full_height), (0,255,0), 1) #
        cv2.imshow("stream", img_cp)
        cv2.waitKey(10)


n_boxes = len(d['level'])
# print(n_boxes)

# file1 = open("text_out.txt","w")
# file1.writelines(d)
# file1.close()

# # Board is 15x15
# for i in range(n_boxes):
#     (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
#     #cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
#     cv2.putText(img, str(x), (x+20,y+40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100,100,255))
#     cv2.putText(img, str(y), (x+20,y+55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100,100,255))
#     cv2.putText(img, str(w), (x+20,y+70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100,100,255))
#     cv2.putText(img, str(h), (x+20,y+85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100,100,255))
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
    


common.save_img(img, 'output1.jpg')
