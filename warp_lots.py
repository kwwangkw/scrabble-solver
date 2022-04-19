import pytesseract 
from pytesseract import Output
import cv2 
import common
import numpy as np
import matplotlib.pyplot as plt
import math
import helper
import config
import pathlib
import find_corners

#img = cv2.imread('coffee_sample.png')


# pic = 'hello.jpeg'
# img = cv2.imread(pic)
files = [f for f in pathlib.Path("will_customs").iterdir()]
iter = 1
for file in files:
    #config.img = cv2.imread(str(file))
    img = cv2.imread(str(file))
    img = cv2.resize(img, (900, 900), cv2.INTER_AREA)
    config.img = img

    full_height,full_width,_ = img.shape
    
    #img = cv2.resize(img, (1200, 1200))
    # imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    hImg, wImg, _ = img.shape

    ave_value = (np.max(img) + np.min(img))/2

    contrast = (img > ave_value) * 255

    contrast = contrast.astype("uint8")

    #plt.savefig('output2.png')
    custom_config = '--psm 1 --oem 3 -c tessedit_char_blacklist=0123456789'
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

    #helper.click_corners(img, img_string, full_width, full_height)  ### CORNERS
    
    while (True):
        k = cv2.waitKey(10)
        if k == 32:
            break

    print("Rect Done [X]")

    
    #pts = np.asarray(config.global_coord, dtype = "float32")
    warped = helper.four_point_transform(img, pts) # Their code

    warp_h, warp_w, _ = warped.shape

    # cv2.imshow("warp", warped)
    # print("warp dimensions: ", warped.shape)
    # print("(B)")
    # print("[Press [space] to continue]")

    # while (True):
    #     k = cv2.waitKey(10)
    #     if k == 32:
    #         break

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

    cv2.imwrite("./error_warp/w" + str(iter) + ".jpg", cp_warped)
    config.click_incr = 0
    config.abs_incr = 0
    config.done = False
    iter = iter + 1
    #common.save_img(cp_warped, 'warped.jpg')
