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

files = [f for f in pathlib.Path("warp_dump").iterdir()]
iter = 1
for file in files:
    pathlib.Path("./crop_chars/"+str(file.stem)).mkdir()

    # print("Rect Done [X]")

    # n_boxes = len(d['level'])

    # pts = np.asarray(config.global_coord, dtype = "float32")
    # warped = helper.four_point_transform(img, pts) # Their code
    warped = cv2.imread(str(file))
    warp_h, warp_w, _ = warped.shape

    # cv2.imshow("warp", warped)
    # print("warp dimensions: ", warped.shape)
    # print("(B)")
    # print("[Press [space] to continue]")

    # while (True):
    #     k = cv2.waitKey(10)
    #     if k == 32:
    #         break

    # print("Loading letter recognition...")

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
            seq = j + i * 15
            print(j)
            print(wi*(i+1), hi*(j+1))
            #catch = False
            
            cv2.rectangle(cp_warped, (wi*j, hi*i), (wi*(j+1), hi*(i+1)), (50, 250, 250), 3)
            #cv2.putText(cp_warped, str(j+1), (wi*j, hi*i+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 50, 50), 2)   
            #cv2.imshow("final", cp_warped)
            #cv2.waitKey(10)
            common.save_img(cp_warped, 'output2.jpg')
            roi = cp_warped[hi*i:hi*(i+1), wi*j:wi*(j+1)]
            cv2.imshow("cropped", cp_warped)
            manual_char = '_'
            while(True):
                keycrop = cv2.waitKey(10)
                #breakpoint()
                if (keycrop != -1):
                    print(keycrop)
                if (keycrop >= 97 and keycrop <= 122):
                    manual_char = keycrop - 32
                    break
                elif (keycrop == 32):
                    manual_char = 95
                    break

            folder_name = str(file.stem)
            cv2.imwrite(f"./crop_chars/{folder_name}/" + chr(manual_char) + "-" + str(seq) + ".png", roi)
            print(seq)
            cv2.rectangle(cp_warped, (wi*j, hi*i), (wi*(j+1), hi*(i+1)), (50, 255, 50), 3)
           