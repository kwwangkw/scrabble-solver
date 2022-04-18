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
    pathlib.Path("./crop_chars/"+str(file.stem)).mkdir(exist_ok=True)


    warped = cv2.imread(str(file))
    warped = cv2.resize(warped, (900, 900), cv2.INTER_AREA)
    warp_h, warp_w, _ = warped.shape

    wi = math.ceil(warp_w/15)
    hi = math.ceil(warp_h/15)
    #print(str(wi) + ", ", str(hi))

    charar = np.chararray((15, 15))
    print("warp dimensions: ", warped.shape)
    cp_warped = warped.copy()
    cp_warped = cv2.resize(cp_warped, (wi*15, hi*15))
    for i in range(15):
        #print("i: ", i)
        for j in range(15):
            seq = j + i * 15
            #print(j)
            # print(wi*(i+1), hi*(j+1))
            #catch = False
            
            cv2.rectangle(cp_warped, (wi*j, hi*i), (wi*(j+1), hi*(i+1)), (250, 20, 250), 3)
        
            roi = cp_warped[hi*i:hi*(i+1), wi*j:wi*(j+1)]
            cv2.imshow("cropped", cp_warped)
            manual_char = '_'
            while(True):
                keycrop = cv2.waitKey(10)
                #breakpoint()
                if (keycrop != -1):
                    pass
                if (keycrop >= 97 and keycrop <= 122):
                    manual_char = keycrop - 32
                    print(chr(keycrop))
                    break
                elif (keycrop == 32):
                    manual_char = 95
                    print('_')
                    break

            folder_name = str(file.stem)
            cv2.imwrite(f"./crop_chars/{folder_name}/" + chr(manual_char) + "-" + str(seq) + ".png", roi)
            #print(seq)
            cv2.rectangle(cp_warped, (wi*j, hi*i), (wi*(j+1), hi*(i+1)), (50, 155, 50), 3)
           