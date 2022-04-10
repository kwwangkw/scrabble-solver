import common
import numpy as np
import matplotlib.pyplot as plt
import math
import cv2 
import config


def nothing(x):
    pass

def click_event(event, x, y, flags, params):
 
    # checking for left mouse clicks
    #print(x, ' ', y)
    print("click_incr:" , config.click_incr)
    if (event == cv2.EVENT_LBUTTONDOWN):
    #if (event == False):
        # displaying the coordinates
        # on the Shell
        config.global_coord [config.click_incr] [0] = x
        config.global_coord [config.click_incr] [1] = y


    xa = config.global_coord[config.click_incr-1][0]
    ya = config.global_coord[config.click_incr-1][1]
    xb = config.global_coord[config.click_incr][0]
    yb = config.global_coord[config.click_incr][1]
    color = (0,255,0)

    print(xb, ' ', yb)

    config.abs_incr = config.abs_incr+1

    if (config.abs_incr % 2 == 0):
            
        if (config.click_incr < 3):
            
            if (config.click_incr == 0):
                pass
            else:
                print("draw")
                cv2.line(config.img, (xa, ya), (xb, yb), color, 2)        
            
            text_coord = str(xb) + ',' + str(yb)
            #cv2.putText(config.img, text_coord, (xb+20,yb+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 50, 50), 2)    
            cv2.imshow("image", config.img)
            config.click_incr = config.click_incr + 1

        elif (config.click_incr == 3 and not(config.done)):
            text_coord = str(xb) + ',' + str(yb)
            cv2.putText(config.img, text_coord, (xb+20,yb+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 50, 50), 2)   
            cv2.line(config.img, (xa, ya), (xb, yb), color, 3)    
            cv2.line(config.img, (config.global_coord[0][0], config.global_coord[0][1]), 
            (config.global_coord[3][0], config.global_coord[3][1]), color, 2) 
            cv2.imshow("image", config.img)
            config.done = True
            print("(A)")
            print("[Press [space] to continue]")
        else:
            pass

        
        print(config.done)

def click_corners(img_in, img_string, full_width_in, full_height_in):
    cv2.setMouseCallback(img_string, click_event)
    #cv2.waitKey(0)

    

def manual_warp(img_in, full_width_in, full_height_in):    
    x_top = 0
    x_bot = 0
    y_top = 0
    y_bot = 0
    cv2.namedWindow("trackbar", cv2.WINDOW_NORMAL)
    cv2.createTrackbar("Upper_X", "trackbar", x_top, full_width_in, nothing)
    cv2.createTrackbar("Lower_X", "trackbar", x_bot, full_width_in, nothing)
    cv2.createTrackbar("Upper_Y", "trackbar", y_top, full_height_in, nothing)
    cv2.createTrackbar("Lower_Y", "trackbar", y_bot, full_height_in, nothing)



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
        img_in_cp = img_in.copy()
        

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

        img_in_cp = cv2.warpPerspective(img_in_cp, M_fixed, (img_in.shape[1], img_in.shape[0]), flags=cv2.INTER_LINEAR)

        x_top = int(cv2.getTrackbarPos("Upper_X", "trackbar"))
        x_bot = int(cv2.getTrackbarPos("Lower_X", "trackbar"))
        y_top = int(cv2.getTrackbarPos("Upper_Y", "trackbar"))
        y_bot = int(cv2.getTrackbarPos("Lower_X", "trackbar"))
        cv2.line(img_in_cp, (x_top, 0), (x_top, full_height_in), (0,255,0), 1) #
        cv2.imshow("stream", img_in_cp)
        cv2.waitKey(10)


def order_points(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	# return the ordered coordinates
	return rect


def four_point_transform(image, pts):
	# obtain a consistent order of the points and unpack them
	# individually
	rect = order_points(pts)
	(tl, tr, br, bl) = rect
	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
	# return the warped image
	return warped
