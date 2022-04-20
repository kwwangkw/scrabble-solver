import common
import numpy as np
import matplotlib.pyplot as plt
import math
import cv2 
import config
import torch
from torch import nn
from torchvision import transforms
from PIL import Image
import pickle

btn_down = False
coord_num = 0

def process_img(img):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Normalize(0, 1)
    ])
    img = cv2.resize(img,(32,32))
    #img.thumbnail((32, 32))
    img_arr = np.asarray(img)
    if len(img_arr.shape) < 3:
        img_arr = np.expand_dims(img_arr, 2)
        img_arr = np.repeat(img_arr, 3, 2)
    if img_arr.shape[2] > 3:
        img_arr = img_arr[:,:,:3]

    tpad = (32 - img_arr.shape[0]) // 2
    bpad = 32 - tpad - img_arr.shape[0]
    lpad = (32 - img_arr.shape[1]) // 2
    rpad = 32 - img_arr.shape[1] - lpad

    img_arr = np.pad(img_arr, ((tpad, bpad), (lpad, rpad), (0, 0)))
    img_arr = img_arr.transpose(2, 0, 1).astype(np.double) / 256
    img_tensor = transform(torch.Tensor(img_arr))
    return img_tensor



# Takes in PIL image and gives letter prediction. Returns None if prediction is a blank tile.
def get_prediction(img, model):
    img_arr = process_img(img)
    img_arr = img_arr.reshape(1, *img_arr.shape) # Reshape to include batch dimension
    model.eval()
    scores = torch.softmax(model(img_arr), 1)
    pred = scores.argmax(1)
    score = scores[0, pred]
    # pred = torch.softmax(model(img_arr), 1).argmax(1)[0]
    if pred == 26:
        return " ", score
    else:
        return chr(65 + pred), score



def nothing(x):
    pass

def draw_rect(img, pts):
    img_cp = img.copy()
    color = (0,255,0)
    # xs = pts[:,0]
    # ys = pts[:,1]
    
    for i in range(4):
        if (i == 0):
            cv2.line(img_cp, (pts[0][0], pts[0][1]), (pts[3][0], pts[3][1]), color, 2)
        else:
            cv2.line(img_cp, (pts[i][0], pts[i][1]), (pts[i-1][0], pts[i-1][1]), color, 2)
    
    for i in range(4):
        cv2.circle(img_cp, (pts[i][0], pts[i][1]), 6, (250,0,255), -1)

    return img_cp
    
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

	rect = order_points(pts)
	(tl, tr, br, bl) = rect

	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))

	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))

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

def mouse_handler(event, x, y, flags, data, img_in, img_wrect_in, pts_in):
    global btn_down
    global coord_num

    if event == cv2.EVENT_LBUTTONUP and btn_down:
        #if you release the button, finish the line
        btn_down = False

        pts_in[coord_num][0] = x
        pts_in[coord_num][1] = y
        img_cp = img_in.copy()
        img_cp = draw_rect(img_in, pts_in)
        cv2.imshow('image', img_cp)

    elif event == cv2.EVENT_MOUSEMOVE and btn_down:
        #thi is just for a ine visualization
        pts_in[coord_num][0] = x
        pts_in[coord_num][1] = y
        img_cp = img_in.copy()
        img_cp = draw_rect(img_in, pts_in)
        cv2.imshow('image', img_cp)

    elif event == cv2.EVENT_LBUTTONDOWN:
        btn_down = True
        min_dist = 100000
        for i in range(4):
            point1 = np.asarray([x,y])
            point2 = np.asarray([pts_in[i][0], pts_in[i][1]])
            dist = np.linalg.norm(point1 - point2)
            if (dist < min_dist):
                min_dist = dist
                coord_num = i

        pts_in[coord_num][0] = x
        pts_in[coord_num][1] = y
        img_cp = img_in.copy()
        img_cp = draw_rect(img_in, pts_in)
        cv2.imshow('image', img_cp)

        # data['lines'].insert(0,[(x, y)]) #prepend the point
        # cv2.circle(data['im'], (x, y), 3, (0, 0, 255), 5, 16)
        # cv2.imshow("Image", data['im'])

def drag_corners(img_in, img_wrect_in, img_string, pts_in):
    cv2.setMouseCallback(img_string, lambda *x: mouse_handler(*x, img_in, img_wrect_in, pts_in))
    cv2.waitKey(0)