"""
Task 5 Code
"""
from sys import flags
import numpy as np
from matplotlib import pyplot as plt
from homography import fit_homography, homography_transform
from common import save_img, read_img
# from testing import fit_homography
import os
import cv2


def make_synthetic_view(img, corners, size):
    '''
    Creates an image with a synthetic view of selected region in the image
    from the front. The region is bounded by a quadrilateral denoted by the
    corners array. The size array defines the size of the final image.

    Input - img: image file of shape (H,W,3)
            corner: array containing corners of the book cover in 
            the order [top-left, top-right, bottom-right, bottom-left]  (4,2)
            size: array containing size of book cover in inches [height, width] (1,2)

    Output - A fronto-parallel view of selected pixels (the book as if the cover is
            parallel to the image plane), using 100 pixels per inch.
    '''
    # print(size)
    h = size[0][0]
    w = size[0][1]

    new_x = np.array([[0], [100*w - 1], [100*w - 1], [0]])
    corners = np.hstack((corners, new_x))
    new_y = np.array([[0], [0], [100*h-1], [100*h-1]])
    corners = np.hstack((corners,new_y))
    H = fit_homography(corners)

    new_w = int(100*w - 1)
    new_h = int(100*h - 1)

    output = cv2.warpPerspective(img, H, (new_w, new_h), flags=cv2.INTER_LINEAR)

    return output
    
if __name__ == "__main__":
    # Task 5

    image = cv2.imread('eg1.jpeg')
    corners = np.load(os.path.join("task5",case_name,"corners.npy"))
    size = np.load(os.path.join("task5",case_name,"size.npy"))

    result = make_synthetic_view(I, corners, size)
    save_img(result, case_name+"_frontoparallel.jpg")


