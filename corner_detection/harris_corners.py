import os
from tkinter import E
from turtle import width

import numpy as np
import scipy.ndimage
# Use scipy.ndimage.convolve() for convolution.
# Use zero padding (Set mode = 'constant'). Refer docs for further info.

from common import read_img, save_img

def order_points(pts):
	rect = np.zeros((4, 2), dtype = "float32")

	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]

	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]

	return rect 

def corner_score(image, u=5, v=5, window_size=(5, 5)):
    """
    Given an input image, x_offset, y_offset, and window_size,
    return the function E(u,v) for window size W
    corner detector score for that pixel.
    Use zero-padding to handle window values outside of the image.

    Input- image: H x W
           u: a scalar for x offset
           v: a scalar for y offset
           window_size: a tuple for window size

    Output- results: a image of size H x W
    """
    temp = v
    v = u
    u = temp
    output = np.copy(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            output[i][j] = 0
            for x in range(i-5, i+5):
                for y in range(j-5, j+5):
                    curr_val = 0
                    next_val = 0
                    if x < image.shape[0] and x >= 0 and y < image.shape[1] and y >= 0:
                        curr_val = image[x][y]
                    if x+u < image.shape[0] and x+u >= 0 and y+v < image.shape[1] and y+v >= 0:
                        next_val = image[x+u][y+v]
                    output[i][j] = output[i][j] + (next_val - curr_val)**2
    return output


def harris_detector(image, window_size=(5, 5)):
    """
    Given an input image, calculate the Harris Detector score for all pixels
    You can use same-padding for intensity (or 0-padding for derivatives)
    to handle window values outside of the image.

    Input- image: H x W
    Output- results: a image of size H x W
    """
    # compute the derivatives
    Ix = np.gradient(image)[0]
    Iy = np.gradient(image)[1]

    Ixx = Ix**2
    Iyy = Iy**2
    Ixy = Ix*Iy

    # For each image location, construct the structure tensor and calculate
    # the Harris response
    response = np.copy(image)
    offset_height = int(window_size[0]/2)
    offset_width = int(window_size[1]/2)
    height = image.shape[0]
    width = image.shape[1]
    for y in range(offset_height, height - offset_height):
        for x in range(offset_width, width - offset_width):
            Sxx = np.sum(Ixx[y-offset_height:y+1+offset_height, x-offset_width:x+1+offset_width])
            Syy = np.sum(Iyy[y-offset_height:y+1+offset_height, x-offset_width:x+1+offset_width])
            Sxy = np.sum(Ixy[y-offset_height:y+1+offset_height, x-offset_width:x+1+offset_width])
            k = 0.05
            det = (Sxx * Syy) - (Sxy**2)
            trace = Sxx + Syy
            r = det - k * (trace**2)
            response[y][x] = r
    return response


def main():
    img = read_img('./test3.png')


    # -- TODO Task 5: Corner Score --
    
    harris_corners = harris_detector(img)
    save_img(harris_corners, "./test_out.png")
    

if __name__ == "__main__":
    main()
