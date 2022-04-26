import cv2
import numpy as np

def order_points(pts):
	rect = np.zeros((4, 2), dtype = "float32")

	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]

	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]

	return rect.astype(int)

def find_corners(img):
    # CV2 cornerHarris tutorial followed from https://answers.opencv.org/question/186538/to-find-the-coordinates-of-corners-detected-by-harris-corner-detection/
    #img = cv2.imread(img)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray,2,3,0.04)
    dst = cv2.dilate(dst,None)
    ret, dst = cv2.threshold(dst,0.01*dst.max(),255,0)
    dst = np.uint8(dst)

    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)

    board_corners = order_points(corners)
    
    xs = board_corners[:,0]
    ys = board_corners[:,1]

    # for i in range(len(xs)):
    #     cv2.circle(img, (xs[i], ys[i]), 5, (0,0,255), -1)
    #cv2.imwrite('dsttest.png',img)

    return board_corners

# filename = 'test.png'
# corners = find_corners(filename)
# print(corners)