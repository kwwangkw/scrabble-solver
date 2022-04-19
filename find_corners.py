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


'''
import numpy as np
import cv2
from matplotlib import pyplot as plt
import os


MIN_MATCH_COUNT = 5


template_image = cv2.imread('template2.png')
template_image_gray = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)


# Initiate SIFT detector
#sift = cv2.SIFT()
sift = cv2.xfeatures2d.SIFT_create()


# find the keypoints and descriptors with SIFT in template image
kp_template, des_template = sift.detectAndCompute(template_image_gray, None)


FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)

flann = cv2.FlannBasedMatcher(index_params, search_params)


img = cv2.imread("test.png")  #  use second parameter 0 for auto gray conversion?


#  convert image to gray
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#  find the keypoints and descriptors with SIFT in query image
kp_img, des_img = sift.detectAndCompute(img, None)

#  get image dimension info
img_height, img_width = img_gray.shape
print("Image height:{}, image width:{}".format(img_height, img_width))


matches = flann.knnMatch(des_template,des_img,k=2)


# store all the good matches as per Lowe's ratio test.
good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)


if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp_template[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp_img[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()

    h,w = template_image_gray.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)

    img_board = img.copy()
    cv2.polylines(img_board,[np.int32(dst)],True,255,10, cv2.LINE_AA)
    """
    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

    img3 = cv2.drawMatches(template_image,kp_template,img,kp_img,good,None,**draw_params)
    """
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()






    #  get axis aligned bounding box for chessboard in input image
    x,y,w,h = cv2.boundingRect(dst)
    img_crop = img.copy()
    cv2.rectangle(img_crop,(x,y),(x+w,y+h),(0,0,255),5)


    #  draw OBB and AABB
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.axis("off")
    ax2.axis("off")
    ax1.set_title('OBB')
    ax2.set_title('AABB')
    ax1.imshow(cv2.cvtColor(img_board, cv2.COLOR_BGR2RGB))
    ax2.imshow(cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB))
    plt.show()


    #  crop board
    cropped_img = img[y:y+h, x:x+w].copy()
    plt.imshow(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
    plt.show()

    #  convert cropped area to gray
    cropped_img_gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
    plt.imshow(cropped_img_gray, cmap="gray")
    plt.show()

else:
    print("Not enough match")

'''

# import cv2
# import numpy as np

# def perspective_transform(image, corners):
#     def order_corner_points(corners):
#         # Separate corners into individual points
#         # Index 0 - top-right
#         #       1 - top-left
#         #       2 - bottom-left
#         #       3 - bottom-right
#         corners = [(corner[0][0], corner[0][1]) for corner in corners]
#         top_r, top_l, bottom_l, bottom_r = corners[0], corners[1], corners[2], corners[3]
#         return (top_l, top_r, bottom_r, bottom_l)

#     # Order points in clockwise order
#     print("Here")
#     ordered_corners = order_corner_points(corners)
#     print(ordered_corners)
#     top_l, top_r, bottom_r, bottom_l = ordered_corners

#     # Determine width of new image which is the max distance between 
#     # (bottom right and bottom left) or (top right and top left) x-coordinates
#     width_A = np.sqrt(((bottom_r[0] - bottom_l[0]) ** 2) + ((bottom_r[1] - bottom_l[1]) ** 2))
#     width_B = np.sqrt(((top_r[0] - top_l[0]) ** 2) + ((top_r[1] - top_l[1]) ** 2))
#     width = max(int(width_A), int(width_B))

#     # Determine height of new image which is the max distance between 
#     # (top right and bottom right) or (top left and bottom left) y-coordinates
#     height_A = np.sqrt(((top_r[0] - bottom_r[0]) ** 2) + ((top_r[1] - bottom_r[1]) ** 2))
#     height_B = np.sqrt(((top_l[0] - bottom_l[0]) ** 2) + ((top_l[1] - bottom_l[1]) ** 2))
#     height = max(int(height_A), int(height_B))

#     # Construct new points to obtain top-down view of image in 
#     # top_r, top_l, bottom_l, bottom_r order
#     dimensions = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], 
#                     [0, height - 1]], dtype = "float32")

#     # Convert to Numpy format
#     ordered_corners = np.array(ordered_corners, dtype="float32")

#     # Find perspective transform matrix
#     matrix = cv2.getPerspectiveTransform(ordered_corners, dimensions)

#     # Return the transformed image
#     return cv2.warpPerspective(image, matrix, (width, height))

# image = cv2.imread('test2.png')
# original = image.copy()
# blur = cv2.bilateralFilter(image,9,75,75)
# gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
# thresh = cv2.threshold(gray,40,255, cv2.THRESH_BINARY_INV)[1]

# cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cnts = cnts[0] if len(cnts) == 2 else cnts[1]
# # print(cnts)

# mask = np.zeros(image.shape, dtype=np.uint8)
# for c in cnts:
#     area = cv2.contourArea(c)
#     peri = cv2.arcLength(c, True)
#     approx = cv2.approxPolyDP(c, 0.015 * peri, True)

#     if area > 150000 and len(approx) == 4:
#         cv2.drawContours(image,[c], 0, (36,255,12), 3)
#         cv2.drawContours(mask,[c], 0, (255,255,255), -1)
#         transformed = perspective_transform(original, approx)

# mask = cv2.bitwise_and(mask, original)

# cv2.imshow('thresh', thresh)
# cv2.imshow('image', image)
# cv2.imshow('mask', mask)
# # cv2.imshow('transformed', transformed)
# cv2.waitKey()