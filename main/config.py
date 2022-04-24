import cv2 

pic = "hello.jpeg"
img = cv2.imread(pic)

global_coord = [[0,0], [0,0], [0,0], [0,0]]
click_incr = 0
abs_incr = 0

done = False