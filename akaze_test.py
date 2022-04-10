    import numpy as np
    import cv2
    import math

    pic = 'hello.jpeg'
    img = cv2.imread(pic)
    ench_image = cv2.imread(ench, 0)
    orig_image = cv2.imread(orig, 0)
    orig_image_rgb = cv2.imread(orig)
    
    try:
        surf = cv2.KAZE_create()
        kp1, des1 = surf.detectAndCompute(ench_image, None)
        kp2, des2 = surf.detectAndCompute(orig_image, None)
    except cv2.error as e:
        raise e