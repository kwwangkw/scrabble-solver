import pytesseract 
from pytesseract import Output
import cv2 
import common
import numpy as np
import matplotlib.pyplot as plt
import math
import helper
import config
import find_corners
import torch
from torch import nn
from torchvision import transforms
from PIL import Image
import pickle
import itertools

# Model definition
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(1)

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2)
        self.drop1 = nn.Dropout(0.1)
        self.norm2 = nn.BatchNorm2d(32)

        self.avg1 = nn.AvgPool2d(4, stride=2)
        self.drop2 = nn.Dropout(0.5)
        self.norm3 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.norm4 = nn.BatchNorm2d(32)
        self.drop3 = nn.Dropout(0.1)

        self.avg2 = nn.AvgPool2d(4, stride=2)
        self.norm5 = nn.BatchNorm2d(32)
        self.drop4 = nn.Dropout(0.5)

        self.avg3 = nn.AvgPool2d(4, stride=2)
        self.norm7 = nn.BatchNorm2d(32)
        
        
        self.fc1 = nn.Linear(128, 64)
        self.drop6 = nn.Dropout()
        self.fc2 = nn.Linear(64, 27)
  
    def forward(self, x):
        relu = nn.ReLU()

        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = relu(x)

        x = self.avg1(x)
        x = self.drop2(x)
        x = self.norm3(x)
        x = relu(x)

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.norm4(x)
        x = relu(x)

        x = self.avg2(x)
        x = self.norm5(x)
        x = self.drop4(x)
        x = relu(x)

        x = self.avg3(x)
        x = self.norm7(x)
        x = relu(x)


        x = x.view(-1,x.shape[1] * x.shape[2] * x.shape[3])
        x = self.fc1(x)
        x = self.drop6(x)
        x = relu(x)
        x = self.fc2(x)
        return x

#img = cv2.imread('coffee_sample.png')


# pic = 'hello.jpeg'
# img = cv2.imread(pic)

pic = 'hello.jpeg'
#pic = 'scrab2.jpg'
img = cv2.imread(pic)

#img = cv2.resize(img, (1200, 1200))
# imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

hImg, wImg, _ = img.shape

ave_value = (np.max(img) + np.min(img))/2

contrast = (img > ave_value) * 255

contrast = contrast.astype("uint8")

#d = pytesseract.image_to_data(img, lang='eng', config=custom_config, output_type=Output.DICT)
full_width = 0
full_height = 0
print(str(full_width) + ", ", str(full_height))
x_top = 0
x_bot = 0
y_top = 0
y_bot = 0

print_bool = False

img_string = 'image'
#pts =  np.zeros((4, 2), dtype = "float32")

model = Network() 
state_dict = torch.load("letter_model_state.sav", map_location=torch.device('cpu'))
model.load_state_dict(state_dict)

pts = find_corners.find_corners(img)
#breakpoint()
highest_score = 0.0
best_corners = np.zeros((4, 2))
incr = 0
for x in pts:
    #x = np.array(x)
    #breakpoint()
    img_wrect = img.copy()
    img_wrect = helper.draw_rect(img, x)


    #cv2.imshow(img_string, img_wrect)
    # helper.drag_corners(img, img_wrect, img_string, pts)
    # while (True):
    #     k = cv2.waitKey(10)
    #     if k == 32:
    #         break
    # print("Rect Done [X]")

    warped = helper.four_point_transform(img, x) # Their code

    warp_h, warp_w, _ = warped.shape


    #print("Loading letter recognition...")

    wi = math.ceil(warp_w/15)
    hi = math.ceil(warp_h/15)
    #print(str(wi) + ", ", str(hi))

    charar = np.chararray((15, 15))
    #print("warp dimensions: ", warped.shape)
    cp_warped = warped.copy()
    max_dim = max(wi, hi)
    cp_warped = cv2.resize(cp_warped, (max_dim*15, max_dim*15))


    # cv2.imshow("warp", cp_warped)
    # print("warp dimensions: ", cp_warped.shape)
    # print("(B)")
    # print("[Press [space] to continue]")

    # while (True):
    #     k = cv2.waitKey(10)
    #     if k == 32:
    #         cv2.destroyWindow("warp")
    #         break


    #charar = np.chararray((15, 15))
    charar = [[0 for a in range(15)]for b in range(15)]
    score_arr = [[0 for c in range(15)]for d in range(15)]
    #print("warp dimensions: ", warped.shape)
    cp_warped = warped.copy()
    cp_warped = cv2.resize(cp_warped, (wi*15, hi*15))
    #model = pickle.load(open('letter_model_state.sav', 'rb'))


    for i in range(15):
        #print("i: ", i)
        for j in range(15):

            
            # #cv2.rectangle(cp_warped, (wi*j, hi*i), (wi*(j+1), hi*(i+1)), (200, 50, 255), 1)
            # cv2.imshow('segmented', cp_warped)
            # cv2.waitKey(1)



            roi = cp_warped[hi*i:hi*(i+1), wi*j:wi*(j+1)]
           
            
            char, score_pred = helper.get_prediction(roi, model)
            #score_pred = helper.get_prediction(roi, model)[1][0]

            charar[i][j] = char
            score_arr[i][j] = round(score_pred[0].item(), 4)

        if print_bool:
            print(charar[i])


            #    common.save_img(roi, 'roi.jpg')
    for i in range(15):
        ax = 0
        if print_bool:
            print(score_arr[i])
    score = np.mean(np.asarray(score_arr))
    #print("mean: ", round(score, 5))
    if (score > highest_score):
        highest_score = score
        best_corners = x
    if (incr % 100 == 0):
        print(incr/100)
        print("score: ", round(score,4), " - [highscore: ", round(highest_score,4), "]")
    incr = incr + 1






### NEXT STEP AFTER BEST
img_wrect = img.copy()
img_wrect = helper.draw_rect(img, best_corners)


#cv2.imshow(img_string, img_wrect)
# helper.drag_corners(img, img_wrect, img_string, pts)
# while (True):
#     k = cv2.waitKey(10)
#     if k == 32:
#         break
# print("Rect Done [X]")

warped = helper.four_point_transform(img, best_corners) # Their code

warp_h, warp_w, _ = warped.shape



wi = math.ceil(warp_w/15)
hi = math.ceil(warp_h/15)

charar = np.chararray((15, 15))
print("warp dimensions: ", warped.shape)
cp_warped = warped.copy()
max_dim = max(wi, hi)
cp_warped = cv2.resize(cp_warped, (max_dim*15, max_dim*15))


# cv2.imshow("warp", cp_warped)
# print("warp dimensions: ", cp_warped.shape)
# print("(B)")
# print("[Press [space] to continue]")

# while (True):
#     k = cv2.waitKey(10)
#     if k == 32:
#         cv2.destroyWindow("warp")
#         break


#charar = np.chararray((15, 15))
charar = [[0 for x in range(15)]for y in range(15)]
score_arr = [[0 for x in range(15)]for y in range(15)]
#print("warp dimensions: ", warped.shape)
cp_warped = warped.copy()
cp_warped = cv2.resize(cp_warped, (wi*15, hi*15))
#model = pickle.load(open('letter_model_state.sav', 'rb'))


for i in range(15):
    #print("i: ", i)
    for j in range(15):

        
        cv2.rectangle(cp_warped, (wi*j, hi*i), (wi*(j+1), hi*(i+1)), (200, 50, 255), 1)
        #cv2.putText(cp_warped, str(j+1), (wi*j, hi*i+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 50, 50), 2)   
        cv2.imshow('segmented', cp_warped)
        cv2.waitKey(10)

        roi = cp_warped[hi*i:hi*(i+1), wi*j:wi*(j+1)]
        
        
        char, score_pred = helper.get_prediction(roi, model)
        #score_pred = helper.get_prediction(roi, model)[1][0]

        charar[i][j] = char
        score_arr[i][j] = round(score_pred[0].item(), 4)

    print(charar[i])


        #    common.save_img(roi, 'roi.jpg')
for i in range(15):
    x = 0
    
    print(score_arr[i])
score = np.mean(np.asarray(score_arr))
print(score)
#print(charar)

