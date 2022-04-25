import pytesseract 
from pytesseract import Output
import cv2 
import common
import numpy as np
import matplotlib.pyplot as plt
import math
import helper as helper
import config
from transformation.find_corners import find_corners
import torch
from torch import nn
from torchvision import transforms
from PIL import Image
import pickle
from solver.dawg import *
from solver.board import ScrabbleBoard
import sys
import random
import re

# Model definition - used for detecting letters
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

def best_move(board, word_rack, root):
    tile_bag = ["A"] * 9 + ["B"] * 2 + ["C"] * 2 + ["D"] * 4 + ["E"] * 12 + ["F"] * 2 + ["G"] * 3 + \
               ["H"] * 2 + ["I"] * 9 + ["J"] * 1 + ["K"] * 1 + ["L"] * 4 + ["M"] * 2 + ["N"] * 6 + \
               ["O"] * 8 + ["P"] * 2 + ["Q"] * 1 + ["R"] * 6 + ["S"] * 4 + ["T"] * 6 + ["U"] * 4 + \
               ["V"] * 2 + ["W"] * 2 + ["X"] * 1 + ["Y"] * 2 + ["Z"] * 1 + ["%"] * 2

    
    game = ScrabbleBoard(root, board)

    wordDict = []
    locDict = []
    currWord = ""
    board_extended = [x + [""] for x in board]

    for row in range(15):
        currWord = ""
        for col in range(16):
            currChar = board_extended[row][col]
            if currChar != '':
                currWord = "".join([currWord, currChar])
            if currChar == '' and currWord != ""and len(currWord) > 1:
                wordDict.append(currWord)
                locDict.append((row, col-len(currWord)))
                currWord = ""
                currChar = ""
            elif currChar == "":
                currWord = ""
                currChar = ""
    #print(wordDict)
    #print(locDict)

    transposedWordDict = []
    transposedLocDict = []
    transposed_board = np.array(board)
    transposed_board = transposed_board.T
    transposed_board = transposed_board.tolist()
    transposed_board = [x + [""] for x in transposed_board]

    for row in range(15):
        currWord = ""
        for col in range(16):
            currChar = transposed_board[row][col]
            if currChar != '':
                currWord = "".join([currWord, currChar])
            if currChar == '' and currWord != ""and len(currWord) > 1:
                transposedWordDict.append(currWord)
                transposedLocDict.append((row, col-len(currWord)))
                currWord = ""
                currChar = ""
            elif currChar == "":
                currWord = ""
                currChar = ""
    
    #print(transposedWordDict)
    #print(transposedLocDict)

    empty = True
    for row in board:
        for char in row:
            if char.isalpha():
                empty = False

    if not empty: # fill the board
        for i in range(len(wordDict)):
            game.insert_word(locDict[i][0]+1, locDict[i][1]+1, wordDict[i])
            # game.print_board()
        
        for i in range(len(transposedWordDict)):
            game._transpose()
            game.insert_word(transposedLocDict[i][0]+1, transposedLocDict[i][1]+1, transposedWordDict[i])
            # game.print_board()
            game._transpose()

    game.print_board()

    if empty:
        game.get_start_move(word_rack)
    else:
        game.get_best_move(word_rack)
    
    if game.best_word == "":
        print("No words found.")
    else:
        print("Best play:", game.best_word, "for", game.word_score_dict[game.best_word], "points.")
    game.print_board()


def set_and_transform(img):
    pts = find_corners(img)
    img_wrect = img.copy()
    img_wrect = helper.draw_rect(img, pts)

    img_string = 'image'

    cv2.imshow(img_string, img_wrect)

    #drag corner for the user (GUI)
    helper.drag_corners(img, img_wrect, img_string, pts)
    while (True):
        k = cv2.waitKey(10)
        if k == 32:
            break

    print("Rect Done [X]")

    #pts = np.asarray(config.global_coord, dtype = "float32")
    warped = helper.four_point_transform(img, pts) # Their code

    warp_h, warp_w, _ = warped.shape

    #return warped height, warped weight and warped image itself.
    return warp_h, warp_w, warped



def divide_to_tiles(warp_h, warp_w, warped):
    #calculate each cell's dimension on 15x15 board
    wi = math.ceil(warp_w/15)
    hi = math.ceil(warp_h/15)
    print("warp dimensions: ", warped.shape)

    cp_warped = warped.copy()
    max_dim = max(wi, hi)
    cp_warped = cv2.resize(cp_warped, (max_dim*15, max_dim*15))
    cv2.imshow("warp", cp_warped)
    print("warp dimensions: ", cp_warped.shape)
    print("(B)")
    print("[Press [space] to continue]")

    while (True):
        k = cv2.waitKey(10)
        if k == 32:
            cv2.destroyWindow("warp")
            break

    #cp_warped = warped.copy()
    #cp_warped = cv2.resize(cp_warped, (wi*15, hi*15))

    return cp_warped, max_dim, max_dim


def detect_and_output(cp_warped, wi, hi, charar, score_arr):
    print("Loading letter recognition...")
    score_charonly = []
    char_count = 0
    for i in range(15):
        #print("i: ", i)
        for j in range(15):

            catch = False
            
            cv2.rectangle(cp_warped, (wi*j, hi*i), (wi*(j+1), hi*(i+1)), (200, 50, 255), 1)
            cv2.imshow('segmented', cp_warped)
            cv2.waitKey(10)
            roi = cp_warped[hi*i:hi*(i+1), wi*j:wi*(j+1)]
    
            char, score_pred = helper.get_prediction(roi, model)

            if char == " ":
                charar[i][j] = ""
            else:
                charar[i][j] = char
                score_charonly.append(round(score_pred[0].item(), 2))
                char_count = char_count + 1

            score_arr[i][j] = round(score_pred[0].item(), 2)

        #print(charar[i])
    #for x in range(15):
        #print(score_arr[x])
    #print("All tile mean score: ", np.mean(np.asarray(score_arr)))
    #print("Only char detected - mean score: ", np.mean(np.asarray(score_charonly)))
    return charar, score_arr

def fix_input(charar):
    cont = True
    pattern = "[a-z]"
    while cont:
        print("Here is your processed board. ")
        helper.print_board(charar)
        y_n = input("Does this look correct? (Y/N)")
        if y_n.upper() == "Y":
            print("We will now return the most optimal gameplay for you.")
            cont = False
        elif y_n.upper() == "N":
            x_coord = 0
            while not int(x_coord) in range(1,16):
                x_coord= input("Please enter the x_coordinate of the tile you want to fix (1-15): ")
            y_coord = 0
            while not int(y_coord) in range(1,16):
                y_coord= input("Please enter the y_coordinate of the tile you want to fix (1-15): ")
            letter = "aasdf"
            while re.findall(pattern, letter) or len(letter) > 1:
                letter = input("Please enter the right letter you want to fix it to: (A-Z)")

            charar[int(y_coord) - 1][int(x_coord) - 1] = letter

    return charar

def insert_wordrack(word_rack):
    num = input("Please enter the number of tiles on your hand: ")
    numpattern = "^[0-9]*$"
    while not re.findall(numpattern, num):
        num = input("Please enter a valid number for the number of tiles on your hand:")
    pattern = "[a-z]"
    num = int(num)
    i = 0
    while i != num:
        letter = input("Please enter a letter in your hand: ")
        if re.findall(pattern, letter.upper()) or len(letter) > 1:
            print("Please enter a letter (A-Z).")
        else:
            word_rack.append(letter.upper())
            i += 1


    return word_rack


if __name__ == '__main__':
    
    #Read input
    pic = './sample_inputs/presentaiton.jpg'
    img = cv2.imread(pic)

    #load up letter detection model
    model = Network() 
    state_dict = torch.load("letter_detection/letter_model_state.sav", map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)

    #load up the scrabble words data
    words_dict = open("lexicon/scrabble_words_complete.pickle", "rb")
    root = pickle.load(words_dict)
    words_dict.close()

    #declare container for detected board
    charar = [[0 for x in range(15)]for y in range(15)]
    score_arr = [[0 for x in range(15)]for y in range(15)]
        
    warp_h, warp_w, warped = set_and_transform(img)
    cp_warped, wi, hi = divide_to_tiles(warp_h, warp_w, warped)

    charar, score_arr = detect_and_output(cp_warped, wi, hi, charar, score_arr)
    
    charar1 = fix_input(charar)

    # for i in range(15):
    #     print(score_arr[i])
    # print("mean: ", round(np.mean(np.asarray(score_arr)), 5))

    word_rack = []
    user_words = insert_wordrack(word_rack)

    #print(charar1)

    best_move(charar1, word_rack, root)

