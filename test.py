from dawg import *
from board import ScrabbleBoard
import sys
import random
import pickle
import numpy as np

def best_move(board, word_rack, to_load):
    tile_bag = ["A"] * 9 + ["B"] * 2 + ["C"] * 2 + ["D"] * 4 + ["E"] * 12 + ["F"] * 2 + ["G"] * 3 + \
               ["H"] * 2 + ["I"] * 9 + ["J"] * 1 + ["K"] * 1 + ["L"] * 4 + ["M"] * 2 + ["N"] * 6 + \
               ["O"] * 8 + ["P"] * 2 + ["Q"] * 1 + ["R"] * 6 + ["S"] * 4 + ["T"] * 6 + ["U"] * 4 + \
               ["V"] * 2 + ["W"] * 2 + ["X"] * 1 + ["Y"] * 2 + ["Z"] * 1 + ["%"] * 2

    root = pickle.load(to_load)
    to_load.close()
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
    print(wordDict)
    print(locDict)

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
    
    print(transposedWordDict)
    print(transposedLocDict)

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


board = ''
word_rack = ["I"] + ["Z"] + ["Z"] + ["Z"] + ["Z"] + ["Z"] + ["X"]
# board = [["T","A","M","E","","M","I","X","","","","","","","",],
#          ["A","T","","L","","","","","","","","","","","",],
#          ["B","","","E","","","","","","","","","","","",],
#          ["","","","M","","","","","","","","","","","",],
#          ["","","","E","","","","","","","","","","","",],
#          ["","","","N","","","","","","","","","","","",],
#          ["","","","T","","","","","","","","","","","",],
#          ["","","","","","","","","","","","","","","",],
#          ["","","","E","","","","","","","","","","","",],
#          ["","","","C","","","","","","","","","","","",],
#          ["","","","H","","","","","","","","","","","",],
#          ["","","","O","","","","","","","","","","","",],
#          ["","","","","","","","","","","","","","","",],
#          ["","","","","","","","","","","","","","","",],
#          ["","","","","","","","","","","","","","","",],
#          ]
board = [["T","A","M","E","","M","I","X","","","","","","","",],
         ["A","T","","L","","","","","","","","","","","",],
         ["B","","","E","","","","","","","","","","","",],
         ["","","","M","","","","","","","","","","","",],
         ["","","","E","","","","","","","","","","","",],
         ["","","","N","","","","","","","","","","","",],
         ["","","","T","","","","","","","","","","","",],
         ["","","","","","","","","","","","","","","",],
         ["","","","E","","","","","","","","","","","",],
         ["","","","C","","","","","","","","","","","",],
         ["","","","H","","","","","","","","","","","",],
         ["","","","O","M","E","L","E","T","","","","","","",],
         ["","","","","","","","","","","","","","","",],
         ["","","","","","","","","","","","","","","",],
         ["","","","","","","","","","","","","","","",],
         ]
