from dawg import *
from board import ScrabbleBoard
import sys
import random
import pickle

if __name__ == "__main__":
    tile_bag = ["A"] * 9 + ["B"] * 2 + ["C"] * 2 + ["D"] * 4 + ["E"] * 12 + ["F"] * 2 + ["G"] * 3 + \
               ["H"] * 2 + ["I"] * 9 + ["J"] * 1 + ["K"] * 1 + ["L"] * 4 + ["M"] * 2 + ["N"] * 6 + \
               ["O"] * 8 + ["P"] * 2 + ["Q"] * 1 + ["R"] * 6 + ["S"] * 4 + ["T"] * 6 + ["U"] * 4 + \
               ["V"] * 2 + ["W"] * 2 + ["X"] * 1 + ["Y"] * 2 + ["Z"] * 1 + ["%"] * 2

    to_load = open("lexicon/scrabble_words_complete.pickle", "rb")
    root = pickle.load(to_load)
    to_load.close()
    word_rack = random.sample(tile_bag, 7)
    game = ScrabbleBoard(root)
    word_rack = ["A"] + ["K"] + ["C"] + ["T"] + ["Z"] + ["Y"] + ["X"]

    game.get_start_move(word_rack)
    # word_rack = game.get_best_move(word_rack)
    if game.best_word == "":
        # draw new hand if can't find any words
        print("No words found.")
    else:
        print(game.best_word)
    print(game.word_score_dict)

