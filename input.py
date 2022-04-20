import re


def insert_wordrack(word_rack):
    print("In UPPERCASE, please enter letters on your word rack.")
    pattern = "[a-z]"
    i = 0
    while i != 7:
        letter = input("Please enter your letter: ")
        if re.findall(pattern, letter) or len(letter) > 1:
            print("wrong shit my guy, enter again")
        else:
            word_rack.append(letter)
            i += 1

    return word_rack

word_rack = []
bro = insert_wordrack(word_rack)
print(bro)