# Scrabble Solver

Scrabble score maximization from image input using corner keypoint warping and neural network letter recognition

### Authors

 - Wei Jie Lee (lweijie@umich.edu)
 - Jaehyun Shim (jaeday@umich.edu)
 - Kevin Wang (kwwangkw@umich.edu)
 - William Wang (willruiz@umich.edu)


## Files



### Training_Data

Contains data and pipeline used to train our neural network letter recognition. Having pictures of scrabble boards as our input, we have first used warp_lots.py to crop and homography transform the board into a perfect square and divide the image into 15x15 smaller squares for individual tiles. We then used crop.py to loop through the processed inputs and manually label each square as appropriate letter.

### Main

This is where our main driver of the program is. To test your own input, add your picture to sample_inputs, change line 308's pic variable to './sample_inputs/{your image name}' and simply run main.py. and follow instructions on the terminal.

Below is a simple workflow of our program:

```mermaid
graph LR
A[User input] -- Homography transform & corner detection --> B(15x15 squares )
B --> D((Letter detection & board processing))
C(crop.py and warp.py)-- neural network model --> D
D--Fix board to be 100% accurate-->F(Correct board & wordrack)
E(wordrack input) -->F
F --Gameplay algorithm--> G{Strategy & board output}
```

### Visuals
<img width="862" alt="image" src="https://user-images.githubusercontent.com/43936507/165176898-0c13dfaa-d2e2-445b-8f59-6dd527fd26de.png">

### Backups

Where some of our old works are.
