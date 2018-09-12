# Deep-learning-for-pong
An implementation of a 4 layer Neural Network and two Reinforcement learning algorithms to play the game of pong with a UI built using pygame.

### Files:
```qlearning.py```: Implementation of the Q-Learning algorithm  
```sarsa.py```: Implementation of the SARSA learning algorithm   
```nnet.py```: A 4 layer neural network architecture with an implementation of minibatch gradient descent

### Dataset
```Data/expert_policy.txt```: Each row has a 5 dimensional feature vector (x position of the ball, y position of the ball, x acceleration, y acceleration, y position of paddle) followed by a label - 0, 1 or 2 (whether the paddle moved down, stayed in the same place or moved up)

### Results
This folder contains 3 graphs of the number of bounces per game over 200 test games, along with a pygame video of the trained agent playing the game.  

* The 4 layer neural net averaged ~10 bounces per game.
* The agent trained using Q-Learning averaged ~12 bounces per game.
* The agent trained using SARSA-Learning averaged ~13 bounces per game.
