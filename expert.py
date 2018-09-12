#coding: utf-8

import numpy as np
import nnet
import random, pygame
from ball import Ball
from scale import Scale
from paddle import Paddle
from game import Game
import pprint
import sys

#%%
def test_at_epoch(tst, e, record):
    e = e
    W = record[e, 1]
    B = record[e, 2]
    return nnet.FourNetwork(tst[:, :-1], W, B, None, True)

#%% Load Data
f = open("expert_policy.txt", mode='r')
aa = f.readlines()
f.close()
for i in range(len(aa)):
    strings = aa[i].split(sep=" ")
    aa[i] = []
    for s in strings:
        aa[i].append(float(s))
aa = np.array(aa)

#%% Normalize Data
normalize = []
for col in range(aa.shape[1]-1):
    mean = aa[:, col].mean()
    sd = aa[:, col].std()
    aa[:, col] = (aa[:, col] - mean)/sd
    normalize.append((mean, sd))

#%% 
epochs = 1000

np.random.shuffle(aa)

trn = aa[:, :]
tst = aa[:, :]
tst_y = tst[:, -1]

record = nnet.MinibatchGDFour(trn, epochs, 128, 0.2)

#%% Accuracy
tst_results = test_at_epoch(tst, epochs-1, record)
eq = tst_results == tst_y
print("Accuracy: {}".format(eq.sum()/eq.size))

#%% Confusion Matrix

confusion = np.zeros((3, 3))
tst_y = tst_y.astype(int)
for i in range(tst_results.shape[0]):
    confusion[tst_y[i], tst_results[i]] = confusion[tst_y[i], tst_results[i]]+1

confusion = confusion/confusion.sum(axis=1).reshape((3, 1))

print(confusion)

#%% Accuracy/Loss by epoch
import matplotlib.pyplot as plt

interval = 10
r = list(range(0, epochs, interval))

accbyepoch = np.zeros((len(r), ))
lossbyepoch = np.zeros((len(r), ))

for i in range(len(r)):
    tst_results = test_at_epoch(tst, i*interval, record)
    eq = tst_results == tst_y
    accbyepoch[i] = eq.sum()/eq.size
    
    lossbyepoch[i] = record[i*interval, 0]

print("Loss by epoch")
print(lossbyepoch)

print("Accuracy")
print(accbyepoch)

plt.title("Accuracy vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.plot(r, accbyepoch)
plt.show()

plt.title("Loss vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.plot(r, lossbyepoch)
plt.show()

#%% Apply to Game

pp = pprint.PrettyPrinter(indent=4)

WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)

SCREEN_WIDTH = 500
SCREEN_HEIGHT = 500
BALL_RADIUS = 8
PADDLE_THICKNESS = 10
PADDLE_SIZE = 100

def draw(ball, paddle, scale, window):

	#Draw window
	window.fill(WHITE)

	#Draw ball
	ball_position = ball.get_position()
	pygame.draw.circle(window, RED, scale.scale_ball(ball_position), BALL_RADIUS, 0)

	#Draw paddle
	paddle_position = paddle.get_position()
	paddle_object = pygame.Rect(scale.scale_paddle(1), scale.scale_paddle(paddle_position), PADDLE_THICKNESS, PADDLE_SIZE)
	pygame.draw.rect(window, BLACK, paddle_object)

	#Draw Wall
	wall_object = pygame.Rect(0, 0, PADDLE_THICKNESS, SCREEN_HEIGHT)
	pygame.draw.rect(window, BLACK, wall_object)

#%%
    
e = 299

W = record[e, 1]
B = record[e, 2]

pong = Game()
pong.init_nagent(W, B, normalize)

pygame.init()
fps = pygame.time.Clock()

window = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), 0, 32)
pygame.display.set_caption('Pong Game')

scale = Scale()
pong.ball = Ball()
pong.paddle = Paddle()
pong.state = (pong.ball.x, pong.ball.y, pong.ball.velocity_x, pong.ball.velocity_y, pong.paddle.y)
pong.agent.epsilon = 0
pong.is_active = True
pong.score = 0
pong.game_number = 0

while True:
	# draw(pong.ball, pong.paddle, scale, window)
	pong.update_test_nagent()
	if pong.finished_testing:
		break

	# pygame.display.update()
	# fps.tick(30)

x_values = [i[0] for i in pong.test_stats]
y_values = [i[1] for i in pong.test_stats]
print(pong.test_stats)
print(float(sum(y_values))/len(y_values))

plt.plot(x_values, y_values)
plt.xlabel('Game number')
plt.ylabel('Reward')
plt.title('Rewards/game for 200 test games')
plt.show()