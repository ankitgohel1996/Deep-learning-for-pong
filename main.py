import random, pygame
from ball import Ball
from scale import Scale
from paddle import Paddle
from game import Game
import pprint
import sys
import matplotlib.pyplot as plt

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

#If you want to see the 200 test games being played, uncomment the pygame stuff in both the SARSA and Q learning functions

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

	paddle_position2 = paddle.get_position()
	paddle_object = pygame.Rect(scale.scale_paddle(1), scale.scale_paddle(paddle_position), PADDLE_THICKNESS, PADDLE_SIZE)
	pygame.draw.rect(window, BLACK, paddle_object)

	#Draw Wall
	wall_object = pygame.Rect(0, 0, PADDLE_THICKNESS, SCREEN_HEIGHT)
	pygame.draw.rect(window, BLACK, wall_object)

def Q_train_and_test():
	pong = Game()

	while True:
		pong.update_q()
		if pong.finished_training:
			break

	x_values = [i[0] for i in pong.training_stats]
	y_values = [i[1] for i in pong.training_stats]
	plt.plot(x_values, y_values, 'b', label = 'Average reward')
	plt.xlabel('Game number')
	plt.ylabel('Average reward')
	plt.title('Mean episode rewards vs episodes')
	plt.grid(True)

	y_values_2 = [9] * len(x_values)
	plt.plot(x_values, y_values_2, 'r', label = '9')

	plt.legend(loc = 'lower right')
	plt.show()

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
		pong.update_test_q()
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

def SARSA_train_and_test():
	pong = Game()

	while True:
		pong.update_sarsa()
		if pong.finished_training:
			break

	x_values = [i[0] for i in pong.training_stats]
	y_values = [i[1] for i in pong.training_stats]
	plt.plot(x_values, y_values, 'b', label = 'Average reward')
	plt.xlabel('Game number')
	plt.ylabel('Average reward')
	plt.title('Mean episode rewards vs episodes')
	plt.grid(True)

	y_values_2 = [9] * len(x_values)
	plt.plot(x_values, y_values_2, 'r', label = '9')

	plt.legend(loc = 'lower right')
	plt.show()

	pygame.init()
	fps = pygame.time.Clock()

	window = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), 0, 32)
	pygame.display.set_caption('Pong Game')

	scale = Scale()
	pong.ball = Ball()
	pong.paddle = Paddle()
	pong.state = (pong.ball.x, pong.ball.y, pong.ball.velocity_x, pong.ball.velocity_y, pong.paddle.y)
	pong.sarsa_agent.epsilon = 0
	pong.is_active = True
	pong.score = 0
	pong.game_number = 0

	while True:
		draw(pong.ball, pong.paddle, scale, window)
		pong.update_test_sarsa()
		if pong.finished_testing:
			break

		pygame.display.update()
		fps.tick(30)

	x_values = [i[0] for i in pong.test_stats]
	y_values = [i[1] for i in pong.test_stats]
	print(pong.test_stats)
	print(float(sum(y_values))/len(y_values))

	plt.plot(x_values, y_values)
	plt.xlabel('Game number')
	plt.ylabel('Reward')
	plt.title('Rewards/game for 200 test games')
	plt.show()

#Run only one agent at a time, a lot of parameters of the Game class are shared, so running both consecutively might give weird results
if __name__ == '__main__':
	# SARSA_train_and_test()
	Q_train_and_test()