from ball import Ball
from qlearning import QLearning
from paddle import Paddle
from sarsa import SARSA
import math, pprint
import nnet_agent

pp = pprint.PrettyPrinter(indent=4)

class Game:
	def __init__(self):
		self.ball = Ball()
		self.paddle = Paddle()
		self.agent = QLearning(10, 0.7, 0.05)
		self.sarsa_agent = SARSA(10, 0.7, 0.05)
		self.state = (self.ball.x, self.ball.y, self.ball.velocity_x, self.ball.velocity_y, self.paddle.y)
		self.score = 0
		self.reward = 0
		self.game_number = 0
		self.scores = []
		self.finished_training = False
		self.finished_testing = False
		self.is_active = True
		self.previous_state = None
		self.previous_action = None	
		self.training_stats = []
		self.test_stats = []

	def discretize_state(self):
		if self.is_active == False:
			return (-1,-1,-1,-1,-1)

		if self.ball.velocity_x > 0:
			discrete_velocity_x = 1
		else:
			discrete_velocity_x = -1

		if self.ball.velocity_y >= 0.015:
			discrete_velocity_y = 1
		elif self.ball.velocity_y <= -0.015:
			discrete_velocity_y = -1
		else:
			discrete_velocity_y = 0

		discrete_paddle = min(11, int(math.floor(12 * self.paddle.y/(1 - self.paddle.height))))

		discrete_ball_x =  min(11, int(math.floor(12 * self.ball.x)))
		discrete_ball_y =  min(11, int(math.floor(12 * self.ball.y)))

		return (discrete_ball_x, discrete_ball_y, discrete_velocity_x, discrete_velocity_y, discrete_paddle)

	def end_game(self):
		if len(self.scores) == 1000:
			self.scores = self.scores[1:]
		self.scores.append(self.score)
		self.score = 0
		self.game_number += 1
		self.is_active = False

		if self.game_number%1000 == 0:
			average = float(sum(self.scores))/1000.0
			print(self.game_number, average)
			self.training_stats.append((self.game_number, average))

		if self.game_number == 20000:
			self.finished_training = True

	def end_test_game(self):
		self.test_stats.append((self.game_number, self.score))
		self.game_number += 1
		self.score = 0
		self.is_active = False

		if self.game_number == 200:
			self.finished_testing = True

	def check_terminal_state(self, mode):
		if self.ball.x > self.paddle.x:
			if self.ball.y > self.paddle.y and self.ball.y < self.paddle.y + self.paddle.height:
				self.ball.hit_paddle()
				self.score += 1
				return True
			else:
				if mode == 'test':
					self.end_test_game()
					return False
				else:
					self.end_game()
					return False
		else:
			return False

	def update_q(self):
		hit_paddle = self.check_terminal_state('train')
		discrete_state = self.discretize_state()

		if self.is_active == False:
			self.reward = -1.0
			if self.previous_state is not None:
				self.agent.learn(self.previous_state, self.previous_action, self.reward, discrete_state)
			self.previous_state = None
			self.ball = Ball()
			self.paddle = Paddle()
			self.is_active = True
			return

		if hit_paddle is True:
			self.reward = 1.0

		if self.previous_state != None:
			self.agent.learn(self.previous_state, self.previous_action, self.reward, discrete_state)

		new_state = self.discretize_state()
		new_action = self.agent.choose_action(new_state)

		self.previous_state = new_state
		self.previous_action = new_action
		self.paddle.update(new_action)
		self.ball.update()
		self.reward = 0.0

	def update_sarsa(self):
		hit_paddle = self.check_terminal_state('train')
		discrete_state = self.discretize_state()
		action = self.sarsa_agent.choose_action(discrete_state)

		if self.is_active == False:
			self.reward = -1.0
			if self.previous_state is not None:
				self.sarsa_agent.learn(self.previous_state, self.previous_action, self.reward, discrete_state, action)
			self.previous_state = None
			self.ball = Ball()
			self.paddle = Paddle()
			self.is_active = True
			return

		if hit_paddle is True:
			self.reward = 1.0

		if self.previous_state != None:
			self.sarsa_agent.learn(self.previous_state, self.previous_action, self.reward, discrete_state, action)

		new_state = self.discretize_state()
		new_action = self.sarsa_agent.choose_action(new_state)

		self.previous_state = new_state
		self.previous_action = new_action
		self.paddle.update(new_action)
		self.ball.update()
		self.reward = 0.0

	def update_test_q(self):
		hit_paddle = self.check_terminal_state('test')
		discrete_state = self.discretize_state()

		if self.is_active == False:
			self.ball = Ball()
			self.paddle = Paddle()
			self.is_active = True
			return

		new_state = self.discretize_state()
		new_action = self.agent.choose_action(new_state)

		self.paddle.update(new_action)
		self.ball.update()

	def update_test_sarsa(self):
		hit_paddle = self.check_terminal_state('test')
		discrete_state = self.discretize_state()

		if self.is_active == False:
			self.ball = Ball()
			self.paddle = Paddle()
			self.is_active = True
			return

		new_state = self.discretize_state()
		new_action = self.sarsa_agent.choose_action(new_state)

		self.paddle.update(new_action)
		self.ball.update()

	def init_nagent(self, W, B, normalize):
		self.nagent = nnet_agent.NAgent(W, B, normalize)

	def update_test_nagent(self):
		hit_paddle = self.check_terminal_state('test')
	
		if self.is_active == False:
			self.ball = Ball()
			self.paddle = Paddle()
			self.is_active = True
			return

		new_state = (self.ball.x, self.ball.y, self.ball.velocity_x, self.ball.velocity_y, self.paddle.y)
		new_action = self.nagent.choose_action(new_state)
		# print(new_action)

		self.paddle.update(new_action)
		self.ball.update()
		self.state = (self.ball.x, self.ball.y, self.ball.velocity_x, self.ball.velocity_y, self.paddle.y)
