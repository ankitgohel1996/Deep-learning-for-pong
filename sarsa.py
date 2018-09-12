import random
from collections import defaultdict

class SARSA:

	def __init__(self, alpha, gamma, epsilon):
		self.alpha = alpha
		self.gamma = gamma
		self.epsilon = epsilon
		self.q = defaultdict(int)
		self.frequencies = defaultdict(int)
		self.actions = [-1, 0, 1]

	def get_q_value(self, state, action):
		return self.q[(state,action)]

	def decay(self, state, action):
		return float(self.alpha)/float((self.alpha + self.frequencies[(state, action)]))

	def choose_action(self, state):
		"""
		Choose an action based on the epsilon greedy strategy
		"""
		guess = random.random()

		if guess <= self.epsilon:
			return random.choice(self.actions)
		else:
			q_values = []
			for action in self.actions:
				q_values.append(self.get_q_value(state, action))

			max_q_value = max(q_values)
			index = q_values.index(max_q_value)
			best_action = self.actions[index]

		return best_action

	def learn(self, previous_state, previous_action, previous_reward, new_state, new_action):
		next_q = self.get_q_value(new_state, new_action)
		self.frequencies[(previous_state, previous_action)] += 1
		
		current_value = self.get_q_value(previous_state, previous_action)

		self.q[(previous_state, previous_action)] = current_value + self.decay(previous_state, previous_action) * ((previous_reward + self.gamma * next_q) - current_value)