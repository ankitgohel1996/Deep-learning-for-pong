class Paddle:
	"""
	Implementation of the paddle class as part of the MDP
	"""

	def __init__(self):
		self.height = 0.2
		self.x = 1.0
		self.y = 0.5 - self.height/2
		self.velocity = 0.04

	def get_position(self):
		return self.y

	def check_bounds(self):
		"""
		If the top/bottom of the paddle goes off the screen
		"""

		self.y = min(1-self.height, self.y)
		self.y = max(0, self.y)
		
	def update(self, action):
		self.y += action * self.velocity
		self.check_bounds()