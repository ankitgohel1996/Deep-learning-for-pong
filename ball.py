import random

class Ball:

	"""
	Implementation of ball class as part of the MDP
	"""
	
	def __init__(self):
		"""
		Setting the initial ball position and velocity as mentioned in the docs
		"""
		self.x = 0.5
		self.y = 0.5
		self.velocity_x = 0.03
		self.velocity_y = 0.01
	
	def get_position(self):
		"""
		Get current ball position as an (x,y) tuple
		"""
		return (self.x, self.y)

	def bounce(self):
		"""
		Update ball position and velocity if it bounces off the top, bottom or left of the screen
		"""
		if self.y < 0:
			self.y = -self.y
			self.velocity_y = -self.velocity_y
		if self.y > 1:
			self.y = 2 - self.y
			self.velocity_y = -self.velocity_y
		if self.x < 0:
			self.x = -self.x
			self.velocity_x = -self.velocity_x
			
		#Added this for testing pygame as the paddle doesnt move, hence the ball bounces off the right side as well
		# if self.x > 1:
		# 	self.hit_paddle()

	def hit_paddle(self):
		"""
		Update positions and velocities if the ball hits the paddle
		"""
		U = random.uniform(-0.015, 0.015)
		V = random.uniform(-0.03, 0.03)

		self.x = 2 - self.x
		self.velocity_x = -self.velocity_x + U
		self.velocity_y = self.velocity_y + V

		if self.velocity_x < 0:
			self.velocity_x = max(-1.0, min(-0.03, self.velocity_x))
		if self.velocity_x > 0:
			self.velocity_x = min(1.0, min(0.03, self.velocity_x))
		if self.velocity_y < 0:
			self.velocity_y = max(-1.0, self.velocity_y)
		if self.velocity_y > 0:
			self.velocity_y = min(1.0, self.velocity_y)

	def update(self):
		"""
		Update ball position based on calculated velocities
		"""
		self.x += self.velocity_x
		self.y += self.velocity_y
		self.bounce()