class Scale:
	"""
	Used to perform the appropriate scaling to convert the given MDP values into values suitable for the pygame window
	"""

	def __init__(self):
		self.screen_width = 500
		self.screen_height = 500
		self.line_thickness = 10

	def scale_ball(self, ball_position):
		scaled_x = ball_position[0] * self.screen_height- self.line_thickness/2
		scaled_y = ball_position[1] * self.screen_height - self.line_thickness/2
		return(int(scaled_x), int(scaled_y))

	def scale_wall(self, wall):
		return wall * self.screen_width

	def scale_paddle(self, paddle):
		return paddle * self.screen_width - self.line_thickness

