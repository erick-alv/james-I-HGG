import gym
import numpy as np
from .fixobj import FixedObjectGoalEnv

class IntervalGoalEnv(FixedObjectGoalEnv):
	def __init__(self, args):
		FixedObjectGoalEnv.__init__(self, args)

	def generate_goal_old(self):
		#goals_data = np.empty([20, 84, 84, 3])
		#for i in range(20):
		if self.has_object:
			goal = self.initial_gripper_xpos[:3] + self.target_offset
			#goal = self.initial_gripper_xpos[:3] + [0.15, 0, 0]
			if self.args.env=='FetchSlide-v1':
				goal[0] += self.target_range*0.5
				goal[1] += np.random.uniform(-self.target_range, self.target_range)*0.5
			else:
				goal[0] += np.random.uniform(-self.target_range, self.target_range)
				goal[1] += self.target_range
				#goal[1] += np.random.uniform(-self.target_range, self.target_range) # TODO: changed
			goal[2] = self.height_offset + int(self.target_in_the_air)*0.45
		else:
			goal = self.initial_gripper_xpos[:3] + np.array([np.random.uniform(-self.target_range, self.target_range), self.target_range, self.target_range])
		return goal.copy()
		#self.env.env._move_object(goal)
		#self.env.env._set_arm_visible(False)

		#rgb_array_0 = np.array(self.env.env.render(mode='rgb_array', width=84, height=84, cam_name="cam_0"))
		#from PIL import Image
		#im = Image.fromarray(rgb_array_0.astype(np.uint8))
		#im.show()
		#im.close()
		#goals_data[i] = rgb_array_0


		#np.save('data/Fetch_Env/slide_goals_2', goals_data)
		#

	def generate_goal(self):
		return self.env.env._sample_goal()
