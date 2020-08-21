import gym
import numpy as np
from .fixobj import FixedObjectGoalEnv
from envs.utils import quat_from_angle_and_axis
from torchvision.utils import save_image

from vae.import_vae import vae_egg
from vae.import_vae import goal_set_egg


class IntervalGoalEnv(FixedObjectGoalEnv):
	def __init__(self, args):
		FixedObjectGoalEnv.__init__(self, args)

	def generate_goal_old(self):
		# Select a goal for the object position.
		target_pos = self.sim.data.get_joint_qpos('object:joint')[:3]

		# Select a goal for the object rotation.
		if self.args.env!='HandManipulatePen-v0':
			angle = np.pi + (np.random.uniform(-1.0, 1.0))*(np.pi/4.0)
		else:
			angle = np.pi/2.0 + (np.random.uniform(-1.0,1.0))*(np.pi/4.0)
		axis = np.array([0., 0., 1.])
		target_quat = quat_from_angle_and_axis(angle, axis)

		target_quat /= np.linalg.norm(target_quat)  # normalized quaternion
		goal = np.concatenate([target_pos, target_quat])
		return goal.copy()

	def generate_goal_old(self):
		print('interval')
		#goal = goal_set[np.random.randint(5)]
		goal = goal_set_egg[19]
		goal = vae_egg.format(goal)
		#save_image(goal.cpu().view(-1, 3, 84, 84), 'videos/goal/goal.png')
		x, y = vae_egg.encode(goal)
		goal = vae_egg.reparameterize(x, y)
		goal = goal.detach().cpu().numpy()
		goal = np.squeeze(goal)
		return goal.copy()