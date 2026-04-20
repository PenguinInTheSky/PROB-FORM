from enum import IntEnum

from matplotlib.pylab import seed
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall, Ball
from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv
import gymnasium as gym

from gymnasium.core import ActType, ObsType
from typing import Any, Iterable, SupportsFloat, TypeVar
from minigrid.wrappers import FlatObsWrapper, FullyObsWrapper
import numpy as np

from rm import RewardMachine

from minigrid.core.constants import COLOR_TO_IDX, OBJECT_TO_IDX, STATE_TO_IDX

from constants import OBS_NOISE, PICKED_UP_DETECTION_TRUE_PROB_MEAN, PICKED_UP_DETECTION_TRUE_PROB_STD, PICKED_UP_DETECTION_FALSE_PROB_MEAN, PICKED_UP_DETECTION_FALSE_PROB_STD

class MyEnv(MiniGridEnv):
	# define action space
	class Actions(IntEnum):
		left = 0
		right = 1
		forward = 2
		pickup = 3
		# drop = 4
		# toggle = 5
		# done = 4 # 6
	
	def __init__(self, size=5, **kwargs):
		mission_space = MissionSpace(mission_func=self._gen_mission)
		super().__init__(
			mission_space=mission_space,
			grid_size=size,
			max_steps=200,
			**kwargs,
		)
		
		# redefine default action space
		self.actions = MyEnv.Actions
		self.action_space = gym.spaces.Discrete(len(self.actions))
		self.rm = RewardMachine(self)
		
	def _gen_grid(self, width, height):
		# size 10 * 10, wall at column 5, gap at (5, 5), goal at (9, 9), agent at (1, 1), yellow balls at (3, 3), (2, 6), (4, 5), (2, 7), blue ball at (7, 7), (1, 3)
		self.grid = Grid(width, height)
		self.grid.wall_rect(0, 0, width, height)
		
		# generate walls with a gap at (5, 5)
		# for i in range(1, height-1):
		# 	if i != 5:
		# 		self.grid.set(5, i, Wall())
				
		# place objects
		ball1 = Ball('yellow')
		ball1.id = "ball_1"
		# ball2 = Ball('yellow')
		# ball2.id = "ball_2"
		# ball3 = Ball('yellow')
		# ball3.id = "ball_3"
		# ball4 = Ball('yellow')
		# ball4.id = "ball_4"
		# ball5 = Ball('blue')
		# ball5.id = "ball_5"
		ball6 = Ball('blue')
		ball6.id = "ball_6"
		self.put_obj(ball1, 3, 3)
		# self.put_obj(ball2, 2, 6)
		# self.put_obj(ball3, 4, 5)
		# self.put_obj(ball4, 2, 7)
		# self.put_obj(ball5, 7, 7)
		self.put_obj(ball6, 1, 3)
		self.picked_up_objects = [ball1, ball6, None]
		self.place_agent()
		
		# generate mission
		self.mission = self._gen_mission()
		
	def reset(
		self,
		*,
		seed: int | None = None,
		options: dict[str, Any] | None = None,
	) -> tuple[ObsType, dict[str, Any]]:
		obs, info = super().reset(seed=seed)
		self.rm.reset()
		info["rm_state"] = self.rm.get_current_int_state()
		return obs, info
	
	def add_obs_noise(self, obs, noise_prob=0.1, corrupt_agent_cell=False):
		noisy_obs = obs.copy()
		image = noisy_obs['image'].copy()
		
		mask = np.random.random(image.shape[:2]) < noise_prob

		if not corrupt_agent_cell:
				mask[self.agent_pos[0], self.agent_pos[1]] = False
		
		image[mask, 0] = np.random.randint(0, len(COLOR_TO_IDX), mask.sum())
		image[mask, 1] = np.random.randint(0, len(OBJECT_TO_IDX), mask.sum())
		image[mask, 2] = np.random.randint(0, len(STATE_TO_IDX), mask.sum())

		noisy_obs["image"] = image
		return noisy_obs

	def add_picked_up_detection_noise(self, pick_up_object_id: str):
		# get gaussian probability of true detection, including None
		true_prob = np.random.normal(PICKED_UP_DETECTION_TRUE_PROB_MEAN, PICKED_UP_DETECTION_TRUE_PROB_STD)
		# get gaussian probability of false detection
		false_probs = np.random.normal(PICKED_UP_DETECTION_FALSE_PROB_MEAN, PICKED_UP_DETECTION_FALSE_PROB_STD, size=len(self.picked_up_objects) - 1)
		sum_probs = true_prob + false_probs.sum()
		true_prob /= sum_probs
		false_probs /= sum_probs
		true_label = [(pick_up_object_id, true_prob)]
		false_labels = [(obj, prob) for obj, prob in zip(self.picked_up_objects, false_probs) if obj.id != pick_up_object_id]
		return true_label + false_labels

	def step(
				self, action: ActType
		) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
		
				self.step_count += 1

				reward = 0
				terminated = False
				truncated = False

				# Get the position in front of the agent
				fwd_pos = self.front_pos

				# Get the contents of the cell in front of the agent
				fwd_cell = self.grid.get(*fwd_pos)
				
				picked_up = None

				# Rotate left
				if action == self.actions.left:
						self.agent_dir -= 1
						if self.agent_dir < 0:
								self.agent_dir += 4

				# Rotate right
				elif action == self.actions.right:
						self.agent_dir = (self.agent_dir + 1) % 4

				# Move forward
				elif action == self.actions.forward:
						if fwd_cell is None or fwd_cell.can_overlap():
								self.agent_pos = tuple(fwd_pos)

				# Pick up an object
				elif action == self.actions.pickup:
						if fwd_cell and fwd_cell.can_pickup():
							picked_up = getattr(fwd_cell, "id", None)
							self.grid.set(fwd_pos[0], fwd_pos[1], None)

				# Done action (not used by default)
				elif action == self.actions.done:
						pass

				else:
						raise ValueError(f"Unknown action: {action}")


				if self.step_count >= self.max_steps:
						truncated = True

				if self.render_mode == "human":
						self.render()

				obs = self.gen_obs()
		
				rm_state = self.rm.get_current_int_state()
		
				# TODO: return noisy obs, have the RM transition taking in the noisy picked-up observation
				noisy_obs = self.add_obs_noise(obs, noise_prob=OBS_NOISE, corrupt_agent_cell=False)

				noisy_picked_up_detection = self.add_picked_up_detection_noise(picked_up)
				terminated, reward, _ = self.rm.transition(noisy_picked_up_detection)
				# print(f"Picked up: {picked_up}, Noisy picked up detection: {noisy_picked_up_detection}, RM state: {rm_state}, Reward: {reward}, Terminated: {terminated}")
				if self.step_count >= self.max_steps:
					truncated = True


				return noisy_obs, reward, terminated, truncated, {"rm_state": rm_state}
		
	@staticmethod
	def _gen_mission():
		return "fetch all yellow balls, then one blue ball, and reach the goal"
	

# show the grid
if __name__ == "__main__":
	env = MyEnv(render_mode="human")
	manual_control = ManualControl(env, seed=42)
	manual_control.start()
 
def make_myenv(**kwargs):
	env = MyEnv(**kwargs)
	env = FullyObsWrapper(env)
	env = FlatObsWrapper(env)
	return env

gym.register(
	id="MyEnv-v1",
	entry_point=make_myenv,
)


# TODO: reward shaping for first-order RM
# assumption: noisy observation and noisy picked-up detection, and these two are independent of each other, and the noise is symmetric (false positive and false negative have the same probability)