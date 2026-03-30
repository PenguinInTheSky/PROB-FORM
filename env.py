from enum import IntEnum

from matplotlib.pylab import seed
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall, Ball
from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.constants import COLOR_NAMES
import gymnasium as gym

from gymnasium.core import ActType, ObsType
from typing import Any, Iterable, SupportsFloat, TypeVar
from minigrid.wrappers import FlatObsWrapper, FullyObsWrapper
import numpy as np

from rm import RewardMachine

class MyEnv(MiniGridEnv):
	# define action space
	class Actions(IntEnum):
		left = 0
		right = 1
		forward = 2
		# pickup = 3
		# drop = 4
		# toggle = 5
		# done = 4 # 6
	
	def __init__(self, size=8, **kwargs):
		mission_space = MissionSpace(mission_func=self._gen_mission)
		super().__init__(
			mission_space=mission_space,
			grid_size=size,
			max_steps=500,
			**kwargs,
		)
		
		# redefine default action space
		self.actions = MyEnv.Actions
		self.action_space = gym.spaces.Discrete(len(self.actions))
		self.rm = RewardMachine(self)
		
	def _gen_grid(self, width, height):
		self.grid = Grid(width, height)
		self.grid.wall_rect(0, 0, width, height)
				
		# place objects
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
		return obs, info
	
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
				
				picked_up = (-1, -1)

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
				else:
						raise ValueError(f"Unknown action: {action}")


				if self.step_count >= self.max_steps:
						truncated = True

				if self.render_mode == "human":
						self.render()

				terminated, reward, _ = self.rm.transition(self.agent_pos)
				obs = self.gen_obs()
		
				# if terminated:
				# 	print("terminated")
				
				if self.step_count >= self.max_steps:
					truncated = True
					# print("truncated")
						
				return obs, reward, terminated, truncated, {}
		
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