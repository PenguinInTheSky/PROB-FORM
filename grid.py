from __future__ import annotations

from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall
from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv
import gymnasium as gym
from minigrid.wrappers import FlatObsWrapper, FullyObsWrapper

# TODO: fix the definition navigation

class MyEnv(MiniGridEnv):
  # Note: miminum size is 4 or else won't be able to reset the environment
	def __init__(self,
							size=4,
							agent_start_pos=(1,1),
							agent_start_dir=0,
							max_steps: int | None=None,
							**kwargs): 
	 
		self.agent_start_pos = agent_start_pos
		self.agent_start_dir = agent_start_dir
	
		mission_space = MissionSpace(mission_func=self._gen_mission)

		if max_steps is None:
			max_steps = 4 * size**2
	 
		super().__init__(
			mission_space=mission_space,
			grid_size=size,
			max_steps=max_steps,
			**kwargs,
		)
	
	@staticmethod
	def _gen_mission():
		return "grand mission"

	def _gen_grid(self, width, height):
		# Create an empty grid
		self.grid = Grid(width, height)
		# Generate walls
		self.grid.wall_rect(0, 0, width, height)
		
		# Place goal
		self.put_obj(Goal(), width-2, height-2)
	
		# Place agent
		self.place_agent()
	
	def _reward(self):
		return 1


		
def make_myenv(**kwargs):
	env = MyEnv(**kwargs)
	env = FullyObsWrapper(env)
	env = FlatObsWrapper(env)
	# env = RecordEpisodeStatistics(env)
	
	return env

gym.register(
	id="MyEnv-v0",
	entry_point=make_myenv,
)

# # show the grid
# if __name__ == "__main__":
# 	env = MyEnv(render_mode="human")
# 	manual_control = ManualControl(env, seed=42)
# 	manual_control.start()