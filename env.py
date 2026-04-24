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

from gym_subgoal_automata_wrapper import OfficeWorldAbstractLabelExtractor, NoisyLabelingFunctionComposer
from objects import CheckPoint

from rm import RewardMachine
from language import Language

from minigrid.core.constants import COLOR_TO_IDX, OBJECT_TO_IDX, STATE_TO_IDX

from constants import SENSOR_FALSE_CONFIDENCE, SENSOR_TRUE_CONFIDENCE

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
	
	def __init__(self, size=5, **kwargs):
		mission_space = MissionSpace(mission_func=self._gen_mission)
		super().__init__(
			mission_space=mission_space,
			grid_size=size,
			max_steps=1500,
			**kwargs,
		)
		
		# redefine default action space
		self.actions = MyEnv.Actions
		self.action_space = gym.spaces.Discrete(len(self.actions))
	
		self.rm = RewardMachine(self)
		
		# initialize the language
		self.constants = ["o0", "o1"]
		self.predicates = ["blue", "yellow"]
		# self.constants = ["o0", "o1", "o2", "o3", "o4", "o5", "o6", "o7", "o8", "o9", "o10", "o11"]
		# self.predicates = ["yellow", "blue", "purple", "red", "grey", "green", "goal"]
		self.language = Language(self.constants, self.predicates)

		# create label extractors for each constant, and compose them together
		self.label_funs = []
		for constant in self.constants:
			label_fun = OfficeWorldAbstractLabelExtractor(
				sensor_true_confidence=SENSOR_TRUE_CONFIDENCE,
				sensor_false_confidence=SENSOR_FALSE_CONFIDENCE,
				label=constant,
				value_true_prior= 0.5
			)
			self.label_funs.append(label_fun)
		self.label_extractor = NoisyLabelingFunctionComposer(self.label_funs)

	def __str__(self):
		lines = []
		for y in range(self.height):
			row = []
			for x in range(self.width):
				if self.agent_pos is not None and (x, y) == tuple(self.agent_pos):
					row.append('A')
				else:
					cell = self.grid.get(x, y)
					if cell is None:
						row.append('.')
					elif cell.type == 'wall':
						row.append('#')
					elif cell.type == 'checkpoint':
						row.append(cell.color[0].upper())
					elif cell.type == 'goal':
						row.append('G')
					else:
						row.append('?')
			lines.append(' '.join(row))
		lines.append(f"RM state: {self.rm.current_state} | steps: {self.step_count}/{self.max_steps}")
		return '\n'.join(lines)

	def _gen_grid(self, width=5, height=5):
		# size 10 * 10, wall at column 5, gap at (5, 5), goal at (9, 9), agent at (1, 1), yellow balls at (3, 3), (2, 6), (4, 5), (2, 7), blue ball at (7, 7), (1, 3)
		self.grid = Grid(width, height)
		self.grid.wall_rect(0, 0, width, height)
		
		# internal walls
		# self.grid.vert_wall(6, 1, 2)
		# self.grid.vert_wall(6, 4, 6)
		# self.grid.vert_wall(6, 11, 1)
		# self.grid.horz_wall(1, 6, 1)
		# self.grid.horz_wall(3, 6, 3)
		# self.grid.horz_wall(7, 7, 2)
		# self.grid.horz_wall(10, 7, 2)
				
		# add objects, including checkpoints and goal
		self.objects = []
	
		tmp = 0
		# generate and place checkpoints: 2 yellow, 2 red, 2 blue, 2 purple, 2 grey, 2 green 
		for c in ["blue", "yellow"]: # ["yellow", "blue", "red", "purple", "grey", "green"]:
			# place object randomly in the grid, avoid placing on walls
			for _ in range(1): # 2
				checkpoint = CheckPoint(c)
	
				# add mapping to language
				self.language.add_constant_mapping(self.constants[tmp], checkpoint)
				# add rules to language
				self.language.add_rule(c, [self.constants[tmp]])

				self.objects.append(checkpoint)
				self.place_obj(checkpoint, top=(1, 1), size=(width - 3, height - 3))
				tmp += 1
		
		# generate and place goal
		# goal = Goal()
		# self.objects.append(goal)
		# self.put_obj(goal, width - 2, height - 2)
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

	def step(
		 	self,
				action: ActType
	) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:	
		self.step_count += 1

		reward = 0
		terminated = False
		truncated = False

		# Get the position in front of the agent
		fwd_pos = self.front_pos

		# Get the contents of the cell in front of the agent
		fwd_cell = self.grid.get(*fwd_pos)

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

		# render the environment if in human mode
		if self.render_mode == "human":
			self.render()

		# get the environment observation
		obs = self.gen_obs()

		# get the object at robot's position
		object = self.grid.get(*self.agent_pos)
	
		# add noise to object
		object_set = set()
		if object is not None:
			object_set.add(self.language.get_constant(object))
		# print(object_set)
		labels = self.label_extractor.get_labels(obs, {"observations": object_set})
		# print("Object observed is:", object)
		# print("Labels observed are:", labels)
  
		# transition the reward machine with the object
		terminated, reward, _ = self.rm.noisy_transition(labels)
		rm_state = self.rm.get_current_int_state()
	
		# check if max steps reached
		if self.step_count >= self.max_steps:
			truncated = True

		return obs, reward, terminated, truncated, {"rm_state": rm_state}

	def get_objects(self):
		return self.objects

	@staticmethod
	def _gen_mission():
		return "fetch one ball, then all yellow balls"
	

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



# assumptions
# robot picks up item which is detected seperately from observations
# item detection is seperated from environment observation
# robot knows exactly what balls are there in the environment
# robot doesn't need to reach the goal position

# so the robot will have a observation/scan of the environment, and a seperate detection of the item it's standing on
# robot has independent noise for both detection

# so need to check if the implementation fits the simulation, for it is rather different from leo's code
# and check if the simulated noise is reasonable for real-world scenarios
# also need to expand the training environment, with walls, and more balls
# need to log the working progress in order to support report writing
# reward machine knows about all the balls