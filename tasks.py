from env import MyEnv
from gym_subgoal_automata_wrapper import OfficeWorldAbstractLabelExtractor
from constants import SENSOR_FALSE_CONFIDENCE, SENSOR_TRUE_CONFIDENCE
from objects import CheckPoint
from minigrid.core.grid import Grid
from minigrid.wrappers import FlatObsWrapper, FullyObsWrapper
from minigrid.manual_control import ManualControl
import gymnasium as gym
from rm import RewardMachine

class OneBlueTwoYellow(MyEnv):
	def __init__(self, **kwargs):
		self.size = 5
		super().__init__(self.size, **kwargs)

	@staticmethod
	def _gen_mission():
		return "fetch one blue ball, then all yellow balls"
	
	def _gen_constants(self):
		return ["o1", "o2", "o3", "o4"]
	
	def _gen_predicates(self):
		return ["blue", "yellow"]
	
	def _gen_label_fun(self):
		label_funs = []
		for constant in self.constants:
			label_fun = OfficeWorldAbstractLabelExtractor(
				sensor_true_confidence=SENSOR_TRUE_CONFIDENCE,
				sensor_false_confidence=SENSOR_FALSE_CONFIDENCE,
				label=constant,
				value_true_prior= 1 / ((self.size - 2)* (self.size - 2))
			)
			label_funs.append(label_fun)
  
		return label_funs
	 
	def _gen_grid(self, width=5, height=5):
		# size 10 * 10, wall at column 5, gap at (5, 5), goal at (9, 9), agent at (1, 1), yellow balls at (3, 3), (2, 6), (4, 5), (2, 7), blue ball at (7, 7), (1, 3)
		self.grid = Grid(width, height)
		self.grid.wall_rect(0, 0, width, height)
				
		# add objects, including checkpoints and goal
		self.objects = []
	
		tmp = 0
		# generate and place checkpoints: 2 yellow, 2 red, 2 blue, 2 purple, 2 grey, 2 green 
		for c in ["blue", "yellow"]: # ["yellow", "blue", "red", "purple", "grey", "green"]:
			# place object randomly in the grid, avoid placing on walls
			for _ in range(2): # 2
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
	
	def _gen_reward_machine(self):
		return RewardMachine(self, None, None, None)
	
	
# show the grid
if __name__ == "__main__":
	env = OneBlueTwoYellow(render_mode="human")
	manual_control = ManualControl(env, seed=42)
	manual_control.start()
 
def make_myenv(**kwargs):
	env = OneBlueTwoYellow(**kwargs)
	env = FullyObsWrapper(env)
	env = FlatObsWrapper(env)
	return env

gym.register(
	id="OneBlueTwoYellow",
	entry_point=make_myenv,
)
