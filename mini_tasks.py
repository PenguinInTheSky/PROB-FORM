from collections import defaultdict

from env import MyEnv
from gym_subgoal_automata_wrapper import OfficeWorldAbstractLabelExtractor
from constants import SENSOR_FALSE_CONFIDENCE, SENSOR_TRUE_CONFIDENCE
from objects import CheckPoint
from minigrid.core.grid import Grid
from minigrid.wrappers import FlatObsWrapper, FullyObsWrapper
from minigrid.manual_control import ManualControl
import gymnasium as gym
from rm import RewardMachine
from minigrid.core.world_object import Goal, Lava

class TwoYellow(MyEnv):
	def __init__(self, **kwargs):
		self.size = 5
		super().__init__(self.size, **kwargs)
	
	@staticmethod
	def _gen_mission():
		return "fetch all yellow balls"
	
	def _gen_constants(self):
		return ["o1", "o2"]
	
	def _gen_predicates(self):
		return ["yellow"]
	
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
		self.grid = Grid(width, height)
		self.grid.wall_rect(0, 0, width, height)
				
		# add objects, including checkpoints and goal
		self.objects = []
	
		tmp = 0
		# generate and place checkpoints: 2 yellow
		for c in ["yellow"]:
			# place object randomly in the grid, avoid placing on walls
			for _ in range(2):
				checkpoint = CheckPoint(c)
	
				# add mapping to language
				self.language.add_constant_mapping(self.constants[tmp], checkpoint)
				# add rules to language
				self.language.add_rule(c, [self.constants[tmp]])

				self.objects.append(checkpoint)
				self.place_obj(checkpoint, top=(1, 1), size=(width - 3, height - 3))
				tmp += 1
		
		self.place_agent()
		
		# generate mission
		self.mission = self._gen_mission()
	
	def _gen_reward_machine(self):
		rm_states = ['u0', 'uA']
	
		rm_state_transitions = defaultdict(list)
		rm_state_transitions[('u0', 'uA')] = ('universal', 'yellow', [])
		
		rm_rewards = defaultdict(float)
		rm_rewards[('u0', 'uA')] = 1
	
		return RewardMachine(self, rm_states, rm_state_transitions, rm_rewards)

	@staticmethod
	def get_num_rm_states():
	 	return 2

def make_two_yellow(**kwargs):
	env = TwoYellow(**kwargs)
	env = FullyObsWrapper(env)
	env = FlatObsWrapper(env)
	return env

gym.register(
	id="TwoYellow",
	entry_point=make_two_yellow,
)

# show the grid
if __name__ == "__main__":
	env = TwoYellow(render_mode="human")
	manual_control = ManualControl(env, seed=42)
	manual_control.start()