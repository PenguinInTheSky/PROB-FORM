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
		self.grid = Grid(width, height)
		self.grid.wall_rect(0, 0, width, height)
				
		# add objects, including checkpoints and goal
		self.objects = []
	
		tmp = 0
		# generate and place checkpoints
		for c in ["blue", "yellow"]:
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
		rm_states = ['u0', 'u1', 'uA']
	
		rm_state_transitions = defaultdict(list)
		rm_state_transitions[('u0', 'u1')] = ('existential', 'blue', [])
		rm_state_transitions[('u1', 'uA')] = ('universal', 'yellow', [])
		
		rm_rewards = defaultdict(float)
		rm_rewards[('u0', 'u1')] = 0.8
		rm_rewards[('u1', 'uA')] = 0.7
	
		return RewardMachine(self, rm_states, rm_state_transitions, rm_rewards)

	@staticmethod
	def get_num_rm_states():
	 	return 3

class TwoYellowThenGoal(MyEnv):
	def __init__(self, **kwargs):
		self.size = 5
		super().__init__(self.size, **kwargs)
	
	@staticmethod
	def _gen_mission():
		return "fetch all yellow balls, then go to the goal"
	
	def _gen_constants(self):
		return ["o1", "o2", "o3"]
	
	def _gen_predicates(self):
		return ["yellow", "goal"]
	
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
		
		# generate and place goal
		goal = Goal()
		self.language.add_constant_mapping("o3", goal)
		self.language.add_rule("goal", ["o3"])
		self.objects.append(goal)
		self.put_obj(goal, width - 2, height - 2)
		
		self.place_agent()
		
		# generate mission
		self.mission = self._gen_mission()
	
	def _gen_reward_machine(self):
		self.num_rm_states = 3
		rm_states = ['u0', 'u1', 'uA']
	
		rm_state_transitions = defaultdict(list)
		rm_state_transitions[('u0', 'u1')] = ('universal', 'yellow', [])
		rm_state_transitions[('u1', 'uA')] = ('propositional', 'goal', ["o3"])
		
		rm_rewards = defaultdict(float)
		rm_rewards[('u0', 'u1')] = 0.7
		rm_rewards[('u1', 'uA')] = 0.8
	
		return RewardMachine(self, rm_states, rm_state_transitions, rm_rewards)

	@staticmethod
	def get_num_rm_states():
	 	return 3

class OneBlueTwoYellowAPurpleThenGoal(MyEnv):
	def __init__(self, **kwargs):
		self.size = 6
		super().__init__(self.size, **kwargs)
	
	@staticmethod
	def _gen_mission():
		return "fetch one blue ball, then all yellow balls, then one specific purple ball, then go to the goal"
	
	def _gen_constants(self):
		return ["o1", "o2", "o3", "o4", "o5", "o6", "o7"]
	
	def _gen_predicates(self):
		return ["blue", "yellow", "purple", "goal"]
	
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
	 
	def _gen_grid(self, width=6, height=6):
		self.grid = Grid(width, height)
		self.grid.wall_rect(0, 0, width, height)
				
		# add objects, including checkpoints and goal
		self.objects = []
	
		tmp = 0
		# generate and place checkpoints
		for c in ["blue", "yellow", "purple"]:
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
		
		# generate and place goal
		goal = Goal()
		self.language.add_constant_mapping("o7", goal)
		self.language.add_rule("goal", ["o7"])
		self.objects.append(goal)
		self.put_obj(goal, width - 2, height - 2)
	
		self.place_agent()
		
		# generate mission
		self.mission = self._gen_mission()
	
	def _gen_reward_machine(self):
		rm_states = ['u0', 'u1', 'u2', 'u3', 'uA']
	
		rm_state_transitions = defaultdict(list)
		rm_state_transitions[('u0', 'u1')] = ('existential', 'blue', [])
		rm_state_transitions[('u1', 'u2')] = ('universal', 'yellow', [])
		rm_state_transitions[('u2', 'u3')] = ('propositional', 'purple', ["o6"])
		rm_state_transitions[('u3', 'uA')] = ('propositional', 'goal', ["o7"])
		
		rm_rewards = defaultdict(float)
		rm_rewards[('u0', 'u1')] = 0.8
		rm_rewards[('u1', 'u2')] = 0.7
		rm_rewards[('u2', 'u3')] = 0.6
		rm_rewards[('u3', 'uA')] = 0.5
	
		return RewardMachine(self, rm_states, rm_state_transitions, rm_rewards)
	
	@staticmethod
	def get_num_rm_states():
	 	return 5

class OneGreenThenGoalLava(MyEnv):
	def __init__(self, **kwargs):
		self.size = 5
		super().__init__(self.size, **kwargs)
	
	@staticmethod
	def _gen_mission():
		return "fetch one green ball, then go to the goal, avoiding lava"
	
	def _gen_constants(self):
		return ["o1", "o2", "o3", "o4"]
	
	def _gen_predicates(self):
		return ["green", "goal", "lava"]
	
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

	def _gen_grid(self, width=6, height=6):
		self.grid = Grid(width, height)
		self.grid.wall_rect(0, 0, width, height)
				
		# add objects, including checkpoints and goal
		self.objects = []
	
		tmp = 0
		# generate and place checkpoints: 2 yellow
		for c in ["green"]:
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
		
		# generate and place goal
		goal = Goal()
		self.language.add_constant_mapping("o3", goal)
		self.language.add_rule("goal", ["o3"])
		self.objects.append(goal)
		self.put_obj(goal, width - 2, height - 2)
	
		# generate and place lava
		lava = Lava()
		self.language.add_constant_mapping("o4", lava)
		self.language.add_rule("lava", ["o4"])
		self.objects.append(lava)	
		self.put_obj(lava, width//2, height//2)
		
		self.place_agent()
	
		# generate mission
		self.mission = self._gen_mission()
	
	def _gen_reward_machine(self):
		rm_states = ['u0', 'u1', 'uR', 'uA']
	
		rm_state_transitions = defaultdict(list)
		rm_state_transitions[('u0', 'u1')] = ('universal', 'green', [])
		rm_state_transitions[('u0', 'uR')] = ('propositional', 'lava', ["o4"])
		rm_state_transitions[('u1', 'uR')] = ('propositional', 'lava', ["o4"])
		rm_state_transitions[('u1', 'uA')] = ('propositional', 'goal', ["o3"])
		
		rm_rewards = defaultdict(float)
		rm_rewards[('u0', 'u1')] = 0.7
		rm_rewards[('u1', 'uA')] = 0.8
		rm_rewards[('u0', 'uR')] = -10.0
		rm_rewards[('u1', 'uR')] = -10.0
	
		return RewardMachine(self, rm_states, rm_state_transitions, rm_rewards)

	@staticmethod
	def get_num_rm_states():
	 	return 4
 
# show the grid
if __name__ == "__main__":
	env = OneBlueTwoYellow(render_mode="human")
	manual_control = ManualControl(env, seed=42)
	manual_control.start()
 
def make_one_blue_two_yellow(**kwargs):
	env = OneBlueTwoYellow(**kwargs)
	env = FullyObsWrapper(env)
	env = FlatObsWrapper(env)
	return env

def make_two_yellow_then_goal(**kwargs):
	env = TwoYellowThenGoal(**kwargs)
	env = FullyObsWrapper(env)
	env = FlatObsWrapper(env)
	return env

def make_one_blue_two_yellow_a_purple_then_goal(**kwargs):
	env = OneBlueTwoYellowAPurpleThenGoal(**kwargs)
	env = FullyObsWrapper(env)
	env = FlatObsWrapper(env)
	return env

def make_one_green_then_goal_lava(**kwargs):
	env = OneGreenThenGoalLava(**kwargs)
	env = FullyObsWrapper(env)
	env = FlatObsWrapper(env)
	return env

gym.register(
	id="OneBlueTwoYellow",
	entry_point=make_one_blue_two_yellow,
)

gym.register(
	id="TwoYellowThenGoal",
	entry_point=make_two_yellow_then_goal,
)

gym.register(
	id="OneBlueTwoYellowAPurpleThenGoal",
	entry_point=make_one_blue_two_yellow_a_purple_then_goal,
)

gym.register(
	id="OneGreenThenGoalLava",
	entry_point=make_one_green_then_goal_lava,
)