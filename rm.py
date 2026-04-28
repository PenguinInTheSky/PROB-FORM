from collections import defaultdict
from objects import CheckPoint
from minigrid.core.world_object import Goal, WorldObj

from constants import ADD_TO_BUFFER_PROB_THRESHOLD, PROB_TRANSITION_THRESHOLD_EXISTENTIAL, PROB_TRANSITION_THRESHOLD_UNIVERSAL
import numpy as np 
class RewardMachine():
  def __init__(self, env, states, state_transitions, rewards):
    self.env = env
    
    self.states = ['u0', 'u1', 'u2']#, 'u3', 'uA']
    self.start_state = 'u0'
    self.state_to_int = {'u0': 0, 'u1': 1, 'u2': 2}#, 'u3': 3, 'uA': 4}
    self.current_state = self.start_state
    self.accept_state = 'u2'#'uA'
    self.hb = defaultdict(list)
    
    self.state_transitions = defaultdict(list)
    self.state_transitions[('u0', 'u1')] = ('existential', 'blue', [])
    self.state_transitions[('u1', 'u2')] = ('universal', 'yellow', [])
    # self.state_transitions[('u2', 'u3')] = ('propositional', 'purple', ['o7'])
    # self.state_transitions[('u3', 'uA')] = ('propositional', 'goal', [])
    # self.state_transitions[('uA', 'uA')] = ('propositional', 'True', [])
    
    self.rewards = defaultdict(float)
    # TODO: reward shaping
    self.rewards[('u0', 'u1')] = 0.8
    self.rewards[('u1', 'u2')] = 0.7
    # self.rewards[('u2', 'u3')] = 0.6
    # self.rewards[('u3', 'uA')] = 0.5
    
    # TODO: buffer for transitions
    # renew the buffer every time we transition to a new state
    # buffer contains the key observational objects (checkpoints and goal) that the agent has seen since the last state transition
    self.buffer = []
    self.noisy_buffer = dict() # label -> prob of being picked up, only for objects that are not yet added to the buffer

    self.reward = 0
    self.step_count = 0
  
  @staticmethod
  def get_num_states():
    return 3  
  
  def get_current_int_state(self):
    return self.state_to_int[self.current_state]
    
  def reset(self):
    self.current_state = self.start_state
    self.buffer = []
    self.step_count = 0
    self.reward = 0
  
  def is_accepted(self):
    return self.current_state == self.accept_state
  
  def get_reward(self):
    return self.reward
  
  def noisy_transition(self, labels):
    # print(labels)
    reward = 0
    self.step_count += 1
    
    # 1. update buffer with noisy labels
    # for key and value in labels, if value > threshold, add key to buffer with some probability
    for (label, prob) in labels.items():
      if prob > ADD_TO_BUFFER_PROB_THRESHOLD:
        # add obj_id to buffer with some probability
        if label not in self.noisy_buffer:
          self.noisy_buffer[label] = prob
        else:
          self.noisy_buffer[label] = max(self.noisy_buffer[label], prob)
          
    # print("Updated noisy buffer is:", self.noisy_buffer)
    
    # 2. check transition with new buffer for universal condition
    # Mote: no seperate goal check 
    for next_state in self.states:
      if (self.current_state, next_state) in self.state_transitions:
        transition_type, predicate, constants = self.state_transitions[(self.current_state, next_state)]
        # Note: universal unary predicates only
        # print("Checking transition from", self.current_state, "to", next_state)
        # print("Transition type:", transition_type)
        # print("Predicate:", predicate)
        # print("Constants:", constants)
        if transition_type == 'universal':
          min_prob = np.min([
            self.noisy_buffer[label] if label in self.noisy_buffer else 0.0
            for label in self.env.constants if self.env.language.check_rule(predicate, [label])
          ])
          # print("Labels that satisfy the predicate are:", [label for label in self.env.constants if self.env.language.check_rule(predicate, [label])])
          if min_prob > PROB_TRANSITION_THRESHOLD_UNIVERSAL:
            self.noisy_buffer = {}
            reward = self.rewards[(self.current_state, next_state)]
            self.current_state = next_state
            break
          
        elif transition_type == 'existential':
          max_prob = np.max([
            self.noisy_buffer[label] if label in self.noisy_buffer else 0.0
            for label in self.env.constants if self.env.language.check_rule(predicate, [label])
          ])
          # print("Labels that satisfy the predicate are:", [label for label in self.env.constants if self.env.language.check_rule(predicate, [label])])
          if max_prob > PROB_TRANSITION_THRESHOLD_EXISTENTIAL:
            self.noisy_buffer = {}
            reward = self.rewards[(self.current_state, next_state)]
            self.current_state = next_state
            break
        elif transition_type == 'propositional':
          # print("Checking propositional transition with constants:", constants)
          # print("With predicate:", predicate)
          if predicate == True:
            self.noisy_buffer = {}
            reward = self.rewards[(self.current_state, next_state)]
            self.current_state = next_state
            break
          # make sure all constants are in buffer         
          # all constant in constants must be guarenteed to be mapped to an object beforehand during task and environment definition
          elif np.min(self.noisy_buffer[label] if label in self.noisy_buffer else 0.0 for label in constants) > PROB_TRANSITION_THRESHOLD_UNIVERSAL:
            self.noisy_buffer = {}
            reward = self.rewards[(self.current_state, next_state)]
            self.current_state = next_state
            break
            
    # 3. if we can transition, update the current state and reset the buffer, and set the reward based on the rewards dict
    # if we cannot transition, do nothing
    
    # 4. return done, reward, info (perhaps)
    # print("And next state is:", self.current_state)
    # print("Reward is:", self.reward)
    done = self.is_accepted()
    # if done:
      # print("Task completed!")
    return done, reward, {}
    
    # 3. check transition with new buffer for existential condition
    
    # 4. reset buffer if transition happens, and update reward and current state accordingly
    pass
  
  def transition(self, obj: WorldObj):
    reward = 0
    self.step_count += 1
    # TODO: use pos for now, might have to change to obs in the future
    # TODO: implement state transition based on current state and observation
    # what is the form of obs
    
    # print("Adding obs to buffer and checking for state transition...")
    # print("Current state is:", self.current_state)
    
    # 1. add obs to buffer      
    if obj is not None:
      self.buffer.append(obj)
      
    # 2. check if we can transition to a new state based on the buffer and the state transition rules
    # print("Next state candidates are:", [next_state for next_state in self.states if (self.current_state, next_state) in self.state_transitions])
    for next_state in self.states:
      if (self.current_state, next_state) in self.state_transitions:
        transition_type, predicate, constants = self.state_transitions[(self.current_state, next_state)]
        # Note: universal unary predicates only
        # print("Checking transition from", self.current_state, "to", next_state)
        # print("Transition type:", transition_type)
        # print("Predicate:", predicate)
        # print("Constants:", constants)
        if transition_type == 'universal':
          count = len([const for const in self.env.constants if self.env.language.check_rule(predicate, [const])])
          count_in_buffer = len([obj for obj in self.buffer if self.env.language.get_constant(obj) != None and self.env.language.check_rule(predicate, [self.env.language.get_constant(obj)])])
          if count_in_buffer == count:
            self.buffer = []
            reward = self.rewards[(self.current_state, next_state)]
            self.current_state = next_state
            break
          
        elif transition_type == 'existential':
          if any(obj for obj in self.buffer if self.env.language.get_constant(obj) != None and self.env.language.check_rule(predicate, [self.env.language.get_constant(obj)])):
            self.buffer = []
            reward = self.rewards[(self.current_state, next_state)]
            self.current_state = next_state
            break
        
        elif transition_type == 'propositional':
          # print("Checking propositional transition with constants:", constants)
          # print("With predicate:", predicate)
          if predicate == True:
            self.buffer = []
            reward = self.rewards[(self.current_state, next_state)]
            self.current_state = next_state
            break
          elif predicate == "goal":
            if isinstance(obj, Goal):
              self.buffer = []
              reward = self.rewards[(self.current_state, next_state)]
              self.current_state = next_state
              break
          # make sure all constants are in buffer         
          # all constant in constants must be guarenteed to be mapped to an object beforehand during task and environment definition
          elif all(self.env.language.get_object(const) in self.buffer for const in constants):
            self.buffer = []
            reward = self.rewards[(self.current_state, next_state)]
            self.current_state = next_state
            break
            
    # 3. if we can transition, update the current state and reset the buffer, and set the reward based on the rewards dict
    # if we cannot transition, do nothing
    
    # 4. return done, reward, info (perhaps)
    # print("And next state is:", self.current_state)
    # print("Reward is:", self.reward)
    done = self.is_accepted()
    # if done:
      # print("Task completed!")
    return done, reward, {}

# for each step (env.step): call transition first, then get reward of RM
# transition will set reward straight away

# reward of reward machine is not cumulative

# ASSUMPTION: we know beforehand all the balls that we need to pick up / in environment