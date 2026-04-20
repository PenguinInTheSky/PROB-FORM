from collections import defaultdict

class RewardMachine():
  def __init__(self, env):
    self.env = env
    self.states = ['u0', 'u1', 'uA']
    self.start_state = 'u0'
    self.state_to_int = {'u0': 0, 'u1': 1, 'uA': 2}
    self.current_state = self.start_state
    self.accept_state = 'uA'
    self.hb = defaultdict(list)
    
    # TODO: put these env specific coordinates in a constant file or something else to avoid magic number
    # TODO: use coordinates for now, might have to change in the future
    self.hb['yellow'] = [(3, 3)]#, (2, 6), (4, 5), (2, 7)]
    self.hb['blue'] = [(1, 3)]
    
    self.state_transitions = defaultdict(list)
    self.state_transitions[('u0', 'u1')] = ('universal', 'yellow')
    self.state_transitions[('u1', 'uA')] = ('universal', 'blue')
    
    self.rewards = defaultdict(float)
    # TODO: reward shaping
    self.rewards[('u0', 'u1')] = 0.6
    self.rewards[('u1', 'uA')] = 1.0
    
    # TODO: buffer for transitions
    # renew the buffer every time we transition to a new state
    self.buffer = []

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
  
  def transition(self, pos):
    reward = 0
    self.step_count += 1
    # TODO: use pos for now, might have to change to obs in the future
    # TODO: implement state transition based on current state and observation
    # what is the form of obs
    
    # print("Adding obs to buffer and checking for state transition...")
    # print("Current state is:", self.current_state)
    # 1. add obs to buffer
    if pos in self.hb['yellow']:
      # print("Adding yellow ball at position", pos, "to buffer")
      self.buffer.append(('yellow', pos))
    if pos in self.hb['blue']:
      # print("Adding blue ball at position", pos, "to buffer")
      self.buffer.append(('blue', pos))
    if pos in self.hb['goal']:
      # print("Adding goal at position", pos, "to buffer")
      self.buffer.append(('goal', pos))
      
    # 2. check if we can transition to a new state based on the buffer and the state transition rules
    for next_state in self.states:
      if (self.current_state, next_state) in self.state_transitions:
        transition_type, required_obs = self.state_transitions[(self.current_state, next_state)]
        if transition_type == 'universal':
          if len(self.hb[required_obs]) == len([obs for obs in self.buffer if obs[0] == required_obs]):
            self.buffer = []
            reward = self.rewards[(self.current_state, next_state)]
            self.current_state = next_state
            break
        elif transition_type == 'existential':
          if any(obs for obs in self.buffer if obs[0] == required_obs):
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