from collections import defaultdict

class RewardMachine():
  def __init__(self, env):
    self.env = env
    self.states = ['u0', 'u1', 'u2', 'uA']
    self.start_state = 'u0'
    self.state = self.start_state
    self.accept_state = 'uA'
    self.hb = defaultdict(list)
    
    # TODO: put these env specific coordinates in a constant file or something else to avoid magic number
    self.hb['yellow'] = [(3, 3), (2, 6), (4, 5), (2, 7)]
    self.hb['blue'] = [(7, 7), (1, 3)]
    self.hb['goal'] = [(9, 9)]
    
    self.state_transitions = defaultdict(list)
    self.state_transitions[('u0', 'u1')] = ('universal', 'yellow')
    self.state_transitions[('u1', 'u2')] = ('existential', 'blue')
    self.state_transitions[('u2', 'uA')] = ('existential', 'goal')
    
    self.rewards = defaultdict(float)
    # TODO: reward shaping
    self.rewards[('u0', 'u1')] = 0
    self.rewards[('u1', 'u2')] = 0
    self.rewards[('u2', 'uA')] = 1
    
  def reset(self):
    self.state = self.start_state
  
  def is_accepted(self):
    return self.state == self.accept_state
  