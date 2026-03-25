from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall, Ball
from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.constants import COLOR_NAMES
import gymnasium as gym

class MyEnv(MiniGridEnv):
  def __init__(self, size=10, **kwargs):
    mission_space = MissionSpace(mission_func=self._gen_mission)
    super().__init__(
      mission_space=mission_space,
      grid_size=size,
      max_steps=200,
      **kwargs,
    )
    
  def _gen_grid(self, width, height):
    # size 10 * 10, wall at column 5, gap at (5, 5), goal at (9, 9), agent at (1, 1), yellow balls at (3, 3), (2, 6), (4, 5), (2, 7), blue ball at (7, 7), (1, 3)
    self.grid = Grid(width, height)
    self.grid.wall_rect(0, 0, width, height)
    
    # generate walls with a gap at (5, 5)
    for i in range(1, height-1):
      if i != 5:
        self.grid.set(5, i, Wall())
        
    # place objects
    self.put_obj(Ball('yellow'), 3, 3)
    self.put_obj(Ball('yellow'), 2, 6)
    self.put_obj(Ball('yellow'), 4, 5)
    self.put_obj(Ball('yellow'), 2, 7)
    self.put_obj(Ball('blue'), 7, 7)
    self.put_obj(Ball('blue'), 1, 3)
    self.put_obj(Goal(), 9, 9)
    self.place_agent()
    
    # generate mission
    self.mission = self._gen_mission()
    
  # def step(self):
  #   obs, reward, done, info = super().step()
  #   return obs, reward, done, info
  #   # TODO: implement the reward machine
    
  @staticmethod
  def _gen_mission():
    return "fetch all yellow balls, then one blue ball, and reach the goal"
  

# show the grid
if __name__ == "__main__":
	env = MyEnv(render_mode="human")
	manual_control = ManualControl(env, seed=42)
	manual_control.start()