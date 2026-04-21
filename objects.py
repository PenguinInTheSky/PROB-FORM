from minigrid.core.world_object import WorldObj
from minigrid.utils.rendering import (
    fill_coords,
    point_in_circle
)
from minigrid.core.constants import COLORS

class CheckPoint(WorldObj):
  def __init__(self, color: str):
    super().__init__("checkpoint", color)

  def can_overlap(self) -> bool:
    return True

  def can_pickup(self):
    return False

  def render(self, img):
    fill_coords(img, point_in_circle(0.5, 0.5, 0.31), COLORS[self.color])

