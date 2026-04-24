from minigrid.core.world_object import Lava, Wall, WorldObj, Goal, Door, Key, Ball, Box, Floor
from minigrid.utils.rendering import (
    fill_coords,
    point_in_circle
)
from minigrid.core.constants import COLORS, OBJECT_TO_IDX, IDX_TO_OBJECT, IDX_TO_COLOR

OBJECT_TO_IDX.update({"checkpoint": 11})
IDX_TO_OBJECT = dict(zip(OBJECT_TO_IDX.values(), OBJECT_TO_IDX.keys()))

def decode(type_idx: int, color_idx: int, state: int) -> WorldObj | None:
    """Create an object from a 3-tuple state description"""

    obj_type = IDX_TO_OBJECT[type_idx]
    color = IDX_TO_COLOR[color_idx]

    if obj_type == "empty" or obj_type == "unseen" or obj_type == "agent":
        return None

    # State, 0: open, 1: closed, 2: locked
    is_open = state == 0
    is_locked = state == 2

    if obj_type == "wall":
        v = Wall(color)
    elif obj_type == "floor":
        v = Floor(color)
    elif obj_type == "ball":
        v = Ball(color)
    elif obj_type == "key":
        v = Key(color)
    elif obj_type == "box":
        v = Box(color)
    elif obj_type == "door":
        v = Door(color, is_open, is_locked)
    elif obj_type == "goal":
        v = Goal()
    elif obj_type == "lava":
        v = Lava()
    elif obj_type == "checkpoint":
        v = CheckPoint(color)
    else:
        assert False, "unknown object type in decode '%s'" % obj_type

    return v

WorldObj.decode = staticmethod(decode)

class CheckPoint(WorldObj):
  def __init__(self, color: str):
    super().__init__("checkpoint", color)
    
  def can_overlap(self) -> bool:
    return True

  def can_pickup(self):
    return False

  def render(self, img):
    fill_coords(img, point_in_circle(0.5, 0.5, 0.31), COLORS[self.color])
