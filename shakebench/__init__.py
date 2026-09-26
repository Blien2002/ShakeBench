"""ShakeBench: the shaken-worktable manipulation benchmark built on robosuite.

Importing this package registers the ShakeBench environments with the
robosuite registry, so shakebench.make accepts them.  A plain import of
robosuite does not, which keeps the upstream package free of any dependency
on this one.

Optional GPU, training and WebSocket dependencies stay out of this import path;
they load inside the entry point that needs them.
"""

from robosuite.environments.base import make

from shakebench.environments.panel_operation import PanelOperation
from shakebench.environments.vibration_pick_place import VibrationPickPlace
from shakebench.models.arenas import ShakeBenchArena

__all__ = ["PanelOperation", "ShakeBenchArena", "VibrationPickPlace", "make"]
