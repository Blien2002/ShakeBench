"""Seven arm joints plus one gripper position channel."""

from .base import Controller


class JointPositionController(Controller):
    action_dim = 8
