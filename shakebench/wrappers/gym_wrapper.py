"""ShakeBench Gym wrapper: the upstream wrapper plus the hidden-state guard."""

from robosuite.wrappers.gym_wrapper import GymWrapper as RobosuiteGymWrapper
from shakebench.utils.privilege import PRIVILEGED_NAMESPACE


class GymWrapper(RobosuiteGymWrapper):
    """Refuse privileged observations so a policy cannot train on hidden state.

    The check runs against the same selected observation keys as the previous
    robosuite-side guard, so an explicit privileged request still fails during
    construction.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if any(str(key).startswith(PRIVILEGED_NAMESPACE) for key in self.keys):
            raise ValueError("GymWrapper refuses privileged observation keys")
