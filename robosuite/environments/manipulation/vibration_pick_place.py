"""Object-independent ShakeBench pick/place task interface."""

from collections import OrderedDict

from robosuite.environments.manipulation.vibration_pick_place_can import VibrationPickPlaceCan
from robosuite.utils.shakebench_tasks import TaskSpec


def _public_key(key):
    return "object_" + key[4:] if key.startswith("can_") else key


class VibrationPickPlace(VibrationPickPlaceCan):
    """Three stock objects on two surfaces, with object-neutral observations.

    The historical VibrationPickPlaceCan retains its frozen defaults. This new
    task uses TaskSpec and object_start_xy; reset and step expose object_* keys.
    The legacy oracle uses the explicit legacy_oracle_observation adapter.
    """

    def __init__(self, task=None, object_start_xy=(-0.10, -0.13), **kwargs):
        if "can_start_xy" in kwargs:
            raise ValueError("use object_start_xy with VibrationPickPlace")
        if "table_friction" in kwargs or "target_container_friction" in kwargs:
            raise ValueError("TaskSpec owns surface/object contact friction")
        kwargs.setdefault("geometry_profile", "direct_mount_v1")
        super().__init__(task=TaskSpec.from_mapping(task), can_start_xy=object_start_xy, **kwargs)

    def _get_observations(self, force_update=False):
        return OrderedDict((_public_key(k), v) for k, v in super()._get_observations(force_update).items())

    @property
    def policy_observation_keys(self):
        return tuple(_public_key(key) for key in super().policy_observation_keys)

    def observation_contract(self):
        return {_public_key(key): value for key, value in super().observation_contract().items()}

    def get_task_context(self):
        context = self.get_policy_task_context()
        context["object"] = context.pop("can")
        context["task_context"] = {_public_key(k): v for k, v in context["task_context"].items()}
        from robosuite.utils.shakebench_artifacts import payload_hash

        context["task_context_sha256"] = payload_hash(context["task_context"])
        return context
