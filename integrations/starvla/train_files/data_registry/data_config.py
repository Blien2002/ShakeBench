"""ShakeBench data registration for StarVLA stable commit 3422b9f2387b.

Link integrations/starvla to STARVLA_ROOT/examples/ShakeBench.
Model architecture and training remain entirely in StarVLA.
"""

from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import StateActionToTensor, StateActionTransform


class ShakeBenchDataConfig:
    embodiment_tag = EmbodimentTag.NEW_EMBODIMENT
    video_keys = ["video.main", "video.wrist"]
    state_keys = []  # Image/language baseline; raw proprioception and IMU remain in the dataset.
    action_keys = ["action.osc", "action.gripper"]
    action_key_dims = {"action.osc": 6, "action.gripper": 1}
    language_keys = ["annotation.human.task_description"]
    action_indices = list(range(8))

    def modality_config(self):
        return {
            "video": ModalityConfig(delta_indices=[0], modality_keys=self.video_keys),
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=[0], modality_keys=self.language_keys),
        }

    def transform(self):
        return ComposedModalityTransform(
            transforms=[
                StateActionToTensor(apply_to=self.action_keys),
                StateActionTransform(
                    apply_to=self.action_keys,
                    normalization_modes={key: "min_max" for key in self.action_keys},
                ),
            ]
        )


ROBOT_TYPE_CONFIG_MAP = {"shakebench": ShakeBenchDataConfig()}
DATASET_NAMED_MIXTURES = {"shakebench": [("oracle-gamma-zero", 1.0, "shakebench")]}
