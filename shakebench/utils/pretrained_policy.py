"""Local LeRobot-style policies using ShakeBench's native state and OSC actions."""

import json
from collections import deque
from dataclasses import fields
from importlib import import_module
from pathlib import Path

import numpy as np

from shakebench.utils.rollout import CAMERAS, validated_actions

CONTRACT_FILE = "shakebench_policy.json"
FEATURE_KEYS = (*CAMERAS, "observation.state", "action")


def policy_class(spec):
    """Import an explicitly selected external model class (module:class)."""
    module, name = spec.split(":", 1)
    return getattr(import_module(module), name)


def load_config(cls, checkpoint, device):
    """Decode declared config fields, excluding upstream training bookkeeping."""
    import draccus

    raw = json.loads((Path(checkpoint) / "config.json").read_text())
    # Saved checkpoints may contain `device: cuda:0`, while LeRobot v0.3.3
    # accepts only `cuda` during config construction. Device is runtime state.
    names = {field.name for field in fields(cls.config_class)} - {"device"}
    config = draccus.decode(cls.config_class, {key: value for key, value in raw.items() if key in names})
    config.device = device
    return config


def validate_offsets(offsets):
    if not offsets or any(type(x) is not int or x > 0 for x in offsets) or offsets[-1] != 0:
        raise ValueError("observation offsets must be nonpositive integers ending in zero")
    if offsets != sorted(set(offsets)):
        raise ValueError("observation offsets must be strictly increasing")
    return offsets


class PretrainedPolicy:
    """One observation per simulation step; the model owns its action queue.

    Simulation pauses during inference, as in the shared benchmark runner.
    This adapter intentionally does not claim a hard inference deadline.
    """

    chunk_size = 1

    def __init__(self, *, checkpoint, device="cuda"):
        import torch
        from safetensors.torch import load_model

        self.checkpoint = Path(checkpoint)
        contract = json.loads((self.checkpoint / CONTRACT_FILE).read_text())
        if contract["action_contract"] != "shakebench.normalized_osc.v1":
            raise ValueError("checkpoint must be fine-tuned on ShakeBench OSC actions")
        if contract["fps"] != 20:
            raise ValueError("checkpoint must match the 20 Hz ShakeBench control rate")
        self.offsets = validate_offsets(contract["observation_offsets"])
        cls = policy_class(contract["policy_class"])
        config = load_config(cls, checkpoint, device)
        if tuple(config.input_features["observation.state"].shape) != (8,):
            raise ValueError("checkpoint must use the native 8D ShakeBench state")
        if tuple(config.output_features["action"].shape) != (7,):
            raise ValueError("checkpoint must produce 7D OSC actions")
        if getattr(config, "use_delta_action", False) or getattr(config, "enable_streaming", False):
            raise ValueError("pose reconstruction and streaming must be disabled")
        if config.n_obs_steps != len(self.offsets):
            raise ValueError("checkpoint history does not match its observation offsets")
        self.model = cls(config)
        # Load normalizers too: inference must restore the fine-tuning dataset statistics.
        load_model(self.model, str(self.checkpoint / "model.safetensors"), strict=True)
        self.model.to(device).eval()
        self.device = torch.device(device)
        self.reset()

    @property
    def identity(self):
        return {
            "adapter": "pretrained_policy",
            "checkpoint": str(self.checkpoint),
            "observation_offsets": self.offsets,
        }

    def reset(self):
        self.history = deque(maxlen=1 - min(self.offsets))
        self.model.reset()

    def predict(self, observation):
        import torch

        frame = {}
        for key in self.model.config.input_features:
            value = np.asarray(observation[key])
            if not np.isfinite(value).all():
                raise ValueError(f"non-finite observation: {key}")
            if key in CAMERAS:
                if value.dtype != np.uint8 or value.ndim != 3 or value.shape[-1] != 3:
                    raise ValueError(f"expected uint8 HWC RGB: {key}")
                tensor = torch.from_numpy(value.copy()).permute(2, 0, 1).float() / 255
            else:
                if value.shape != (8,):
                    raise ValueError("expected 8D ShakeBench state")
                tensor = torch.as_tensor(value.copy(), dtype=torch.float32)
            frame[key] = tensor
        self.history.append(frame)
        selected = [self.history[max(0, len(self.history) - 1 + offset)] for offset in self.offsets]
        batch = {key: torch.stack([item[key] for item in selected])[None].to(self.device) for key in frame}
        batch["task"] = [observation["task"]]
        with torch.inference_mode():
            action = self.model.select_action(batch)
        # Reject malformed/non-finite output before applying actuator saturation.
        action = action.detach().cpu().numpy()
        if action.shape != (1, 7) or not np.isfinite(action).all():
            raise ValueError("model must return one finite 7D action")
        return validated_actions(np.clip(action, -1, 1))


def make_policy(*, checkpoint, device="cuda"):
    return PretrainedPolicy(checkpoint=checkpoint, device=device)
