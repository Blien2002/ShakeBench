"""StarVLA adapter: message layout, normalization key, resize, registry metadata.

The environment, observation contract, action contract, and rollout live in
robosuite.utils.shakebench_rollout, so any policy implementation can be evaluated
without importing a model-specific module.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from robosuite.utils.shakebench_rollout import (
    ACTION_NAMES,
    CAMERAS,
    STATE_NAMES,
    TASK,
    PolicyTimeoutError,
    ShakeBenchCameraObservation,
    ShakeBenchTaskEnv,
    observation_features,
    rollout_policy,
    task_description,
    validated_actions,
)

# Compatibility aliases: the shared boundary previously lived in this module.
StarVLAObservation = ShakeBenchCameraObservation
StarVLAEnvironment = ShakeBenchTaskEnv
rollout_starvla = rollout_policy


def modality_metadata():
    """StarVLA's GR00T loader mapping, alongside standard LeRobot v2.1 metadata."""
    return {
        "video": {name: {"original_key": f"observation.images.{name}"} for name in ("main", "wrist")},
        "state": {"proprio": {"original_key": "observation.state", "start": 0, "end": 8}},
        "action": {
            "osc": {"original_key": "action", "start": 0, "end": 6, "absolute": False},
            "gripper": {"original_key": "action", "start": 6, "end": 7, "absolute": True},
        },
        "annotation": {"human.task_description": {"original_key": "task_index"}},
    }


class StarVLAPolicy:
    """Image/language baseline using StarVLA's official client and server normalization.

    Train with the supplied ShakeBench registry (include_state=false). Proprioception
    and IMU are recorded for external StarVLA research, but are not model inputs here.
    """

    def __init__(self, *, host="127.0.0.1", port=10093, deadline_s=None):
        if deadline_s is not None and (
            isinstance(deadline_s, bool)
            or not isinstance(deadline_s, (int, float))
            or not np.isfinite(deadline_s)
            or deadline_s <= 0
        ):
            raise ValueError("deadline_s must be a positive number of seconds")
        self.deadline_s = deadline_s
        from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy

        self.client = WebsocketClientPolicy(host, port)
        try:
            meta = self.client.get_server_metadata()
            if "new_embodiment" not in meta.get("available_unnorm_keys", []):
                raise ValueError("server checkpoint must use the ShakeBench data registry and statistics")
            if meta.get("action_keys") != ["action.osc", "action.gripper"]:
                raise ValueError("server action contract does not match the ShakeBench registry")
            if meta.get("state_keys"):
                raise ValueError("this client requires the image/language ShakeBench registry (include_state=false)")
            self.chunk_size = int(meta["action_chunk_size"])
            if self.chunk_size < 1:
                raise ValueError("invalid server action_chunk_size")
        except BaseException:
            self.client.close()
            raise

    @property
    def identity(self) -> dict:
        """Adapter identity recorded next to evaluation results."""
        return {
            "adapter": "starvla_websocket",
            "unnorm_key": "new_embodiment",
            "chunk_size": self.chunk_size,
            "deadline_s": getattr(self, "deadline_s", None),
        }

    def _request_with_deadline(self, query, deadline_s):
        """Send one request with a hard response deadline on the pinned client connection.

        The pinned client's predict_action recv has no timeout, and the simulation
        is paused while waiting, so the deadline is imposed here on its connection.
        """
        from deployment.model_server.tools import msgpack_numpy

        client = self.client
        client._ws.send(client._packer.pack(query))
        try:
            raw = client._ws.recv(timeout=deadline_s)
        except TimeoutError:
            raise PolicyTimeoutError(f"StarVLA server did not answer within {deadline_s}s") from None
        if isinstance(raw, str):
            raise RuntimeError(f"Error in inference server:\n{raw}")
        return msgpack_numpy.unpackb(raw)

    def predict(self, observation):
        # Match the pinned StarVLA loader's PIL resize exactly (including interpolation).
        example = {
            "image": [np.asarray(Image.fromarray(observation[key]).resize((224, 224))) for key in CAMERAS],
            "lang": observation["task"],
        }
        query = {"examples": [example], "unnorm_key": "new_embodiment"}
        deadline_s = getattr(self, "deadline_s", None)
        if deadline_s is None:
            response = self.client.predict_action(query)
        else:
            response = self._request_with_deadline(query, deadline_s)
        if response.get("ok") is not True:
            raise RuntimeError(f"StarVLA inference failed: {response.get('error', response)}")
        actions = np.asarray(response["data"]["actions"])
        if actions.shape != (1, self.chunk_size, 7):
            raise ValueError(f"StarVLA returned {actions.shape}; expected (1, {self.chunk_size}, 7)")
        # Server restores the original dataset units: normalized OSC commands, not meters.
        return validated_actions(actions[0])

    def close(self):
        self.client.close()


def make_policy(*, host="127.0.0.1", port=10093, inference_timeout_s=None):
    """Factory for robosuite.scripts.shakebench_evaluate --policy."""

    return StarVLAPolicy(host=host, port=port, deadline_s=inference_timeout_s)


__all__ = [
    "ACTION_NAMES",
    "CAMERAS",
    "STATE_NAMES",
    "TASK",
    "StarVLAEnvironment",
    "StarVLAObservation",
    "StarVLAPolicy",
    "make_policy",
    "modality_metadata",
    "observation_features",
    "rollout_starvla",
    "task_description",
    "validated_actions",
]
