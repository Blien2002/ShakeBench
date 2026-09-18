"""WebSocket policy adapter: wire message layout, normalization key, dataset registry
metadata.

The environment, observation contract, action contract, and rollout live in
robosuite.utils.shakebench_rollout, so any policy implementation can be evaluated
without importing a model-specific module.  The peer speaks msgpack over a WebSocket and
serves get_server_metadata/predict_action; it lives outside this repository.
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
    observation_features,
    task_description,
    validated_actions,
)


def modality_metadata():
    """GR00T-style loader mapping, alongside standard LeRobot v2.1 metadata."""
    return {
        "video": {name: {"original_key": f"observation.images.{name}"} for name in ("main", "wrist")},
        "state": {"proprio": {"original_key": "observation.state", "start": 0, "end": 8}},
        "action": {
            "osc": {"original_key": "action", "start": 0, "end": 6, "absolute": False},
            "gripper": {"original_key": "action", "start": 6, "end": 7, "absolute": True},
        },
        "annotation": {"human.task_description": {"original_key": "task_index"}},
    }


class WebsocketPolicy:
    """Image/language client for the ShakeBench registry, with server-side normalization.

    Train with the supplied ShakeBench registry (include_state=false). Proprioception
    and IMU are recorded in the dataset for other research, but are not model inputs here.
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
        self.host, self.port = host, port
        self.client = None
        self.chunk_size = None
        self._connect()

    def _connect(self):
        """Open a fresh connection and re-check the server contract."""

        from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy

        client = WebsocketClientPolicy(self.host, self.port)
        try:
            meta = client.get_server_metadata()
            if "new_embodiment" not in meta.get("available_unnorm_keys", []):
                raise ValueError("server checkpoint must use the ShakeBench data registry and statistics")
            if meta.get("action_keys") != ["action.osc", "action.gripper"]:
                raise ValueError("server action contract does not match the ShakeBench registry")
            if meta.get("state_keys"):
                raise ValueError("this client requires the image/language ShakeBench registry (include_state=false)")
            chunk_size = int(meta["action_chunk_size"])
            if chunk_size < 1:
                raise ValueError("invalid server action_chunk_size")
        except BaseException:
            client.close()
            raise
        self.client = client
        self.chunk_size = chunk_size

    def _discard_connection(self):
        """Close a connection that may still deliver a late response.

        A timed-out request leaves its answer in flight, so the socket must not
        serve the next episode; the next predict() opens a fresh one.
        """

        client, self.client = self.client, None
        if client is not None:
            client.close()

    @property
    def identity(self) -> dict:
        """Adapter identity recorded next to evaluation results."""
        return {
            "adapter": "websocket_policy",
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
            raise PolicyTimeoutError(f"policy server did not answer within {deadline_s}s") from None
        if isinstance(raw, str):
            raise RuntimeError(f"Error in inference server:\n{raw}")
        return msgpack_numpy.unpackb(raw)

    def predict(self, observation):
        # Match the dataset loader's PIL resize exactly (including interpolation).
        example = {
            "image": [np.asarray(Image.fromarray(observation[key]).resize((224, 224))) for key in CAMERAS],
            "lang": observation["task"],
        }
        query = {"examples": [example], "unnorm_key": "new_embodiment"}
        if self.client is None:
            self._connect()
        deadline_s = getattr(self, "deadline_s", None)
        try:
            if deadline_s is None:
                response = self.client.predict_action(query)
            else:
                response = self._request_with_deadline(query, deadline_s)
        except PolicyTimeoutError:
            # The late answer must never be read as this or the next episode's action.
            self._discard_connection()
            raise
        if response.get("ok") is not True:
            raise RuntimeError(f"policy server inference failed: {response.get('error', response)}")
        actions = np.asarray(response["data"]["actions"])
        if actions.shape != (1, self.chunk_size, 7):
            raise ValueError(f"policy server returned {actions.shape}; expected (1, {self.chunk_size}, 7)")
        # Server restores the original dataset units: normalized OSC commands, not meters.
        return validated_actions(actions[0])

    def close(self):
        self._discard_connection()


def make_policy(*, host="127.0.0.1", port=10093, inference_timeout_s=None):
    """Factory for robosuite.scripts.shakebench_evaluate --policy."""

    return WebsocketPolicy(host=host, port=port, deadline_s=inference_timeout_s)


__all__ = [
    "ACTION_NAMES",
    "CAMERAS",
    "STATE_NAMES",
    "TASK",
    "WebsocketPolicy",
    "make_policy",
    "modality_metadata",
    "observation_features",
    "task_description",
    "validated_actions",
]
