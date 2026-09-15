"""VLA-Adapter (OpenVLA-Adapter) bridge for the ShakeBench policy evaluation runner.

The checkpoint runtime and the ShakeBench runtime cannot share one interpreter: VLA-Adapter
pins MuJoCo 3.3, under which the ShakeBench scene refuses to compile ("Can collision source
model hash drifted").  The model therefore runs as a local server in the VLA-Adapter
environment and the evaluation runner talks to it over one length-prefixed msgpack socket.

Model side (VLA-Adapter environment, owns torch/prismatic/tensorflow):

    PYTHONPATH=/path/to/VLA-Adapter python -m robosuite.utils.shakebench_vla_adapter \\
        --checkpoint /path/to/VLA-Adapter/outputs/LIBERO-Spatial-Pro --port 10095

Evaluation side (ShakeBench environment, owns MuJoCo):

    python -m robosuite.scripts.shakebench_evaluate \\
        --policy robosuite.utils.shakebench_vla_adapter:make_policy --policy-arg port=10095 ...

The request layout and the action handling mirror VLA-Adapter's own LIBERO evaluation: two
images plus 8D proprioception and the English instruction, and the checkpoint's gripper
convention (0 = close, 1 = open) binarized and flipped back to robosuite's (-1 = open,
+1 = close).  Weights are used as published; nothing here retrains or fine-tunes them.
"""

from __future__ import annotations

import argparse
import os
import socket
import struct
from types import SimpleNamespace

import msgpack
import numpy as np

from robosuite.utils.shakebench_rollout import PolicyTimeoutError, validated_actions

HANDSHAKE = "handshake"
PREDICT = "predict"
DEFAULT_PORT = 10095
DEFAULT_CONNECT_TIMEOUT_S = 600.0
MAX_CHUNK_SIZE = 8  # prismatic.vla.constants.NUM_ACTIONS_CHUNK for the released checkpoints


def resolve_unnorm_key(norm_stats, requested):
    """Pick the action un-normalization key the checkpoint actually stores.

    VLA-Adapter's own rule: a suite key may only exist with the ``_no_noops`` suffix.  A
    single-suite checkpoint is unambiguous; anything else must be named exactly.
    """
    keys = list(norm_stats)
    if requested in keys:
        return requested
    fallback = f"{requested}_no_noops"
    if fallback in keys:
        return fallback
    if len(keys) == 1:
        return keys[0]
    raise ValueError(f"no action un-normalization key for {requested!r} in {sorted(keys)}")


def _pack_array(value):
    """Arrays cross the socket as bytes plus shape and dtype; no numpy msgpack codec."""
    array = np.ascontiguousarray(value)
    return {"bytes": array.tobytes(), "shape": list(array.shape), "dtype": array.dtype.str}


def _unpack_array(payload):
    dtype = np.dtype(payload["dtype"])
    return np.frombuffer(payload["bytes"], dtype=dtype).reshape(payload["shape"]).copy()


def _send_message(connection, payload):
    body = msgpack.packb(payload, use_bin_type=True)
    connection.sendall(struct.pack(">I", len(body)) + body)


def _recv_exactly(connection, count):
    chunks = []
    while count:
        chunk = connection.recv(count)
        if not chunk:
            raise EOFError("connection closed")
        chunks.append(chunk)
        count -= len(chunk)
    return b"".join(chunks)


def _recv_message(connection):
    (size,) = struct.unpack(">I", _recv_exactly(connection, 4))
    return msgpack.unpackb(_recv_exactly(connection, size), raw=False)


class VLAAdapterPolicy:
    """Client policy: any checkpoint served by this module's server, as a ShakeBench policy."""

    def __init__(self, *, host="127.0.0.1", port=DEFAULT_PORT, deadline_s=None):
        if deadline_s is not None and (
            isinstance(deadline_s, bool)
            or not isinstance(deadline_s, (int, float))
            or not np.isfinite(deadline_s)
            or deadline_s <= 0
        ):
            raise ValueError("deadline_s must be a positive number of seconds")
        self.host, self.port, self.deadline_s = host, int(port), deadline_s
        self.connection = None
        self.chunk_size = None
        self.handshake = {}
        self._connect()

    def _connect(self):
        """Open a fresh connection and re-read the server contract."""
        connection = socket.create_connection(
            (self.host, self.port), timeout=self.deadline_s or DEFAULT_CONNECT_TIMEOUT_S
        )
        connection.settimeout(self.deadline_s)
        try:
            _send_message(connection, {"command": HANDSHAKE})
            reply = _recv_message(connection)
            if reply.get("ok") is not True:
                raise RuntimeError(f"VLA-Adapter server refused the handshake: {reply.get('error', reply)}")
            chunk_size = int(reply["chunk_size"])
            if not 1 <= chunk_size <= MAX_CHUNK_SIZE:
                raise ValueError(f"server reported an unusable chunk_size {chunk_size}")
        except BaseException:
            connection.close()
            raise
        self.connection = connection
        self.chunk_size = chunk_size
        self.handshake = reply

    def _discard_connection(self):
        """A timed-out request leaves its answer in flight, so the socket must not serve the next episode."""
        connection, self.connection = self.connection, None
        if connection is not None:
            connection.close()

    @property
    def identity(self) -> dict:
        """Adapter identity recorded next to evaluation results."""
        return {"adapter": "vla_adapter_socket", "deadline_s": self.deadline_s, **self.handshake.get("identity", {})}

    def predict(self, observation):
        request = {
            "command": PREDICT,
            "instruction": observation["task"],
            "full_image": _pack_array(observation["observation.images.main"]),
            "wrist_image": _pack_array(observation["observation.images.wrist"]),
            "state": _pack_array(observation["observation.state"]),
        }
        if self.connection is None:
            self._connect()
        try:
            _send_message(self.connection, request)
            reply = _recv_message(self.connection)
        except TimeoutError:
            self._discard_connection()
            raise PolicyTimeoutError(f"VLA-Adapter server did not answer within {self.deadline_s}s") from None
        except (OSError, EOFError) as exc:
            self._discard_connection()
            raise RuntimeError(f"VLA-Adapter connection failed: {exc}") from exc
        if reply.get("ok") is not True:
            raise RuntimeError(f"VLA-Adapter inference failed: {reply.get('error', reply)}")
        actions = _unpack_array(reply["actions"])
        if actions.shape != (self.chunk_size, 7):
            raise ValueError(f"VLA-Adapter returned {actions.shape}; expected ({self.chunk_size}, 7)")
        return validated_actions(actions)

    def close(self):
        self._discard_connection()


def make_policy(*, host="127.0.0.1", port=DEFAULT_PORT, inference_timeout_s=None):
    """Factory for robosuite.scripts.shakebench_evaluate --policy."""
    return VLAAdapterPolicy(host=host, port=port, deadline_s=inference_timeout_s)


def _serve_connection(connection, runtime):
    """Answer requests on one connection until the peer closes it."""
    while True:
        try:
            request = _recv_message(connection)
        except EOFError:
            return
        command = request.get("command")
        if command == HANDSHAKE:
            reply = {"ok": True, "chunk_size": runtime.chunk_size, "identity": runtime.identity}
        elif command == PREDICT:
            try:
                reply = {"ok": True, "actions": _pack_array(runtime.predict(request))}
            except Exception as exc:  # noqa: BLE001 - one bad request must not end the session
                reply = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        else:
            reply = {"ok": False, "error": f"unknown command {command!r}"}
        _send_message(connection, reply)


def serve(runtime, *, host="127.0.0.1", port=DEFAULT_PORT):
    """Serve one runtime until the process is stopped; connections are handled one at a time."""
    with socket.create_server((host, int(port)), reuse_port=False) as listener:
        print(f"vla-adapter policy server listening on {host}:{port} ({runtime.identity})", flush=True)
        while True:
            connection, _ = listener.accept()
            with connection:
                try:
                    _serve_connection(connection, runtime)
                except OSError:  # a client that hit its deadline closed while the answer was in flight
                    continue


def build_runtime(*, checkpoint, chunk_size=MAX_CHUNK_SIZE, proprio_dim=8, seed=7):
    """Load exactly what the checkpoint's own LIBERO evaluation loads, then expose predict().

    Requires the VLA-Adapter repository on PYTHONPATH.  A trained LiDAR/overload variant reuses
    the same loader; only the checkpoint path changes.
    """
    # TensorFlow's lanczos resize runs in this process; without growth the framework reserves
    # the whole card and the policy no longer fits next to it.
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    try:
        from experiments.robot.openvla_utils import (
            get_action_head,
            get_processor,
            get_proprio_projector,
            resize_image_for_policy,
        )
        from experiments.robot.robot_utils import (
            get_action,
            get_image_resize_size,
            get_model,
            invert_gripper_action,
            normalize_gripper_action,
            set_seed_everywhere,
        )
    except ImportError as exc:  # pragma: no cover - depends on the caller's environment
        raise ImportError("the VLA-Adapter repository must be on PYTHONPATH to serve a checkpoint") from exc

    if not 1 <= int(chunk_size) <= MAX_CHUNK_SIZE:
        raise ValueError(f"chunk_size must be within 1..{MAX_CHUNK_SIZE}")
    config = SimpleNamespace(
        model_family="openvla",
        pretrained_checkpoint=str(checkpoint),
        use_l1_regression=True,
        use_minivlm=True,
        use_film=False,
        num_images_in_input=2,
        use_proprio=True,
        center_crop=True,
        num_open_loop_steps=int(chunk_size),
        unnorm_key="",
        load_in_8bit=False,
        load_in_4bit=False,
        use_pro_version=True,
        save_version="vla-adapter",
        task_suite_name="libero_spatial",
    )
    model = get_model(config)
    model.set_version(config.save_version)
    proprio_projector = get_proprio_projector(config, model.llm_dim, proprio_dim=int(proprio_dim))
    action_head = get_action_head(config, model.llm_dim)
    processor = get_processor(config)
    config.unnorm_key = resolve_unnorm_key(model.norm_stats, config.task_suite_name)
    set_seed_everywhere(seed)
    resize_size = get_image_resize_size(config)

    def predict(request):
        observation = {
            "full_image": resize_image_for_policy(_unpack_array(request["full_image"]), resize_size),
            "wrist_image": resize_image_for_policy(_unpack_array(request["wrist_image"]), resize_size),
            "state": np.asarray(_unpack_array(request["state"]), dtype=np.float32),
        }
        actions = get_action(
            config,
            model,
            observation,
            request["instruction"],
            processor=processor,
            action_head=action_head,
            proprio_projector=proprio_projector,
        )
        chunk = [
            invert_gripper_action(normalize_gripper_action(np.asarray(action, dtype=np.float64))) for action in actions
        ]
        return np.asarray(chunk, dtype=np.float32)

    return SimpleNamespace(
        predict=predict,
        chunk_size=int(chunk_size),
        identity={
            "checkpoint": str(checkpoint),
            "unnorm_key": config.unnorm_key,
            "chunk_size": int(chunk_size),
            "proprio_dim": int(proprio_dim),
        },
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="VLA-Adapter checkpoint directory")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--chunk-size", type=int, default=MAX_CHUNK_SIZE)
    parser.add_argument("--proprio-dim", type=int, default=8, help="8 for LIBERO/ShakeBench proprio layout")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)
    runtime = build_runtime(
        checkpoint=args.checkpoint,
        chunk_size=args.chunk_size,
        proprio_dim=args.proprio_dim,
        seed=args.seed,
    )
    serve(runtime, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
