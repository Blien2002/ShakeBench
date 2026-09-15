"""VLA-Adapter bridge protocol: framing, contract checks and the stale-answer regression.

No model weights are loaded; the server side is a stand-in runtime, so this runs in the
ShakeBench environment alone.
"""

import socket
import threading
import time

import numpy as np
import pytest

from robosuite.utils.shakebench_rollout import PolicyTimeoutError
from robosuite.utils.shakebench_vla_adapter import (
    VLAAdapterPolicy,
    _pack_array,
    _serve_connection,
    _unpack_array,
    resolve_unnorm_key,
)


def _observation():
    return {
        "observation.images.main": np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3),
        "observation.images.wrist": np.zeros((4, 4, 3), dtype=np.uint8),
        "observation.state": np.arange(8, dtype=np.float32),
        "task": "Pick up the can from the table and place it in the target tray.",
    }


class _Runtime:
    """Stand-in for a loaded checkpoint; answers after ``delay_s`` with a call-tagged chunk."""

    def __init__(self):
        self.calls = []
        self.delay_s = 0.0
        self.chunk_size = 8
        self.identity = {"checkpoint": "fake", "unnorm_key": "libero_spatial_no_noops"}

    def predict(self, request):
        self.calls.append(request)
        time.sleep(self.delay_s)
        return np.full((self.chunk_size, 7), len(self.calls) / 10, dtype=np.float32)


def _serve_in_background(runtime):
    """A real socket server; every connection gets its own thread so a stalled one cannot block."""
    server = socket.create_server(("127.0.0.1", 0))

    def handle(connection):
        with connection:
            try:
                _serve_connection(connection, runtime)
            except OSError:  # the client dropped a connection whose answer arrived late
                pass

    def accept_forever():
        while True:
            connection, _ = server.accept()
            threading.Thread(target=handle, args=(connection,), daemon=True).start()

    threading.Thread(target=accept_forever, daemon=True).start()
    return server, server.getsockname()[1]


def test_arrays_and_unnorm_keys_round_trip():
    value = np.arange(24, dtype=np.uint8).reshape(2, 4, 3)
    restored = _unpack_array(_pack_array(value))

    assert restored.dtype == value.dtype and restored.shape == value.shape and np.array_equal(restored, value)
    restored[0, 0, 0] = 255  # a writable copy, never a view of the transient socket buffer
    assert value[0, 0, 0] == 0

    assert resolve_unnorm_key({"libero_spatial_no_noops": {}}, "libero_spatial") == "libero_spatial_no_noops"
    assert resolve_unnorm_key({"libero_goal": {}}, "libero_spatial") == "libero_goal"
    with pytest.raises(ValueError, match="un-normalization"):
        resolve_unnorm_key({"a": {}, "b": {}}, "libero_spatial")


def test_client_handshakes_and_sends_the_evaluation_observation():
    runtime = _Runtime()
    server, port = _serve_in_background(runtime)
    try:
        policy = VLAAdapterPolicy(port=port)
        assert policy.chunk_size == 8
        assert policy.identity == {
            "adapter": "vla_adapter_socket",
            "deadline_s": None,
            "checkpoint": "fake",
            "unnorm_key": "libero_spatial_no_noops",
        }
        actions = policy.predict(_observation())
        assert actions.shape == (8, 7) and actions.dtype == np.float32
        assert np.allclose(actions, 0.1)
        request = runtime.calls[0]
        assert request["instruction"].startswith("Pick up the can")
        assert np.array_equal(_unpack_array(request["state"]), np.arange(8, dtype=np.float32))
        assert _unpack_array(request["full_image"]).shape == (4, 4, 3)
        policy.close()
    finally:
        server.close()


def test_a_timed_out_request_never_answers_the_next_episode():
    """Regression: the late response used to stay readable on the reused connection."""
    runtime = _Runtime()
    runtime.delay_s = 0.5
    server, port = _serve_in_background(runtime)
    try:
        policy = VLAAdapterPolicy(port=port, deadline_s=0.2)
        with pytest.raises(PolicyTimeoutError):
            policy.predict(_observation())
        assert policy.connection is None  # the poisoned connection is dropped, not reused
        runtime.delay_s = 0.0
        actions = policy.predict(_observation())
        assert actions.shape == (8, 7)
        assert float(actions[0, 0]) == pytest.approx(0.2, abs=1e-6)
        policy.close()
    finally:
        server.close()
