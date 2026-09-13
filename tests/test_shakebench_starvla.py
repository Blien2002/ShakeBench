"""Check StarVLA message mapping, action validation and chunk termination without model weights."""

import json
import os
import runpy
import shutil
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from robosuite.scripts.shakebench_collect_lerobot import dataset_features
from robosuite.utils.shakebench_starvla import (
    CAMERAS,
    StarVLAEnvironment,
    StarVLAPolicy,
    modality_metadata,
    observation_features,
    rollout_starvla,
)


def test_starvla_request_and_schema():
    observation = {key: np.full((4, 4, 3), i, dtype=np.uint8) for i, key in enumerate(CAMERAS)}
    observation["task"] = "Pick up the can."
    requests = []

    def predict(query):
        requests.append(query)
        return {"ok": True, "data": {"actions": np.zeros((1, 8, 7))}}

    policy = object.__new__(StarVLAPolicy)
    policy.client = SimpleNamespace(predict_action=predict)
    policy.chunk_size = 8
    assert policy.predict(observation).shape == (8, 7)
    assert requests[0]["unnorm_key"] == "new_embodiment"
    example = requests[0]["examples"][0]
    assert example["lang"] == observation["task"] and len(example["image"]) == 2
    assert "state" not in example
    assert observation_features(4, 4).items() <= dataset_features(4, 4).items()
    assert modality_metadata()["annotation"]["human.task_description"]["original_key"] == "task_index"
    policy.client.predict_action = lambda _: {"ok": False, "error": "bad checkpoint"}
    with pytest.raises(RuntimeError, match="bad checkpoint"):
        policy.predict(observation)
    for value in (np.zeros((1, 8, 8)), np.full((1, 8, 7), np.nan), np.full((1, 8, 7), 2)):
        policy.client.predict_action = lambda _, value=value: {"ok": True, "data": {"actions": value}}
        with pytest.raises(ValueError):
            policy.predict(observation)


def test_chunk_stops_and_cuda_action_is_converted():
    task = StarVLAEnvironment({"state_id": "test", "object_xy_m": [0, 0]}, horizon=3)
    executed = []
    task.env = SimpleNamespace(
        step=lambda action: (executed.append(action) or {}, 0, False, {}),
        get_metrics=lambda: {"max_illegal_penetration_m": 0, "success": {"passed": False}},
    )
    task._observation = lambda obs: obs
    task.reset = lambda: (setattr(task, "done", False) or {}, {})
    policy = SimpleNamespace(chunk_size=8, predict=lambda _: np.zeros((8, 7)))
    result = rollout_starvla(task, policy, action_horizon=2)
    assert result["truncated"] and result["policy_calls"] == 2 and len(executed) == 3
    torch = pytest.importorskip("torch")
    task.done = False
    action = torch.zeros(7, device="cuda" if torch.cuda.is_available() else "cpu", requires_grad=True)
    task.step(action)
    assert isinstance(executed[-1], np.ndarray)


def test_official_websocket_client():
    server_module = pytest.importorskip("websockets.sync.server")
    msgpack = pytest.importorskip("deployment.model_server.tools.msgpack_numpy")
    requests = []

    def handler(socket):
        socket.send(
            msgpack.packb(
                {
                    "available_unnorm_keys": ["new_embodiment"],
                    "action_chunk_size": 8,
                    "action_keys": ["action.osc", "action.gripper"],
                }
            )
        )
        requests.append(msgpack.unpackb(socket.recv()))
        socket.send(msgpack.packb({"ok": True, "data": {"actions": np.zeros((1, 8, 7))}}))

    with server_module.serve(handler, "127.0.0.1", 0) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        policy = StarVLAPolicy(port=server.socket.getsockname()[1])
        try:
            observation = {key: np.zeros((32, 32, 3), dtype=np.uint8) for key in CAMERAS}
            observation["task"] = "Pick up the can."
            assert policy.predict(observation).shape == (8, 7)
        finally:
            policy.close()
            server.shutdown()
            thread.join(timeout=5)
    assert requests[0]["examples"][0]["image"][0].shape == (224, 224, 3)


def test_official_starvla_loader(tmp_path):
    source = os.environ.get("SHAKEBENCH_TEST_DATASET")
    if not source:
        pytest.skip("set SHAKEBENCH_TEST_DATASET to a newly collected dataset and add StarVLA to PYTHONPATH")
    from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset

    registry_path = (
        Path(__file__).resolve().parents[1] / "integrations/starvla/train_files/data_registry/data_config.py"
    )
    config = runpy.run_path(str(registry_path))["ROBOT_TYPE_CONFIG_MAP"]["shakebench"]
    root = tmp_path / "dataset"
    shutil.copytree(source, root)
    dataset = LeRobotSingleDataset(
        root,
        config.modality_config(),
        config.embodiment_tag,
        transforms=config.transform(),
        data_cfg={"include_state": False, "lerobot_version": "v2.0"},
    )
    sample = dataset[0]
    assert sample["action"].shape == (8, 7)
    assert len(sample["image"]) == 2 and sample["image"][0].size == (224, 224)
    task = json.loads((root / "meta/tasks.jsonl").read_text().splitlines()[0])["task"]
    assert sample["lang"] == task
    assert "state" not in sample
    raw = dataset.get_step_data(int(dataset.trajectory_ids[0]), 0)
    original = {key: raw[key].copy() for key in config.action_keys}
    restored = dataset.transforms.unapply(dataset.transforms(raw))
    for key in config.action_keys:
        np.testing.assert_allclose(restored[key], original[key], atol=1e-6)
