"""Live integration check: run with collection dependencies and MUJOCO_GL=egl."""

import json
import os

import numpy as np
import pytest

pytest.importorskip("lerobot.datasets.lerobot_dataset")
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from robosuite.scripts.shakebench_collect_lerobot import CAMERAS, TASK, main
from robosuite.scripts.shakebench_gpu_batch import make_environment
from robosuite.scripts.shakebench_run_oracle import load_dev_states
from robosuite.utils.shakebench_expert import oracle_observation
from robosuite.utils.shakebench_oracle import OracleControllerProfile, ShakeBenchOracleController, WorktableTaskContext


@pytest.mark.skipif(os.environ.get("MUJOCO_GL") not in {"egl", "osmesa"}, reason="headless rendering required")
def test_live_rollout_v21_roundtrip(tmp_path):
    root = tmp_path / "dataset"
    argv = ["--output", str(root), "--limit", "2", "--horizon-steps", "2", "--width", "64", "--height", "64"]
    assert main(argv) == 0
    dataset = LeRobotDataset("shakebench/oracle-gamma-zero", root=root)
    assert dataset.meta.info["codebase_version"] == "v2.1"
    assert (root / "meta/modality.json").is_file()
    assert dataset.num_episodes == 2 and len(dataset) == 4
    assert len(dataset.meta.episodes_stats) == 2
    manifest = json.loads((root / "meta/shakebench_collection.json").read_text())
    assert manifest["complete"] and manifest["gamma"] == 0
    assert all(ep["termination_cause"] == "horizon_exhausted" for ep in manifest["episodes"])
    assert not (root / "images").exists()  # Images must remain readable after scratch PNG removal.
    states = load_dev_states("robosuite/models/assets/shakebench_states_dev.json")
    for episode_index in range(2):
        env, program = make_environment(states[episode_index], gamma=0, horizon=2)
        try:
            controller = ShakeBenchOracleController(
                OracleControllerProfile(),
                task_context=WorktableTaskContext.from_mapping(env.get_policy_task_context()["task_context"]),
            )
            observation = env._get_observations()
            for step in range(2):
                row = dataset[episode_index * 2 + step]
                assert row["task"] == TASK
                assert row["observation.state"].shape == (8,)
                assert np.isfinite(row["observation.state"]).all()
                assert row["episode_index"].item() == episode_index
                assert row["index"].item() == episode_index * 2 + step
                assert row["frame_index"].item() == step
                assert row["timestamp"].item() == pytest.approx(step / 20)
                assert row["next.done"].item() == (step == 1)
                assert not row["next.success"].item()
                for key in CAMERAS:
                    image = row[key].numpy()
                    assert image.shape == (3, 64, 64) and np.ptp(image) > 0.1
                assert not np.array_equal(row["observation.images.main"], row["observation.images.wrist"])
                for name in ("table_imu_window", "table_imu_timestamps_s", "table_imu_dt_s"):
                    np.testing.assert_allclose(row[f"observation.{name}"], np.atleast_1d(observation[name]), atol=1e-6)
                expected = np.clip(controller.action(oracle_observation(env), time_s=step / 20), -1, 1)
                np.testing.assert_allclose(row["action"], expected, atol=1e-6)
                np.testing.assert_array_equal(program.evaluate(step / 20).qdd, 0)
                observation = env.step(row["action"].numpy())[0]
        finally:
            env.close()
    info_before = (root / "meta/info.json").read_bytes()
    with pytest.raises(FileExistsError):
        main(argv)
    assert (root / "meta/info.json").read_bytes() == info_before
