"""Small CPU/MJWarp and optional EGL interface check; never starts collection/training.

Run from an isolated repository with that repository first on PYTHONPATH.
CUDA_VISIBLE_DEVICES must select a freshly checked, exclusively locked free GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", choices=("official", "development"), required=True)
    p.add_argument("--backend", choices=("cpu", "mjwarp"), default="cpu")
    p.add_argument("--egl", action="store_true")
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.variant == "official" and args.backend != "cpu":
        p.error("the official package uses CPU MuJoCo")
    if not 1 <= args.steps <= 20:
        p.error("this smoke check is limited to 1-20 steps")
    import mujoco
    import robosuite

    import shakebench
    from shakebench import models

    area = "physics" if args.variant == "official" else "utils"
    calibration = importlib.import_module(f"shakebench.{area}.calibration")
    scenarios = importlib.import_module(f"shakebench.{area}.vibration_scenarios")
    runtime = importlib.import_module(
        "shakebench.tasks.runtime" if args.variant == "official" else "shakebench.utils.task_runtime"
    )
    states = json.loads(Path(models.assets_root, "shakebench_task_states_official.json").read_text())["states"]
    state = next(s for s in states if s.get("task", {}).get("object_id") == "apple")
    report = {
        "variant": args.variant,
        "backend": args.backend,
        "steps": args.steps,
        "egl": args.egl,
        "imports": {
            "shakebench": shakebench.__file__,
            "robosuite": robosuite.__file__,
            "runtime": runtime.__file__,
            "scenarios": scenarios.__file__,
        },
        "mujoco_version": mujoco.__version__,
        "state_id": state["state_id"],
        "scenarios": {},
    }
    for key in scenarios.RECOMMENDED:
        env, program = runtime.make_environment(
            state, gamma=1, horizon=args.steps + 1, mode="scenario_lines_v1", mode_params={"scenario": key}
        )
        batch = None
        renderer = None
        try:
            record = {"vibration": calibration.vibration_record(program)}
            if args.backend == "mjwarp":
                from shakebench.utils.mjwarp import MJWarpBatch

                batch = MJWarpBatch([env], [program], device="cuda:0", nconmax=128, njmax=512)
                batch.reset()
            for step in range(args.steps):
                action = np.zeros(env.action_dim)
                if batch is None:
                    obs, _, _, _ = env.step(action)
                else:
                    observations, metrics = batch.step(np.array([action]))
                    assert not metrics["invalid"][0]
                    import mujoco_warp as mjw

                    mjw.get_data_into(env.sim.data._data, env.sim.model._model, batch.data, world_id=0)
                    obs = observations[0]
                timestamps = np.asarray(obs["table_imu_timestamps_s"])
                assert np.all(np.isfinite(timestamps))
                assert np.all(np.diff(timestamps) >= 0)
                age = (step + 1) / 20 - float(timestamps[-1])
                assert -1e-10 <= age <= 0.02, age  # Preserve the existing canonical sensor delivery delay.
                positive_delta = np.diff(timestamps)
                assert np.all(positive_delta[positive_delta > 1e-10] <= 0.01 + 1e-10)
                window = np.asarray(obs["table_imu_window"])
                assert np.all(np.isfinite(window))
            record["time_s"] = float(env.sim.data.time)
            record["imu_last_timestamp_s"] = float(timestamps[-1])
            record["imu_window_shape"] = list(window.shape)
            record["finite_qpos"] = bool(np.isfinite(env.sim.data.qpos).all())
            assert record["finite_qpos"]
            if args.egl:
                renderer = mujoco.Renderer(env.sim.model._model, height=64, width=64)
                renderer.update_scene(env.sim.data._data, camera="agentview")
                frame = renderer.render()
                assert frame.shape == (64, 64, 3) and frame.dtype == np.uint8 and np.ptp(frame) > 20
                record["egl_frame_sha256"] = hashlib.sha256(frame.tobytes()).hexdigest()
            report["scenarios"][key] = record
            print(key, record["time_s"], "passed", flush=True)
        finally:
            if renderer is not None:
                renderer.close()
            env.close()
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
