"""Collect live gamma=0 oracle demonstrations on MJWarp: device physics and device rendering.

Runs with: python -m robosuite.scripts.shakebench_collect_lerobot_gpu --output out/lerobot_gpu

Same LeRobot v2.1 schema as shakebench_collect_lerobot, but no CPU rollout and no EGL: the
compiled MJCF is only built on the host to seed the device model. The artifacts stay
non-scoreable (see docs/mjwarp_collection.md); the CPU collector remains the scoreable path.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np

from robosuite.demos.demo_shakebench_oracle_video import TASK_CLOSE_CAMERA, task_close_camera_pose
from robosuite.scripts.shakebench_collect_lerobot import dataset_features
from robosuite.scripts.shakebench_export_sft_subset import sft_subset_summary
from robosuite.scripts.shakebench_gpu_batch import make_environment
from robosuite.scripts.shakebench_run_oracle import _json_ready, load_state_asset
from robosuite.utils import transform_utils as T
from robosuite.utils.shakebench_artifacts import write_json_atomic
from robosuite.utils.shakebench_calibration import vibration_record
from robosuite.utils.shakebench_metrics import DEFAULT_SUCCESS_THRESHOLDS
from robosuite.utils.shakebench_oracle import OracleControllerProfile, ShakeBenchOracleController, WorktableTaskContext
from robosuite.utils.shakebench_outcomes import resolve_termination_cause
from robosuite.utils.shakebench_rollout import CAMERAS, task_description
from robosuite.utils.shakebench_starvla import modality_metadata

MAIN_CAMERA_HOST = "frontview"  # compiled model camera that carries the main-view pose
WRIST_CAMERA = CAMERAS["observation.images.wrist"]
PHYSICS_PROFILES = ("probe", "official")


def state_vector(observation) -> np.ndarray:
    """8D base-frame proprio, identical to the CPU collector's observation contract."""
    return np.concatenate(
        (
            observation["robot0_eef_pos_robot_base"],
            T.quat2axisangle(np.asarray(observation["robot0_eef_quat_robot_base"], dtype=float)),
            np.asarray(observation["robot0_gripper_state"], dtype=float)[:2],
        )
    ).astype(np.float32)


def resolve_main_camera(model, requested) -> str:
    """Return the model camera to render, stamping the shared task pose onto a host camera."""
    if requested == TASK_CLOSE_CAMERA:
        position, quat, _ = task_close_camera_pose()
        camera_id = model.camera(MAIN_CAMERA_HOST).id
        model.cam_pos[camera_id] = position
        model.cam_quat[camera_id] = quat
        return MAIN_CAMERA_HOST
    model.camera(requested)
    return requested


def collect_episode(dataset, state, *, horizon, width, height, device, physics_profile, main_camera):
    """Store (observation_t, applied_action_t, outcome_t+1), rendered on the device."""
    from robosuite.utils.shakebench_mjwarp import MJWarpBatch

    profile = OracleControllerProfile()
    instruction = task_description(state)["instruction"]
    env, program = make_environment(state, gamma=0.0, horizon=horizon, physics_profile=physics_profile)
    try:
        if env.control_freq != dataset.fps or env.action_dim != 7:
            raise ValueError("dataset must match the current 20 Hz, 7D oracle contract")
        model = env.sim.model._model
        main_name = resolve_main_camera(model, main_camera)
        batch = MJWarpBatch([env], [program], device=device)
        batch.enable_rendering([main_name, WRIST_CAMERA], resolution=(width, height))
        observations = batch.reset()
        controller = ShakeBenchOracleController(
            profile, task_context=WorktableTaskContext.from_mapping(env.get_policy_task_context()["task_context"])
        )
        cause = None
        for step in range(horizon):
            sample = program.evaluate(step / dataset.fps)
            if any(np.any(value != 0) for value in (sample.q, sample.qdot, sample.qdd)):
                raise ValueError("gamma=0 must command zero external excitation")
            observation = observations[0]
            action = np.clip(controller.action(observation, time_s=step / dataset.fps), -1, 1)
            if action.shape != (7,) or not np.isfinite(action).all():
                raise ValueError("oracle produced an invalid action")
            rendered = batch.render_rgb()
            frame = {
                "action": action.astype(np.float32),
                "observation.images.main": rendered[main_name][0],
                "observation.images.wrist": rendered[WRIST_CAMERA][0],
                "observation.state": state_vector(observation),
                "observation.table_imu_window": np.asarray(observation["table_imu_window"], dtype=np.float32),
                "observation.table_imu_timestamps_s": np.asarray(
                    observation["table_imu_timestamps_s"], dtype=np.float64
                ),
                "observation.table_imu_dt_s": np.atleast_1d(
                    np.asarray(observation["table_imu_dt_s"], dtype=np.float32)
                ),
            }
            # Only executed actions are recorded; images and IMU are pre-step device state.
            observations, metrics = batch.step([action])
            cause = (
                "invalid_execution"
                if metrics["invalid"][0]
                else resolve_termination_cause(
                    prior_cause=None,
                    task_rule_violation=bool(
                        metrics["contacts"][0, 0] >= DEFAULT_SUCCESS_THRESHOLDS.max_illegal_penetration_m
                    ),
                    success_latched=bool(metrics["success"][0]),
                    policy_abort=controller.abort_requested,
                    horizon_exhausted=step + 1 == horizon,
                )
            )
            frame["next.reward"] = np.array([1.0 if cause == "success_latched" else 0.0], dtype=np.float32)
            frame["next.done"] = np.array([cause is not None], dtype=bool)
            frame["next.success"] = np.array([cause == "success_latched"], dtype=bool)
            dataset.add_frame(frame, task=instruction, timestamp=step / dataset.fps)
            if cause is not None:
                break
        dataset.save_episode()
        # PNG bytes are embedded by the official writer; temporary image files are redundant.
        shutil.rmtree(dataset.root / "images", ignore_errors=True)
        return _json_ready(
            {
                "episode_index": dataset.num_episodes - 1,
                "state": state,
                "instruction": instruction,
                "steps": step + 1,
                "success": cause == "success_latched",
                "termination_cause": cause,
                "vibration": vibration_record(program),
                "imu_mount": env._imu_mount_audit,
                "task_context": env.get_policy_task_context(),
                "controller_profile": profile.to_dict(),
                "physics_backend": "mujoco_warp",
                "physics_profile": physics_profile,
                "main_camera": main_name,
            }
        )
    finally:
        env.close()


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True, help="New local dataset directory; never overwritten")
    parser.add_argument("--repo-id", default="shakebench/oracle-gamma-zero-gpu", help="Local dataset ID; no upload")
    parser.add_argument("--states", type=Path, default=Path("robosuite/models/assets/shakebench_states_dev.json"))
    parser.add_argument("--state-id", action="append", help="Select state IDs; default: all in the asset")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--horizon-steps", type=int, default=1200)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--main-camera", default=TASK_CLOSE_CAMERA, help="task_close preset or a compiled camera")
    parser.add_argument("--device", default="cuda:0", help="cuda:N; the CPU fallback is not offered here")
    parser.add_argument(
        "--physics-profile",
        choices=PHYSICS_PROFILES,
        default="probe",
        help="probe uses the legacy 20 ms contacts; official applies device-side contact calibration",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if min(args.width, args.height, args.horizon_steps) <= 0 or (args.limit is not None and args.limit <= 0):
        raise ValueError("dimensions, horizon and limit must be positive")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    state_asset = load_state_asset(args.states)
    states = state_asset["states"]
    if args.state_id:
        unknown = set(args.state_id) - {state["state_id"] for state in states}
        if unknown:
            raise ValueError(f"unknown state IDs: {sorted(unknown)}")
        states = [state for state in states if state["state_id"] in args.state_id]
    if args.limit is not None:
        states = states[: args.limit]
    from lerobot.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset

    if CODEBASE_VERSION != "v2.1":
        raise RuntimeError("LeRobot v2.1 writer required: install requirements-collection.txt")
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        root=args.output,
        fps=20,
        robot_type="Panda",
        features=dataset_features(args.height, args.width),
        use_videos=False,
    )
    instructions = {state["state_id"]: task_description(state)["instruction"] for state in states}
    manifest = {
        "complete": False,
        "scoreable": False,
        "gamma": 0.0,
        "physics_backend": "mujoco_warp",
        "physics_profile": args.physics_profile,
        "device": str(args.device),
        "geometry_profile": "world_fixed_arm_v1",
        "tasks": instructions,
        "cameras": {**CAMERAS, "observation.images.main": args.main_camera},
        "alignment": "observation_t, applied_action_t, next outcome; timestamp is episode-relative seconds",
        "action_space": {
            "controller": "OSC_POSE",
            "frame": "robot_base",
            "normalized_bounds": [-1, 1],
            "translation_scale_m": 0.05,
            "rotation_scale_rad": 0.5,
            "gripper": "-1 open, +1 close",
        },
        "requested_states": [state["state_id"] for state in states],
        "state_authority": state_asset["authority"],
        "episodes": [],
    }
    manifest_path = args.output / "meta" / "shakebench_collection.json"
    write_json_atomic(args.output / "meta" / "modality.json", modality_metadata())
    write_json_atomic(manifest_path, manifest)
    try:
        for state in states:
            episode = collect_episode(
                dataset,
                state,
                horizon=args.horizon_steps,
                width=args.width,
                height=args.height,
                device=str(args.device),
                physics_profile=args.physics_profile,
                main_camera=args.main_camera,
            )
            manifest["episodes"].append(episode)
            write_json_atomic(manifest_path, manifest)
            print(f"{state['state_id']}: {episode['steps']} steps, {episode['termination_cause']}", flush=True)
        manifest["complete"] = True
    except BaseException as exc:
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        manifest["sft_subset"] = sft_subset_summary(manifest["episodes"])
        write_json_atomic(manifest_path, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
