"""Collect live gamma=0 oracle demonstrations in LeRobot v2.1 (embedded RGB PNGs)."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np

from robosuite.scripts.shakebench_gpu_batch import make_environment
from robosuite.scripts.shakebench_export_sft_subset import sft_subset_summary
from robosuite.scripts.shakebench_run_oracle import _json_ready, load_state_asset
from robosuite.utils.shakebench_artifacts import write_json_atomic
from robosuite.utils.shakebench_calibration import vibration_record
from robosuite.utils.shakebench_expert import oracle_observation
from robosuite.utils.shakebench_metrics import DEFAULT_SUCCESS_THRESHOLDS
from robosuite.utils.shakebench_oracle import OracleControllerProfile, ShakeBenchOracleController, WorktableTaskContext
from robosuite.utils.shakebench_outcomes import resolve_termination_cause
from robosuite.utils.shakebench_rollout import (
    ACTION_NAMES,
    CAMERAS,
    ShakeBenchCameraObservation,
    observation_features,
    proprioception_metadata,
    task_description,
)
from robosuite.utils.shakebench_starvla import modality_metadata


def dataset_features(height, width):
    return {
        **observation_features(height, width),
        "action": {"dtype": "float32", "shape": (7,), "names": ACTION_NAMES},
        "next.reward": {"dtype": "float32", "shape": (1,), "names": None},
        "next.done": {"dtype": "bool", "shape": (1,), "names": None},
        "next.success": {"dtype": "bool", "shape": (1,), "names": None},
    }


def collect_episode(dataset, state, *, horizon, width, height, main_camera="task_close"):
    """Store (observation_t, applied_action_t, outcome_t+1), with no padded frames."""
    profile = OracleControllerProfile()
    instruction = task_description(state)["instruction"]
    env, program = make_environment(state, gamma=0.0, horizon=horizon)
    reader = None
    try:
        if env.control_freq != dataset.fps or env.action_dim != 7:
            raise ValueError("dataset must match the current 20 Hz, 7D oracle contract")
        controller = ShakeBenchOracleController(
            profile, task_context=WorktableTaskContext.from_mapping(env.get_policy_task_context()["task_context"])
        )
        reader = ShakeBenchCameraObservation(env, height=height, width=width, main_camera=main_camera)
        observation = env._get_observations()
        for step in range(horizon):
            sample = program.evaluate(step / dataset.fps)
            if any(np.any(value != 0) for value in (sample.q, sample.qdot, sample.qdd)):
                raise ValueError("gamma=0 must command zero external excitation")
            action = np.clip(controller.action(oracle_observation(env), time_s=step / dataset.fps), -1, 1)
            if action.shape != (7,) or not np.isfinite(action).all():
                raise ValueError("oracle produced an invalid action")
            visual = reader.read(observation)
            frame = {
                "action": action.astype(np.float32),
                **visual,
            }
            # Record only actions that were actually executed. Images and IMU above are pre-step.
            observation, reward, _, _ = env.step(frame["action"].copy())
            metrics = env.get_metrics()
            cause = resolve_termination_cause(
                prior_cause=None,
                task_rule_violation=metrics["max_illegal_penetration_m"]
                >= DEFAULT_SUCCESS_THRESHOLDS.max_illegal_penetration_m,
                success_latched=bool(metrics["success"]["passed"]),
                policy_abort=controller.abort_requested,
                horizon_exhausted=step + 1 == horizon,
            )
            frame["next.reward"] = np.array([reward], dtype=np.float32)
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
                "imu_contract": env.observation_contract(),
                "task_context": env.get_policy_task_context(),
                "controller_profile": profile.to_dict(),
            }
        )
    finally:
        if reader is not None:
            reader.close()
        env.close()


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New local dataset directory; never overwritten")
    parser.add_argument("--repo-id", default="shakebench/oracle-gamma-zero", help="Local dataset ID; no upload")
    parser.add_argument(
        "--states",
        type=Path,
        default=Path("robosuite/models/assets/shakebench_states_dev.json"),
        help="Verified state asset: frozen dev, committed official/knee, task variants, or a generated train pool",
    )
    parser.add_argument("--state-id", action="append", help="Select exact state IDs; default: every state in the asset")
    parser.add_argument("--limit", type=int, default=None, help="Episode budget; cannot exceed the selected state count")
    parser.add_argument("--horizon-steps", type=int, default=1200)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--main-camera", default="task_close", help="task_close preset or a compiled camera name")
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
    if args.limit is not None and args.limit > len(states):
        raise ValueError(
            f"--limit {args.limit} exceeds the {len(states)} selected states; "
            "generate a larger pool with robosuite.scripts.shakebench_generate_train_states"
        )
    states = states[: args.limit]
    from lerobot.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset

    if CODEBASE_VERSION != "v2.1":
        raise RuntimeError("LeRobot v2.1 writer required: install requirements-collection.txt")
    instructions = list(dict.fromkeys(task_description(state)["instruction"] for state in states))
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        root=args.output,
        fps=20,
        robot_type="Panda",
        features=dataset_features(args.height, args.width),
        use_videos=False,
    )
    manifest = {
        "complete": False,
        "scoreable": False,
        "gamma": 0.0,
        "physics_backend": "mujoco_cpu",
        "physics_profile": "official",
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
        "proprioception": proprioception_metadata(),
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
        # Training consumes the success-only selection; every attempt stays in
        # this manifest so failures remain auditable.
        manifest["sft_subset"] = sft_subset_summary(manifest["episodes"])
        write_json_atomic(manifest_path, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
