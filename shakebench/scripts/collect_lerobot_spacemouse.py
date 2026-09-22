"""Collect static keyboard/SpaceMouse demonstrations with resumable LeRobot output."""

from __future__ import annotations

import argparse
import importlib
import json
import shutil
from pathlib import Path

import numpy as np

from shakebench import models
from shakebench.scripts.export_sft_subset import sft_subset_summary
from shakebench.scripts.run_oracle import _json_ready, load_state_asset
from shakebench.utils.artifacts import write_json
from shakebench.utils.calibration import vibration_record
from shakebench.utils.expert import oracle_observation
from shakebench.utils.geometry import DEFAULT_GEOMETRY_PROFILE
from shakebench.utils.oracle import OracleControllerProfile, ShakeBenchOracleController, WorktableTaskContext
from shakebench.utils.outcomes import resolve_termination_cause
from shakebench.utils.rollout import (
    ACTION_NAMES,
    CAMERAS,
    ShakeBenchCameraObservation,
    observation_features,
    proprioception_metadata,
    task_description,
)
from shakebench.utils.task_registry import require_pick_place, task_type
from shakebench.utils.task_runtime import make_environment
from shakebench.utils.websocket_policy import modality_metadata


def dataset_features(height, width, *, include_imu=True):
    return {
        **observation_features(height, width, include_imu=include_imu),
        "action": {"dtype": "float32", "shape": (7,), "names": ACTION_NAMES},
        "next.reward": {"dtype": "float32", "shape": (1,), "names": None},
        "next.done": {"dtype": "bool", "shape": (1,), "names": None},
        "next.success": {"dtype": "bool", "shape": (1,), "names": None},
    }


def _load_spacemouse_resume(output, requested_states):
    """Load a resumable SpaceMouse manifest and return it with its prefix length."""
    manifest_path = output / "meta" / "shakebench_collection.json"
    if not manifest_path.is_file():
        raise FileExistsError(f"cannot resume {output}: missing shakebench_collection.json")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot resume {output}: invalid collection manifest") from exc
    if manifest.get("collector") != "spacemouse":
        raise FileExistsError(f"refusing to resume non-SpaceMouse dataset {output}")
    requested_ids = [state["state_id"] for state in requested_states]
    if manifest.get("requested_states") != requested_ids:
        raise ValueError("resume arguments must select the same states as the existing collection")
    episodes = manifest.get("episodes")
    if not isinstance(episodes, list) or len(episodes) > len(requested_states):
        raise ValueError("existing collection manifest has an invalid episode prefix")
    for index, episode in enumerate(episodes):
        if not isinstance(episode, dict) or episode.get("state", {}).get("state_id") != requested_ids[index]:
            raise ValueError("existing collection episodes do not match the requested state order")
    return manifest, len(episodes)


def collect_episode(
    dataset,
    state,
    *,
    horizon,
    width,
    height,
    main_camera="task_close",
    device="oracle",
    physics_profile="official",
    pos_sensitivity=1.0,
    rot_sensitivity=1.0,
):
    """Store (observation_t, applied_action_t, outcome_t+1), with no padded frames."""
    if device == "oracle":
        require_pick_place(state, consumer="oracle collection")
    include_imu = device != "spacemouse"
    profile = OracleControllerProfile()
    instruction = task_description(state)["instruction"]
    env, program = make_environment(state, gamma=0.0, horizon=horizon, physics_profile=physics_profile)
    reader = teleop = None
    try:
        metrics_hook = getattr(env, "_record_post_physics_metrics", None)
        imu_hook = getattr(env, "_update_phase05_provider", None)
        if getattr(getattr(env, "physics_profile", None), "status", None) == "teleop_non_scoreable":

            def control_boundary_metrics(sample_time_s, policy_step=False):
                if (env._physics_step_index + 1) % env._control_steps == 0:
                    metrics_hook(sample_time_s, policy_step)

            if metrics_hook is not None:
                env._post_physics_step_hooks = [
                    control_boundary_metrics if hook == metrics_hook else hook for hook in env._post_physics_step_hooks
                ]
            env._post_integration_refresh_stride = (
                env._control_steps
                if not include_imu
                else int(env.physics_profile.scheduler["post_integration_refresh_stride"])
            )
        if not include_imu and imu_hook is not None:
            env._post_physics_step_hooks = [hook for hook in env._post_physics_step_hooks if hook != imu_hook]
        if env.control_freq != dataset.fps or env.action_dim != 7:
            raise ValueError("dataset must match the current 20 Hz, 7D OSC contract")
        controller = (
            ShakeBenchOracleController(
                profile, task_context=WorktableTaskContext.from_mapping(env.get_policy_task_context()["task_context"])
            )
            if device == "oracle"
            else None
        )
        reader = ShakeBenchCameraObservation(
            env, height=height, width=width, main_camera=main_camera, include_imu=include_imu
        )
        if device == "spacemouse":
            from shakebench.utils.teleop import SpaceMouseTeleop

            teleop = SpaceMouseTeleop(
                env,
                reader,
                pos_sensitivity=pos_sensitivity,
                rot_sensitivity=rot_sensitivity,
            )
        observation = env._get_observations()
        for step in range(horizon):
            sample = program.evaluate(step / dataset.fps)
            if any(np.any(value != 0) for value in (sample.q, sample.qdot, sample.qdd)):
                raise ValueError("gamma=0 must command zero external excitation")
            if teleop is not None:
                action = teleop.action()
                if action is None:
                    image_writer = getattr(dataset, "image_writer", None)
                    if image_writer is not None:
                        image_writer.wait_until_done()
                    dataset.clear_episode_buffer()
                    return None
            else:
                action = np.clip(controller.action(oracle_observation(env), time_s=step / dataset.fps), -1, 1)
            if action.shape != (7,) or not np.isfinite(action).all() or np.any(np.abs(action) > 1):
                raise ValueError("controller produced an invalid action")
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
                task_rule_violation=False,
                success_latched=bool(metrics["success"]["passed"]),
                policy_abort=controller.abort_requested if controller is not None else False,
                horizon_exhausted=step + 1 == horizon,
            )
            frame["next.reward"] = np.array([reward], dtype=np.float32)
            frame["next.done"] = np.array([cause is not None], dtype=bool)
            frame["next.success"] = np.array([cause == "success_latched"], dtype=bool)
            dataset.add_frame(frame, task=instruction, timestamp=step / dataset.fps)
            if teleop is not None:
                teleop.sync(frame)
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
                **(
                    {"imu_mount": env._imu_mount_audit, "imu_contract": env.observation_contract()}
                    if include_imu
                    else {}
                ),
                "task_context": env.get_policy_task_context(),
                "task_metrics": metrics,
                "controller_profile": profile.to_dict() if controller is not None else None,
                "collector": device,
                **({"pos_sensitivity": pos_sensitivity, "rot_sensitivity": rot_sensitivity} if teleop else {}),
            }
        )
    finally:
        try:
            if teleop is not None:
                teleop.close()
        finally:
            try:
                if reader is not None:
                    reader.close()
            finally:
                env.close()


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, help="Explicit task name, e.g. pick_place or ring_on_peg")
    parser.add_argument(
        "--config", type=Path, help="Override the task's collection JSON; relative states resolve beside it"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Local dataset directory; interrupted SpaceMouse collections resume in place",
    )
    parser.add_argument("--repo-id", default=None, help="Local dataset ID; defaults to shakebench/<device>-gamma-zero")
    parser.add_argument("--device", choices=("spacemouse",), default="spacemouse")
    parser.add_argument(
        "--physics-profile",
        choices=("official", "teleop"),
        default=None,
        help="Defaults to teleop for SpaceMouse and official for oracle collection",
    )
    parser.add_argument("--pos-sensitivity", type=float)
    parser.add_argument("--rot-sensitivity", type=float)
    parser.add_argument(
        "--states",
        type=Path,
        help="Verified state asset: frozen dev, committed official/knee, task variants, or a generated train pool",
    )
    parser.add_argument(
        "--task-module", action="append", default=[], help="Import a task registration module before loading states"
    )
    parser.add_argument(
        "--episodes-per-state", type=int, default=1, help="Repeat each selected initial state this many times"
    )
    parser.add_argument("--state-id", action="append", help="Select exact state IDs; default: every state in the asset")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Episode budget; cannot exceed selected states times --episodes-per-state",
    )
    parser.add_argument("--horizon-steps", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--main-camera", help="task_close preset or a compiled camera name")
    return parser


def parse_args(argv=None):
    """Apply explicit task configuration, then command-line overrides."""
    parser = build_parser()
    selected, _ = parser.parse_known_args(argv)
    config_path = selected.config
    if config_path is None:
        if not selected.task.isidentifier():
            parser.error("--task must be a task identifier")
        config_path = Path(models.assets_root, f"collection_{selected.task}.json")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        parser.error(f"cannot load task configuration {config_path}: {exc}")
    required = {
        "states",
        "task_module",
        "horizon_steps",
        "pos_sensitivity",
        "rot_sensitivity",
        "width",
        "height",
        "main_camera",
    }
    if not isinstance(config, dict) or set(config) != required:
        parser.error(f"task configuration must contain exactly: {sorted(required)}")
    if (
        not isinstance(config["states"], str)
        or not isinstance(config["task_module"], list)
        or not all(isinstance(module, str) for module in config["task_module"])
    ):
        parser.error("configuration states must be a path and task_module must be a list of module names")
    for key in ("width", "height", "horizon_steps"):
        if type(config[key]) is not int or config[key] <= 0:
            parser.error(f"configuration {key} must be a positive integer")
    for key in ("pos_sensitivity", "rot_sensitivity"):
        if type(config[key]) not in (int, float) or not np.isfinite(config[key]) or config[key] <= 0:
            parser.error(f"configuration {key} must be finite and positive")
    if not isinstance(config["main_camera"], str) or not config["main_camera"]:
        parser.error("configuration main_camera must be a nonempty name")
    config["states"] = config_path.parent / config["states"]
    parser.set_defaults(**config)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if min(args.width, args.height, args.horizon_steps, args.episodes_per_state) <= 0 or (
        args.limit is not None and args.limit <= 0
    ):
        raise ValueError("dimensions, horizon and limit must be positive")
    if any(not np.isfinite(value) or value <= 0 for value in (args.pos_sensitivity, args.rot_sensitivity)):
        raise ValueError("SpaceMouse sensitivities must be finite and positive")
    for module in args.task_module:
        importlib.import_module(module)
    state_asset = load_state_asset(args.states)
    states = state_asset["states"]
    if args.state_id:
        unknown = set(args.state_id) - {state["state_id"] for state in states}
        if unknown:
            raise ValueError(f"unknown state IDs: {sorted(unknown)}")
        states = [state for state in states if state["state_id"] in args.state_id]
    states = [state for state in states for _ in range(args.episodes_per_state)]
    if args.limit is not None and args.limit > len(states):
        raise ValueError(
            f"--limit {args.limit} exceeds the {len(states)} selected states; "
            "increase --episodes-per-state or select more states"
        )
    states = states[: args.limit]
    if any(task_type(state) != args.task for state in states):
        raise ValueError(f"selected states do not belong to --task {args.task}")
    if args.device == "oracle":
        for state in states:
            require_pick_place(state, consumer="oracle collection")
    selected_states = states
    resume_manifest = None
    completed_count = 0
    existing_dataset = False
    if args.output.exists():
        if args.device != "spacemouse":
            raise FileExistsError(f"refusing to overwrite {args.output}")
        resume_manifest, completed_count = _load_spacemouse_resume(args.output, selected_states)
        if completed_count == len(selected_states):
            resume_manifest["complete"] = True
            resume_manifest.pop("error", None)
            resume_manifest["sft_subset"] = sft_subset_summary(resume_manifest["episodes"])
            write_json(args.output / "meta" / "shakebench_collection.json", resume_manifest)
            print(f"Collection already complete: {args.output}", flush=True)
            return 0
        parquet_files = tuple(args.output.rglob("*.parquet"))
        if parquet_files:
            existing_dataset = True
            shutil.rmtree(args.output / "images", ignore_errors=True)
        elif completed_count:
            raise ValueError("existing manifest lists episodes but the LeRobot dataset has no parquet files")
        else:
            # Only an interrupted first episode can reach this branch. Its scratch
            # images are incomplete and there is no saved data to discard.
            shutil.rmtree(args.output)
        states = states[completed_count:]
    from lerobot.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset

    if CODEBASE_VERSION != "v2.1":
        raise RuntimeError("LeRobot v2.1 writer required: install requirements-collection.txt")
    physics_profile = args.physics_profile or ("teleop" if args.device == "spacemouse" else "official")
    instructions = list(dict.fromkeys(task_description(state)["instruction"] for state in selected_states))
    repo_id = args.repo_id or f"shakebench/{args.device}-gamma-zero"
    if existing_dataset:
        dataset = LeRobotDataset(repo_id, root=args.output)
        dataset.episode_buffer = dataset.create_episode_buffer()
    else:
        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            root=args.output,
            fps=20,
            robot_type="Panda",
            features=dataset_features(args.height, args.width, include_imu=args.device != "spacemouse"),
            use_videos=False,
        )
    if args.device == "spacemouse":
        dataset.start_image_writer(num_processes=0, num_threads=8)
    if resume_manifest is None:
        manifest = {
            "complete": False,
            "task": args.task,
            "collector": args.device,
            "scoreable": False,
            "gamma": 0.0,
            "physics_backend": "mujoco_cpu",
            "physics_profile": physics_profile,
            "imu_enabled": args.device != "spacemouse",
            "geometry_profile": DEFAULT_GEOMETRY_PROFILE,
            "tasks": instructions,
            "task_modules": args.task_module,
            "episodes_per_state": args.episodes_per_state,
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
            "requested_states": [state["state_id"] for state in selected_states],
            "proprioception": proprioception_metadata(),
            "state_authority": state_asset["authority"],
            "episodes": [],
        }
    else:
        manifest = resume_manifest
        manifest["complete"] = False
        manifest.pop("error", None)
    manifest_path = args.output / "meta" / "shakebench_collection.json"
    write_json(args.output / "meta" / "modality.json", modality_metadata())
    write_json(manifest_path, manifest)
    try:
        for state in states:
            episode = None
            while episode is None:
                print(f"Collecting {state['state_id']} ({args.device})", flush=True)
                episode = collect_episode(
                    dataset,
                    state,
                    horizon=args.horizon_steps,
                    width=args.width,
                    height=args.height,
                    main_camera=args.main_camera,
                    device=args.device,
                    physics_profile=physics_profile,
                    pos_sensitivity=args.pos_sensitivity,
                    rot_sensitivity=args.rot_sensitivity,
                )
                if episode is None:
                    print("Discarded attempt; retrying the same state.", flush=True)
            manifest["episodes"].append(episode)
            # ponytail: the growing manifest is rewritten in place; an interrupt mid-write truncates it.
            write_json(manifest_path, manifest)
            print(f"{state['state_id']}: {episode['steps']} steps, {episode['termination_cause']}", flush=True)
        manifest["complete"] = True
    except BaseException as exc:
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        # Training consumes the success-only selection; every attempt stays in
        # this manifest so failures remain auditable.
        manifest["sft_subset"] = sft_subset_summary(manifest["episodes"])
        try:
            write_json(manifest_path, manifest)
        finally:
            dataset.stop_image_writer()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
