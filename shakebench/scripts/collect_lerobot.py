"""Collect static Oracle demonstrations in LeRobot v2.1 without IMU observations."""

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


def dataset_features(height, width, *, include_imu=False):
    return {
        **observation_features(height, width, include_imu=include_imu),
        "action": {"dtype": "float32", "shape": (7,), "names": ACTION_NAMES},
        "next.reward": {"dtype": "float32", "shape": (1,), "names": None},
        "next.done": {"dtype": "bool", "shape": (1,), "names": None},
        "next.success": {"dtype": "bool", "shape": (1,), "names": None},
    }


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
    specialized_oracle = device == "oracle" and task_type(state) in {"ring_on_peg", "push_t"}
    if device == "oracle" and not specialized_oracle:
        require_pick_place(state, consumer="oracle collection")
    profile = OracleControllerProfile()
    instruction = task_description(state)["instruction"]
    env, program = make_environment(state, gamma=0.0, horizon=horizon, physics_profile=physics_profile)
    reader = None
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
            env._post_integration_refresh_stride = env._control_steps
        if imu_hook is not None:
            env._post_physics_step_hooks = [hook for hook in env._post_physics_step_hooks if hook != imu_hook]
        if env.control_freq != dataset.fps or env.action_dim != 7:
            raise ValueError("dataset must match the current 20 Hz, 7D OSC contract")
        controller = (
            ShakeBenchOracleController(
                profile, task_context=WorktableTaskContext.from_mapping(env.get_policy_task_context()["task_context"])
            )
            if device == "oracle" and not specialized_oracle
            else None
        )
        if specialized_oracle:
            if task_type(state) == "ring_on_peg":
                from shakebench.utils.ring_oracle import RingStackOracle

                controller = RingStackOracle(env)
            else:
                from shakebench.utils.push_t_oracle import PushTOracle

                controller = PushTOracle(env)
        reader = ShakeBenchCameraObservation(
            env, height=height, width=width, main_camera=main_camera, include_imu=False
        )
        observation = env._get_observations()
        for step in range(horizon):
            sample = program.evaluate(step / dataset.fps)
            if any(np.any(value != 0) for value in (sample.q, sample.qdot, sample.qdd)):
                raise ValueError("gamma=0 must command zero external excitation")
            if specialized_oracle:
                action = controller.action()
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
                task_rule_violation=bool(metrics.get("task_rule_violation", False)),
                success_latched=bool(metrics["success"]["passed"])
                and (task_type(state) != "push_t" or controller.verified),
                policy_abort=controller.abort_requested if controller is not None else False,
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
                "task_context": env.get_policy_task_context(),
                "task_metrics": metrics,
                "controller_profile": (
                    (
                        {"controller": "ring_stack_oracle"}
                        if task_type(state) == "ring_on_peg"
                        else (
                            {"controller": "push_t_oracle", "pushes": controller.trace}
                            if task_type(state) == "push_t"
                            else profile.to_dict()
                        )
                    )
                    if controller is not None
                    else None
                ),
                "collector": device,
            }
        )
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
        help="New local Oracle dataset directory",
    )
    parser.add_argument("--repo-id", default=None, help="Local dataset ID; defaults to shakebench/<device>-gamma-zero")
    parser.add_argument("--device", choices=("oracle",), default="oracle")
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
    if (
        type(config["pos_sensitivity"]) not in (int, float)
        or not np.isfinite(config["pos_sensitivity"])
        or config["pos_sensitivity"] <= 0
    ):
        parser.error("configuration pos_sensitivity must be finite and positive")
    if (
        type(config["rot_sensitivity"]) not in (int, float)
        or not np.isfinite(config["rot_sensitivity"])
        or config["rot_sensitivity"] < 0
    ):
        parser.error("configuration rot_sensitivity must be finite and nonnegative")
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
    if not np.isfinite(args.pos_sensitivity) or args.pos_sensitivity <= 0:
        raise ValueError("SpaceMouse position sensitivity must be finite and positive")
    if not np.isfinite(args.rot_sensitivity) or args.rot_sensitivity < 0:
        raise ValueError("SpaceMouse rotation sensitivity must be finite and nonnegative")
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
            if task_type(state) != "ring_on_peg":
                require_pick_place(state, consumer="oracle collection")
    selected_states = states
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    from lerobot.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset

    if CODEBASE_VERSION != "v2.1":
        raise RuntimeError("LeRobot v2.1 writer required: install requirements-collection.txt")
    physics_profile = args.physics_profile or ("teleop" if args.device == "spacemouse" else "official")
    instructions = list(dict.fromkeys(task_description(state)["instruction"] for state in selected_states))
    repo_id = args.repo_id or f"shakebench/{args.device}-gamma-zero"
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=args.output,
        fps=20,
        robot_type="Panda",
        features=dataset_features(args.height, args.width),
        use_videos=False,
    )
    manifest = {
        "complete": False,
        "task": args.task,
        "collector": args.device,
        "scoreable": False,
        "gamma": 0.0,
        "physics_backend": "mujoco_cpu",
        "physics_profile": physics_profile,
        "imu_enabled": False,
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
