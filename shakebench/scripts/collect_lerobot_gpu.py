"""Collect live gamma=0 oracle demonstrations on MJWarp: device physics and device rendering.

Runs with: python -m shakebench.scripts.collect_lerobot_gpu --output out/lerobot_gpu

Same LeRobot v2.1 schema as shakebench_collect_lerobot, but no CPU rollout and no EGL: the
compiled MJCF is only built on the host to seed the device model. The artifacts stay
non-scoreable (see docs/mjwarp_collection.md); the CPU collector remains the scoreable path.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np

from robosuite.utils import transform_utils as T
from shakebench import models
from shakebench.demos.demo_oracle_video import TASK_CLOSE_CAMERA, task_close_camera_pose
from shakebench.scripts.collect_lerobot import dataset_features
from shakebench.scripts.export_sft_subset import sft_subset_summary
from shakebench.scripts.gpu_batch import make_environment
from shakebench.scripts.run_oracle import _json_ready, load_state_asset
from shakebench.utils.artifacts import write_json_atomic
from shakebench.utils.calibration import vibration_record
from shakebench.utils.oracle import OracleControllerProfile, ShakeBenchOracleController, WorktableTaskContext
from shakebench.utils.outcomes import resolve_termination_cause
from shakebench.utils.rollout import CAMERAS, proprioception_metadata, task_description
from shakebench.utils.websocket_policy import modality_metadata

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


def _frame(observation, action, rendered, *, main_name, world):
    """Build one pre-step frame and copy only this world's rendered pixels."""
    return {
        "action": np.asarray(action, dtype=np.float32).copy(),
        "observation.images.main": rendered[main_name][world].copy(),
        "observation.images.wrist": rendered[WRIST_CAMERA][world].copy(),
        "observation.state": state_vector(observation),
        "observation.table_imu_window": np.asarray(observation["table_imu_window"], dtype=np.float32),
        "observation.table_imu_timestamps_s": np.asarray(observation["table_imu_timestamps_s"], dtype=np.float64),
        "observation.table_imu_dt_s": np.atleast_1d(np.asarray(observation["table_imu_dt_s"], dtype=np.float32)),
    }


def collect_batch(dataset, states, *, horizon, width, height, device, physics_profile, main_camera, on_saved=None):
    """Run and save one homogeneous batch; one LeRobot episode is emitted per world.

    ``on_saved`` receives each episode's metadata immediately after the dataset write, so a
    later world's failure cannot leave earlier saved episodes out of the manifest.
    """
    from shakebench.utils.mjwarp import MJWarpBatch

    if not states:
        raise ValueError("at least one state is required")
    profile = OracleControllerProfile()
    envs, programs = [], []
    try:
        for state in states:
            env, program = make_environment(state, gamma=0.0, horizon=horizon, physics_profile=physics_profile)
            envs.append(env)
            programs.append(program)
        if any(env.control_freq != dataset.fps or env.action_dim != 7 for env in envs):
            raise ValueError("dataset must match the current 20 Hz, 7D oracle contract")
        main_names = [resolve_main_camera(env.sim.model._model, main_camera) for env in envs]
        main_name = main_names[0]
        batch = MJWarpBatch(envs, programs, device=device)
        batch.enable_rendering([main_name, WRIST_CAMERA], resolution=(width, height))
        observations = batch.reset()
        controllers = [
            ShakeBenchOracleController(
                profile, task_context=WorktableTaskContext.from_mapping(env.get_policy_task_context()["task_context"])
            )
            for env in envs
        ]
        causes = [None] * len(states)
        frames = [[] for _ in states]
        for step in range(horizon):
            active = [world for world, cause in enumerate(causes) if cause is None]
            if not active:
                break
            actions = np.zeros((len(states), 7), dtype=np.float32)
            for world in active:
                sample = programs[world].evaluate(step / dataset.fps)
                if any(np.any(value != 0) for value in (sample.q, sample.qdot, sample.qdd)):
                    raise ValueError("gamma=0 must command zero external excitation")
                action = np.clip(controllers[world].action(observations[world], time_s=step / dataset.fps), -1, 1)
                if action.shape != (7,) or not np.isfinite(action).all():
                    raise ValueError("oracle produced an invalid action")
                actions[world] = action
            rendered = batch.render_rgb()
            pre_step_frames = {
                world: _frame(observations[world], actions[world], rendered, main_name=main_name, world=world)
                for world in active
            }
            # Every world advances together; inactive tails receive a zero action and are not recorded.
            observations, metrics = batch.step(actions)
            for world in active:
                cause = (
                    "invalid_execution"
                    if metrics["invalid"][world]
                    else resolve_termination_cause(
                        prior_cause=None,
                        task_rule_violation=False,
                        success_latched=bool(metrics["success"][world]),
                        policy_abort=controllers[world].abort_requested,
                        horizon_exhausted=step + 1 == horizon,
                    )
                )
                frame = pre_step_frames[world]
                frame["next.reward"] = np.array([1.0 if cause == "success_latched" else 0.0], dtype=np.float32)
                frame["next.done"] = np.array([cause is not None], dtype=bool)
                frame["next.success"] = np.array([cause == "success_latched"], dtype=bool)
                frames[world].append(frame)
                causes[world] = cause
        episodes = []
        for world, (state, env, program) in enumerate(zip(states, envs, programs)):
            cause = causes[world] or "horizon_exhausted"
            instruction = task_description(state)["instruction"]
            for step, frame in enumerate(frames[world]):
                dataset.add_frame(frame, task=instruction, timestamp=step / dataset.fps)
            dataset.save_episode()
            # PNG bytes are embedded by the official writer; temporary image files are redundant.
            shutil.rmtree(dataset.root / "images", ignore_errors=True)
            episode = _json_ready(
                {
                    "episode_index": dataset.num_episodes - 1,
                    "state": state,
                    "instruction": instruction,
                    "steps": len(frames[world]),
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
            if on_saved is not None:
                on_saved(episode)
            episodes.append(episode)
        return episodes
    finally:
        for env in envs:
            env.close()


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True, help="New local dataset directory; never overwritten")
    parser.add_argument("--resume", action="store_true", help="Resume an incomplete dataset in --output")
    parser.add_argument("--repo-id", default="shakebench/oracle-gamma-zero-gpu", help="Local dataset ID; no upload")
    parser.add_argument("--states", type=Path, default=Path(models.assets_root, "shakebench_states_dev.json"))
    parser.add_argument("--state-id", action="append", help="Select state IDs; default: all in the asset")
    parser.add_argument("--limit", type=int, default=None, help="Per-shard episode budget")
    parser.add_argument("--num-worlds", type=int, default=4, help="States simulated and rendered together per batch")
    parser.add_argument("--num-shards", type=int, default=1, help="Number of independent state-list shards")
    parser.add_argument("--shard-index", type=int, default=0, help="Zero-based shard selected by this process")
    parser.add_argument("--horizon-steps", type=int, default=1200)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--main-camera", default=TASK_CLOSE_CAMERA, help="task_close preset or a compiled camera")
    parser.add_argument("--device", default="cuda:0", help="cuda:N; cpu is available for debugging")
    parser.add_argument("--image-writer-processes", type=int, default=0)
    parser.add_argument("--image-writer-threads", type=int, default=4)
    parser.add_argument(
        "--physics-profile",
        choices=PHYSICS_PROFILES,
        default="official",
        help="official applies device-side contact calibration; probe is the legacy exploratory profile",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if (
        min(args.width, args.height, args.horizon_steps, args.num_worlds, args.num_shards) <= 0
        or (args.limit is not None and args.limit <= 0)
        or args.shard_index < 0
        or args.shard_index >= args.num_shards
        or args.image_writer_processes < 0
        or args.image_writer_threads <= 0
    ):
        raise ValueError("dimensions, batch size, shard values, limit and image writer settings are invalid")
    state_asset = load_state_asset(args.states)
    states = state_asset["states"]
    if args.state_id:
        unknown = set(args.state_id) - {state["state_id"] for state in states}
        if unknown:
            raise ValueError(f"unknown state IDs: {sorted(unknown)}")
        states = [state for state in states if state["state_id"] in args.state_id]
    states = states[args.shard_index :: args.num_shards]
    if args.limit is not None and args.limit > len(states):
        raise ValueError(f"--limit {args.limit} exceeds the {len(states)} states assigned to this shard")
    if args.limit is not None:
        states = states[: args.limit]
    if not states:
        raise ValueError("no states selected for this shard")
    manifest_path = args.output / "meta" / "shakebench_collection.json"
    manifest = None
    if args.output.exists():
        if not args.resume:
            raise FileExistsError(f"refusing to overwrite {args.output}; pass --resume for an incomplete dataset")
        if not manifest_path.is_file():
            raise ValueError(f"cannot resume without {manifest_path}")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("complete"):
            raise ValueError(f"dataset is already complete: {args.output}")
        requested = [state["state_id"] for state in states]
        if manifest.get("requested_states") != requested:
            raise ValueError("resume states do not match the original collection")
        completed = [episode["state"]["state_id"] for episode in manifest.get("episodes", [])]
        if len(completed) != len(set(completed)):
            raise ValueError("resume manifest contains duplicate episodes")
        states = [state for state in states if state["state_id"] not in set(completed)]
    from lerobot.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset

    if CODEBASE_VERSION != "v2.1":
        raise RuntimeError("LeRobot v2.1 writer required: install requirements-collection.txt")
    if manifest is None:
        dataset = LeRobotDataset.create(
            repo_id=args.repo_id,
            root=args.output,
            fps=20,
            robot_type="Panda",
            features=dataset_features(args.height, args.width),
            use_videos=False,
            image_writer_processes=args.image_writer_processes,
            image_writer_threads=args.image_writer_threads,
        )
        manifest = {
            "complete": False,
            "scoreable": False,
            "gamma": 0.0,
            "physics_backend": "mujoco_warp",
            "physics_profile": args.physics_profile,
            "device": str(args.device),
            "geometry_profile": "world_fixed_arm_v1",
            "tasks": {state["state_id"]: task_description(state)["instruction"] for state in states},
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
            "batch_size": args.num_worlds,
            "shard": {"index": args.shard_index, "num_shards": args.num_shards},
            "image_writer": {"processes": args.image_writer_processes, "threads": args.image_writer_threads},
            "episodes": [],
        }
        write_json_atomic(args.output / "meta" / "modality.json", modality_metadata())
        write_json_atomic(manifest_path, manifest)
    else:
        dataset = LeRobotDataset(args.repo_id, root=args.output, download_videos=False)
        expected_features = dataset_features(args.height, args.width)
        same_features = set(expected_features) <= set(dataset.features) and all(
            dataset.features[key]["dtype"] == expected_features[key]["dtype"]
            and tuple(dataset.features[key]["shape"]) == tuple(expected_features[key]["shape"])
            and dataset.features[key]["names"] == expected_features[key]["names"]
            for key in expected_features
        )
        if dataset.fps != 20 or not same_features:
            raise ValueError("resume dataset format does not match the requested collection")
        if dataset.num_episodes != len(manifest["episodes"]):
            raise ValueError("resume metadata and collection manifest disagree")
        dataset.start_image_writer(args.image_writer_processes, args.image_writer_threads)
        manifest.pop("error", None)
        shutil.rmtree(args.output / "images", ignore_errors=True)
        write_json_atomic(manifest_path, manifest)
    grouped = defaultdict(list)
    for state in states:
        grouped[json.dumps(state.get("task"), sort_keys=True)].append(state)

    def record_episode(episode):
        manifest["episodes"].append(episode)
        write_json_atomic(manifest_path, manifest)
        print(
            f"{episode['state']['state_id']}: {episode['steps']} steps, {episode['termination_cause']}",
            flush=True,
        )

    try:
        for group in grouped.values():
            for begin in range(0, len(group), args.num_worlds):
                collect_batch(
                    dataset,
                    group[begin : begin + args.num_worlds],
                    horizon=args.horizon_steps,
                    width=args.width,
                    height=args.height,
                    device=str(args.device),
                    physics_profile=args.physics_profile,
                    main_camera=args.main_camera,
                    on_saved=record_episode,
                )
        manifest["complete"] = True
    except BaseException as exc:
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        dataset.stop_image_writer()
        manifest["sft_subset"] = sft_subset_summary(manifest["episodes"])
        write_json_atomic(manifest_path, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
