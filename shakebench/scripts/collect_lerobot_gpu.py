"""Collect pick-place, ring, Push-T, or Upright Oracle episodes on MJWarp physics with host rendering.

Runs with: python -m shakebench.scripts.collect_lerobot_gpu --output out/lerobot_gpu

LeRobot mode uses the same v2.1 schema and MuJoCo camera path as shakebench_collect_lerobot;
--video-only writes MP4 files without a dataset, while --video-output saves synchronized
rollout MP4s alongside LeRobot data. The rollout runs on the device. Frames use host MuJoCo
rendering because MJWarp's
renderer maps textures only onto plane and mesh geoms, which would flatten every textured
box in this scene. The artifacts stay non-scoreable (see docs/mjwarp_collection.md); the CPU
collector remains the scoreable path.
"""

from __future__ import annotations

import argparse
import importlib
import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np

from shakebench import models
from shakebench.demos.demo_oracle_video import TASK_CLOSE_CAMERA, FFmpegVideoWriter, task_close_camera_pose
from shakebench.scripts.collect_lerobot import dataset_features
from shakebench.scripts.export_sft_subset import sft_subset_summary
from shakebench.scripts.run_oracle import _json_ready, load_state_asset
from shakebench.utils.artifacts import write_json
from shakebench.utils.calibration import vibration_record
from shakebench.utils.geometry import DEFAULT_GEOMETRY_PROFILE
from shakebench.utils.oracle import OracleControllerProfile, ShakeBenchOracleController, WorktableTaskContext
from shakebench.utils.outcomes import resolve_termination_cause
from shakebench.utils.rollout import (
    CAMERAS,
    ShakeBenchCameraObservation,
    proprioception_metadata,
    task_description,
)
from shakebench.utils.task_registry import task_type
from shakebench.utils.task_runtime import make_environment
from shakebench.utils.websocket_policy import modality_metadata

MAIN_CAMERA_HOST = "frontview"  # compiled model camera that carries the main-view pose
WRIST_CAMERA = CAMERAS["observation.images.wrist"]
PHYSICS_PROFILES = ("probe", "official")


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


def _frame(action, images):
    """Build one pre-step frame from this world's host-rendered images."""
    return {
        "action": np.asarray(action, dtype=np.float32).copy(),
        "observation.images.main": np.asarray(images["observation.images.main"]).copy(),
        "observation.images.wrist": np.asarray(images["observation.images.wrist"]).copy(),
        "observation.state": np.asarray(images["observation.state"], dtype=np.float32).copy(),
    }


def collection_task_description(state, *, legacy_metal_table=False):
    """Return current task identity with optional legacy dataset wording."""
    description = task_description(state)
    if legacy_metal_table and task_type(state) == "pick_place":
        object_name = {"apple": "apple", "mug": "mug", "can": "food can"}[state["task"]["object_id"]]
        description["instruction"] = f"Pick up the {object_name} from the metal table and place it in the target crate."
    return description


def _batch_key(state):
    """Group states whose compiled task models match."""
    return json.dumps(
        (state.get("task"), state.get("goal_id") if task_type(state) == "push_t" else None),
        sort_keys=True,
    )


def collect_batch(
    dataset,
    states,
    *,
    horizon,
    width,
    height,
    device,
    physics_profile,
    main_camera,
    on_saved=None,
    video_output=None,
    legacy_task_instructions=False,
):
    """Run one homogeneous batch and save each world to LeRobot or MP4.

    ``on_saved`` receives each episode's metadata immediately after the dataset write, so a
    later world's failure cannot leave earlier saved episodes out of the manifest.
    """
    from shakebench.utils.mjwarp import MJWarpBatch

    if not states:
        raise ValueError("at least one state is required")
    profile = OracleControllerProfile()
    envs, programs, readers, writers = [], [], [], []
    task = task_type(states[0])
    specialized = task in {"ring_on_peg", "push_t", "upright"}
    if any(_batch_key(state) != _batch_key(states[0]) for state in states):
        raise ValueError("one batch must contain one compiled task model")
    try:
        for state in states:
            env, program = make_environment(state, gamma=0.0, horizon=horizon, physics_profile=physics_profile)
            envs.append(env)
            programs.append(program)
        if (dataset is not None and dataset.fps != 20) or any(
            env.control_freq != 20 or env.action_dim != 7 for env in envs
        ):
            raise ValueError("dataset must match the current 20 Hz, 7D oracle contract")
        main_names = [resolve_main_camera(env.sim.model._model, main_camera) for env in envs]
        main_name = main_names[0]
        batch = MJWarpBatch(
            envs,
            programs,
            device=device,
            nconmax=512 if task == "ring_on_peg" else 128,
            njmax=2048 if task == "ring_on_peg" else 512,
        )
        readers.extend(
            ShakeBenchCameraObservation(env, height=height, width=width, main_camera=name, include_imu=False)
            for env, name in zip(envs, main_names)
        )
        observations = batch.reset()
        if specialized:
            # Specialized oracles use host state; frame zero needs one refresh after reset.
            for world, reader in enumerate(readers):
                reader.sync_device_state(batch.data, world)
            if task == "ring_on_peg":
                from shakebench.utils.ring_oracle import RingStackOracle

                controllers = [RingStackOracle(env) for env in envs]
            elif task == "push_t":
                from shakebench.utils.push_t_oracle import PushTOracle

                controllers = [PushTOracle(env) for env in envs]
            else:
                from shakebench.utils.upright_oracle import UprightOracle

                controllers = [UprightOracle(env) for env in envs]
        else:
            controllers = [
                ShakeBenchOracleController(
                    profile,
                    task_context=WorktableTaskContext.from_mapping(env.get_policy_task_context()["task_context"]),
                )
                for env in envs
            ]
        if video_output is not None:
            video_output.mkdir(parents=True, exist_ok=True)
            for state in states:
                state_id = state["state_id"]
                if Path(state_id).name != state_id:
                    raise ValueError("video state_id must be a filename")
                path = video_output / f"{state_id}.mp4"
                if path.exists():
                    raise FileExistsError(path)
                writers.append(FFmpegVideoWriter(path, width=width, height=height, fps=20))
        causes = [None] * len(states)
        frames = [[] for _ in states]
        step_counts = [0] * len(states)
        for step in range(horizon):
            active = [world for world, cause in enumerate(causes) if cause is None]
            if not active:
                break
            actions = np.zeros((len(states), 7), dtype=np.float32)
            for world in active:
                sample = programs[world].evaluate(step / 20)
                if any(np.any(value != 0) for value in (sample.q, sample.qdot, sample.qdd)):
                    raise ValueError("gamma=0 must command zero external excitation")
                action = (
                    controllers[world].action()
                    if specialized
                    else controllers[world].action(observations[world], time_s=step / 20)
                )
                action = np.clip(action, -1, 1)
                if action.shape != (7,) or not np.isfinite(action).all():
                    raise ValueError("oracle produced an invalid action")
                actions[world] = action
            images = {}
            for world in active:
                if not specialized:
                    readers[world].sync_device_state(batch.data, world)
                images[world] = readers[world].read(observations[world])
            pre_step_frames = (
                {world: _frame(actions[world], images[world]) for world in active} if dataset is not None else {}
            )
            # Every world advances together; inactive tails receive a zero action and are not recorded.
            observations, metrics = batch.step(actions)
            for world in active:
                cause = (
                    "invalid_execution"
                    if metrics["invalid"][world]
                    else resolve_termination_cause(
                        prior_cause=None,
                        task_rule_violation=bool(metrics["task_rule_violation"][world]) if task == "push_t" else False,
                        success_latched=bool(metrics["success"][world])
                        and (task not in {"push_t", "upright"} or controllers[world].verified),
                        policy_abort=controllers[world].abort_requested,
                        horizon_exhausted=step + 1 == horizon,
                    )
                )
                if dataset is not None:
                    frame = pre_step_frames[world]
                    frame["next.reward"] = np.array([1.0 if cause == "success_latched" else 0.0], dtype=np.float32)
                    frame["next.done"] = np.array([cause is not None], dtype=bool)
                    frame["next.success"] = np.array([cause == "success_latched"], dtype=bool)
                    frames[world].append(frame)
                if video_output is not None:
                    writers[world].append_data(images[world]["observation.images.main"])
                step_counts[world] += 1
                causes[world] = cause
        episodes = []
        for world, (state, env, program) in enumerate(zip(states, envs, programs)):
            cause = causes[world] or "horizon_exhausted"
            instruction = collection_task_description(state, legacy_metal_table=legacy_task_instructions)["instruction"]
            if dataset is not None:
                for step, frame in enumerate(frames[world]):
                    dataset.add_frame(frame, task=instruction, timestamp=step / 20)
                dataset.save_episode()
                # PNG bytes are embedded by the official writer; temporary image files are redundant.
                shutil.rmtree(dataset.root / "images", ignore_errors=True)
            episode = _json_ready(
                {
                    "episode_index": dataset.num_episodes - 1 if dataset is not None else world,
                    "state": state,
                    "instruction": instruction,
                    "steps": step_counts[world],
                    "success": cause == "success_latched",
                    "termination_cause": cause,
                    "vibration": vibration_record(program),
                    "task_context": env.get_policy_task_context(),
                    "controller_profile": (
                        {"controller": "ring_stack_oracle"}
                        if task == "ring_on_peg"
                        else (
                            {"controller": "push_t_oracle", "pushes": controllers[world].trace}
                            if task == "push_t"
                            else (
                                {
                                    "controller": "upright_oracle",
                                    "failure_reason": controllers[world].failure_reason,
                                    "trace": controllers[world].trace,
                                }
                                if task == "upright"
                                else profile.to_dict()
                            )
                        )
                    ),
                    "physics_backend": "mujoco_warp",
                    "physics_profile": physics_profile,
                    "main_camera": main_name,
                }
            )
            if specialized:
                episode["task_metrics"] = _json_ready(env.get_metrics())
            if on_saved is not None:
                on_saved(episode)
            episodes.append(episode)
        return episodes
    finally:
        for writer in writers:
            writer.close()
        for reader in readers:
            reader.close()
        for env in envs:
            env.close()


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--output", type=Path, required=True, help="New local dataset or video directory; never overwritten"
    )
    parser.add_argument("--video-only", action="store_true", help="Write one MP4 per state, without a LeRobot dataset")
    parser.add_argument(
        "--video-output", type=Path, help="Write synchronized per-state MP4 files alongside LeRobot data"
    )
    parser.add_argument(
        "--legacy-task-instructions", action="store_true", help="Use the previous dataset's metal-table wording"
    )
    parser.add_argument(
        "--task-module", action="append", default=[], help="Import a registered task before loading states"
    )
    parser.add_argument("--resume", action="store_true", help="Resume an incomplete dataset in --output")
    parser.add_argument("--repo-id", default="shakebench/oracle-gamma-zero-gpu", help="Local dataset ID; no upload")
    parser.add_argument("--states", type=Path, default=Path(models.assets_root, "shakebench_states_dev.json"))
    parser.add_argument("--state-id", action="append", help="Select state IDs; default: all in the asset")
    parser.add_argument("--limit", type=int, default=None, help="Per-shard episode budget")
    parser.add_argument("--num-worlds", type=int, default=4, help="States simulated and rendered together per batch")
    parser.add_argument("--num-shards", type=int, default=1, help="Number of independent state-list shards")
    parser.add_argument("--shard-index", type=int, default=0, help="Zero-based shard selected by this process")
    parser.add_argument("--horizon-steps", type=int, default=600)
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
    for module in args.task_module:
        importlib.import_module(module)
    if args.video_only and args.resume:
        raise ValueError("--resume is unavailable for video-only collection")
    if args.video_only and args.video_output is not None:
        raise ValueError("use --output for videos in --video-only mode; --video-output is for dataset mode")
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
    if args.video_only:
        if args.output.exists() and any(args.output.iterdir()):
            raise FileExistsError(f"refusing to overwrite videos in {args.output}")
        args.output.mkdir(parents=True, exist_ok=True)
        grouped = defaultdict(list)
        for state in states:
            grouped[_batch_key(state)].append(state)
        for group in grouped.values():
            for begin in range(0, len(group), args.num_worlds):
                episodes = collect_batch(
                    None,
                    group[begin : begin + args.num_worlds],
                    horizon=args.horizon_steps,
                    width=args.width,
                    height=args.height,
                    device=str(args.device),
                    physics_profile=args.physics_profile,
                    main_camera=args.main_camera,
                    video_output=args.output,
                    legacy_task_instructions=args.legacy_task_instructions,
                )
                for episode in episodes:
                    print(
                        f"{episode['state']['state_id']}: {episode['steps']} steps, {episode['termination_cause']}",
                        flush=True,
                    )
                    if "task_metrics" in episode and episode["termination_cause"] != "success_latched":
                        print(json.dumps(episode["task_metrics"]["success"], sort_keys=True), flush=True)
        return 0
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
        if manifest.get("video_output") != (str(args.video_output) if args.video_output else None):
            raise ValueError("resume video output does not match the original collection")
        if manifest.get("legacy_task_instructions", False) != args.legacy_task_instructions:
            raise ValueError("resume task wording does not match the original collection")
        requested = [state["state_id"] for state in states]
        if manifest.get("requested_states") != requested:
            raise ValueError("resume states do not match the original collection")
        completed = [episode["state"]["state_id"] for episode in manifest.get("episodes", [])]
        if len(completed) != len(set(completed)):
            raise ValueError("resume manifest contains duplicate episodes")
        states = [state for state in states if state["state_id"] not in set(completed)]
    if args.video_output is not None:
        if args.video_output.exists() and any(args.video_output.iterdir()) and manifest is None:
            raise FileExistsError(f"refusing to overwrite videos in {args.video_output}")
        args.video_output.mkdir(parents=True, exist_ok=True)
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
            "imu_enabled": False,
            "scoreable": False,
            "gamma": 0.0,
            "physics_backend": "mujoco_warp",
            "physics_profile": args.physics_profile,
            "device": str(args.device),
            "geometry_profile": DEFAULT_GEOMETRY_PROFILE,
            "tasks": {
                state["state_id"]: collection_task_description(state, legacy_metal_table=args.legacy_task_instructions)[
                    "instruction"
                ]
                for state in states
            },
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
            "video_output": str(args.video_output) if args.video_output else None,
            "legacy_task_instructions": args.legacy_task_instructions,
            "episodes": [],
        }
        write_json(args.output / "meta" / "modality.json", modality_metadata())
        write_json(manifest_path, manifest)
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
        write_json(manifest_path, manifest)
    grouped = defaultdict(list)
    for state in states:
        grouped[_batch_key(state)].append(state)

    def record_episode(episode):
        manifest["episodes"].append(episode)
        # ponytail: the growing manifest is rewritten in place; an interrupt mid-write truncates it.
        write_json(manifest_path, manifest)
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
                    video_output=args.video_output,
                    legacy_task_instructions=args.legacy_task_instructions,
                )
        manifest["complete"] = True
    except BaseException as exc:
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        dataset.stop_image_writer()
        manifest["sft_subset"] = sft_subset_summary(manifest["episodes"])
        write_json(manifest_path, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
