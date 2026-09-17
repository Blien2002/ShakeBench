"""Evaluate a policy on ShakeBench states with MJWarp device physics and rendering.

Runs with:
    python -m robosuite.scripts.shakebench_evaluate_gpu \\
        --policy robosuite.utils.shakebench_starvla:make_policy \\
        --policy-arg host=127.0.0.1 --policy-arg port=10100 \\
        --states robosuite/models/assets/shakebench_task_states_official_v2.json \\
        --state-ids ID,ID --dataset out/lerobot_gpu_can_knee_100 --output out/eval/gpu/result.json

Why this runner exists: a policy trained on datasets written by shakebench_collect_lerobot_gpu
saw MJWarp-rendered pixels, while shakebench_evaluate renders observations with MuJoCo/EGL.
Those two renderers disagree on brightness and shading, so EGL observations are out of
distribution for a GPU-collected checkpoint. This runner mirrors the GPU collector's device
physics, device rendering, and termination logic, batched over worlds, and records the same
result schema as shakebench_evaluate so downstream tooling keeps working.
"""

from __future__ import annotations

import argparse
import json
import time
from contextlib import ExitStack
from pathlib import Path

import numpy as np

from robosuite.scripts.shakebench_collect_lerobot_gpu import WRIST_CAMERA, resolve_main_camera, state_vector
from robosuite.scripts.shakebench_evaluate import (
    EVALUATION_SCHEMA_ID,
    EVALUATION_SCHEMA_VERSION,
    build_policy,
    load_policy_factory,
    observation_config_from_collection,
    parse_policy_args,
    requested_state_ids,
    summarize,
)
from robosuite.scripts.shakebench_gpu_batch import make_environment
from robosuite.scripts.shakebench_run_oracle import load_state_asset
from robosuite.utils.shakebench_artifacts import write_json_atomic
from robosuite.utils.shakebench_metrics import DEFAULT_SUCCESS_THRESHOLDS
from robosuite.utils.shakebench_outcomes import resolve_termination_cause
from robosuite.utils.shakebench_rollout import (
    ERROR_TAXONOMY,
    PolicyOutputError,
    PolicyTimeoutError,
    _predict as predict_actions,
    action_evidence,
    declared_deadline_s,
    episode_outcome,
    proprioception_metadata,
    task_description,
)
from robosuite.utils.shakebench_train_states import assert_split_disjoint, split_overlap

PHYSICS_PROFILES = ("probe", "official")
OBSERVATION_SOURCE = "cameras_mjwarp"
TIMING = "device_simulation_paused_during_inference"


def observation_identity(main_name, main_camera, width, height):
    """Executable observation identity for one evaluation result."""
    return {
        "source": OBSERVATION_SOURCE,
        "renderer": "mujoco_warp",
        "height": int(height),
        "width": int(width),
        "proprioception": proprioception_metadata(),
        "cameras": {
            "observation.images.main": {"kind": "device_camera", "camera": main_name, "preset": main_camera},
            "observation.images.wrist": {"kind": "device_camera", "camera": WRIST_CAMERA},
        },
    }


def _new_record(state, args, policy_id):
    return {
        "state_id": state.get("state_id"),
        "gamma": args.gamma,
        "horizon_steps": args.horizon_steps,
        "policy_id": policy_id,
        "success": False,
        "episode_validity": None,
        "score_outcome": None,
        "termination_cause": None,
        "terminated": False,
        "truncated": False,
        "invalid_execution_reason": None,
        "reward_sum": 0.0,
        "policy_calls": 0,
        "policy_errors": [],
        "error_taxonomy": ERROR_TAXONOMY,
        "inference_wall_time_s": 0.0,
        "action_horizon": args.action_horizon,
        "inference_timeout_s": args.inference_timeout_s,
        "timing": TIMING,
        "physics_backend": "mujoco_warp",
        "physics_profile": args.physics_profile,
        "_executed": [],
    }


def _predict(policy, observation, inference_timeout_s):
    """Run one policy call, keeping the CPU runner's typed failure taxonomy."""
    started = time.perf_counter()
    try:
        chunk, elapsed = predict_actions(
            policy, observation, inference_timeout_s=inference_timeout_s, policy_deadline_s=declared_deadline_s(policy)
        )
    except PolicyTimeoutError as exc:
        return None, time.perf_counter() - started, "policy_timeout", str(exc)
    except PolicyOutputError as exc:
        return None, time.perf_counter() - started, "policy_output_violation", str(exc)
    except Exception as exc:  # noqa: BLE001 - one adapter failure must not lose the batch
        return None, time.perf_counter() - started, "policy_exception", f"{type(exc).__name__}: {exc}"
    return chunk, elapsed, None, None


def _terminate(record, cause):
    record["termination_cause"] = cause
    record["success"] = cause == "success_latched"
    record["episode_validity"], record["score_outcome"] = episode_outcome(cause)


def run_batch(policy, states, *, args, policy_id):
    """Simulate up to ``args.num_worlds`` states together on the device; one record per world."""
    from robosuite.utils.shakebench_mjwarp import MJWarpBatch

    if not states:
        raise ValueError("at least one state is required")
    if not 1 <= args.action_horizon <= policy.chunk_size:
        raise ValueError("action-horizon must be positive and no greater than the checkpoint chunk size")
    deadline = declared_deadline_s(policy)
    timeout = args.inference_timeout_s
    if timeout is not None:
        if not np.isfinite(timeout) or timeout <= 0:
            raise ValueError("inference-timeout-s must be positive and finite")
        if deadline is None or deadline > timeout:
            raise ValueError("policy must declare deadline_s no greater than inference-timeout-s")
    reset = getattr(policy, "reset", None)
    if callable(reset) and len(states) > 1:
        raise ValueError("policies with reset() require --num-worlds 1 to isolate episode state")
    if callable(reset):
        reset()
    envs, programs = [], []
    videos = ExitStack()
    try:
        for state in states:
            env, program = make_environment(
                state, gamma=args.gamma, horizon=args.horizon_steps, physics_profile=args.physics_profile
            )
            envs.append(env)
            programs.append(program)
        if any(env.control_freq != 20 or env.action_dim != 7 for env in envs):
            raise ValueError("evaluation requires the current 20 Hz, 7D action contract")
        main_names = [resolve_main_camera(env.sim.model._model, args.main_camera) for env in envs]
        main_name = main_names[0]
        identity = observation_identity(main_name, args.main_camera, args.width, args.height)
        batch = MJWarpBatch(envs, programs, device=args.device)
        batch.enable_rendering([main_name, WRIST_CAMERA], resolution=(args.width, args.height))
        observations = batch.reset()
        instructions = [task_description(state)["instruction"] for state in states]
        records = [_new_record(state, args, policy_id) for state in states]
        writers = []
        if getattr(args, "video_dir", None) is not None:
            from robosuite.demos.demo_shakebench_oracle_video import FFmpegVideoWriter

            args.video_dir.mkdir(parents=True, exist_ok=True)
            for state, record in zip(states, records):
                name = str(state["state_id"])
                if Path(name).name != name or name in {".", ".."}:
                    raise ValueError("state ID must be a plain filename for video recording")
                path = args.video_dir / f"{name}.mp4"
                if path.exists():
                    raise FileExistsError(path)
                writer = FFmpegVideoWriter(path, width=2 * args.width, height=args.height, fps=20)
                videos.callback(writer.close)
                writers.append(writer)
                record.update(video=str(path), video_frames=0, video_fps=20)

        def record_frames(rendered, worlds):
            for w in worlds:
                writers[w].append_data(np.concatenate([rendered[main_name][w], rendered[WRIST_CAMERA][w]], axis=1))
                records[w]["video_frames"] += 1

        queues = [[] for _ in states]
        for step in range(args.horizon_steps):
            active = [w for w, record in enumerate(records) if record["termination_cause"] is None]
            if not active:
                break
            refill = [w for w in active if not queues[w]]
            if refill or writers:
                try:
                    rendered = batch.render_rgb()
                except Exception as exc:
                    for w in active:
                        records[w]["invalid_execution_reason"] = f"{type(exc).__name__}: {exc}"
                        _terminate(records[w], "invalid_execution")
                    break
                if writers:
                    record_frames(rendered, active)
                for w in refill:
                    observation = {
                        "observation.images.main": rendered[main_name][w],
                        "observation.images.wrist": rendered[WRIST_CAMERA][w],
                        "observation.state": state_vector(observations[w]),
                        "task": instructions[w],
                    }
                    chunk, elapsed, error_type, message = _predict(policy, observation, args.inference_timeout_s)
                    record = records[w]
                    record["inference_wall_time_s"] += elapsed
                    if error_type is not None:
                        record["policy_errors"].append(
                            {"step": len(record["_executed"]), "error_type": error_type, "message": message}
                        )
                        _terminate(record, "policy_error")
                        continue
                    record["policy_calls"] += 1
                    queues[w] = list(chunk[: args.action_horizon])
            active = [w for w in active if records[w]["termination_cause"] is None]
            if not active:
                break
            actions = np.zeros((len(states), 7), dtype=np.float32)
            for w in active:
                if queues[w]:
                    actions[w] = queues[w].pop(0)
            try:
                observations, metrics = batch.step(actions)
            except Exception as exc:
                for w in active:
                    records[w]["invalid_execution_reason"] = f"{type(exc).__name__}: {exc}"
                    _terminate(records[w], "invalid_execution")
                break
            for w in active:
                record = records[w]
                record["_executed"].append(actions[w].copy())
                cause = None
                if bool(metrics["invalid"][w]):
                    cause = "invalid_execution"
                else:
                    cause = resolve_termination_cause(
                        prior_cause=None,
                        task_rule_violation=bool(
                            metrics["contacts"][w, 0] >= DEFAULT_SUCCESS_THRESHOLDS.max_illegal_penetration_m
                        ),
                        success_latched=bool(metrics["success"][w]),
                        policy_abort=False,
                        horizon_exhausted=step + 1 == args.horizon_steps,
                    )
                if cause is not None:
                    _terminate(record, cause)
            if writers:
                finished = [w for w in active if records[w]["termination_cause"] is not None]
                if finished:
                    record_frames(batch.render_rgb(), finished)
        episodes = []
        for world, record in enumerate(records):
            if record["termination_cause"] is None:
                _terminate(record, "horizon_exhausted")
            record["reward_sum"] = 1.0 if record["success"] else 0.0
            record["truncated"] = record["termination_cause"] == "horizon_exhausted"
            record["terminated"] = record["termination_cause"] in {
                "success_latched",
                "task_rule_violation",
                "policy_abort",
                "policy_error",
            }
            record["observation_identity"] = identity
            record["task_context"] = envs[world].get_policy_task_context()
            executed = record.pop("_executed")
            record["steps"] = len(executed)
            record.update(action_evidence(executed))
            episodes.append(record)
        return episodes
    finally:
        try:
            videos.close()
        finally:
            for env in envs:
                env.close()


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policy", required=True, help="module:factory returning a predict()-capable policy")
    parser.add_argument("--policy-arg", action="append", default=[], metavar="NAME=VALUE")
    parser.add_argument("--policy-id", default=None, help="Label recorded with the results; defaults to --policy")
    parser.add_argument("--states", type=Path, default=Path("robosuite/models/assets/shakebench_states_dev.json"))
    parser.add_argument("--state-ids", action="append", help="Comma-separated or repeated exact state IDs")
    parser.add_argument("--dataset", type=Path, help="Collected dataset or manifest used for training")
    parser.add_argument(
        "--allow-train-states", action="store_true", help="Permit evaluating states that trained the policy"
    )
    parser.add_argument("--gamma", type=float, default=0.0)
    parser.add_argument("--horizon-steps", type=int, default=600)
    parser.add_argument("--action-horizon", type=int, default=8)
    parser.add_argument("--inference-timeout-s", type=float, default=None)
    parser.add_argument("--num-worlds", type=int, default=4, help="States simulated and rendered together per batch")
    parser.add_argument("--width", type=int, default=None, help="Defaults to the training dataset, else 256")
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--main-camera", default=None, help="Defaults to the training dataset, else task_close")
    parser.add_argument("--device", default="cuda:0", help="cuda:N; cpu is available for debugging")
    parser.add_argument("--physics-profile", choices=PHYSICS_PROFILES, default="official")
    parser.add_argument("--output", type=Path, required=True, help="New JSON result path; never overwritten")
    parser.add_argument("--video-dir", type=Path, help="Record main and wrist MJWarp views side by side at 20 fps")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if min(args.horizon_steps, args.action_horizon, args.num_worlds) < 1:
        raise ValueError("horizon-steps, action-horizon and num-worlds must be positive")

    states = requested_state_ids(load_state_asset(args.states)["states"], args.state_ids)
    if not states:
        raise ValueError("no states selected")
    camera_config = {
        "main_camera": args.main_camera or "task_close",
        "height": args.height or 256,
        "width": args.width or 256,
    }
    train_split = {"status": "unavailable", "reason": "no --dataset was given"}
    train_states = []
    if args.dataset is not None:
        trained_on = observation_config_from_collection(args.dataset)
        train_states = trained_on.pop("train_states")
        camera_config = {
            "main_camera": args.main_camera or trained_on["main_camera"],
            "height": args.height or trained_on["height"],
            "width": args.width or trained_on["width"],
        }
        train_split = {"status": "checked", **trained_on, "train_state_count": len(train_states)}
    if train_states and not args.allow_train_states:
        assert_split_disjoint(train_states, states)
        train_split["overlap"] = 0
    elif train_states:
        train_split["overlap"] = len(split_overlap(train_states, states))
        train_split["status"] = "train_states_allowed"

    args.main_camera = camera_config["main_camera"]
    args.height = camera_config["height"]
    args.width = camera_config["width"]
    policy_id = args.policy_id or args.policy
    policy = build_policy(
        load_policy_factory(args.policy),
        arguments=parse_policy_args(args.policy_arg),
        inference_timeout_s=args.inference_timeout_s,
    )
    episodes = []

    def result_payload():
        return {
            "schema_id": EVALUATION_SCHEMA_ID,
            "schema_version": EVALUATION_SCHEMA_VERSION,
            "scoreable": False,
            "scoreable_reason": "policy evaluation evidence; official scorecards come from shakebench_run_oracle",
            "physics_backend": "mujoco_warp",
            "policy": {
                "spec": args.policy,
                "policy_id": policy_id,
                "identity": policy.identity if isinstance(getattr(policy, "identity", None), dict) else None,
                "chunk_size": int(policy.chunk_size),
            },
            "run_contract": {
                "gamma": args.gamma,
                "horizon_steps": args.horizon_steps,
                "action_horizon": args.action_horizon,
                "inference_timeout_s": args.inference_timeout_s,
                "observation_source": OBSERVATION_SOURCE,
                "observation": {
                    "source": OBSERVATION_SOURCE,
                    **camera_config,
                    "identity": episodes[0].get("observation_identity") if episodes else None,
                },
                "action_space": (episodes[0].get("task_context") or {}).get("action_space") if episodes else None,
                "timing": TIMING,
                "device": str(args.device),
                "num_worlds": args.num_worlds,
                "physics_profile": args.physics_profile,
                "state_asset": str(args.states),
            },
            "train_split_check": train_split,
            "episodes": episodes,
            "summary": summarize(episodes),
        }

    def save_results():
        write_json_atomic(args.output, result_payload())

    grouped = {}
    for state in states:
        grouped.setdefault(json.dumps(state.get("task"), sort_keys=True), []).append(state)
    try:
        for group in grouped.values():
            for begin in range(0, len(group), args.num_worlds):
                episodes.extend(
                    run_batch(policy, group[begin : begin + args.num_worlds], args=args, policy_id=policy_id)
                )
                save_results()
                print(
                    f"[gpu] {len(episodes)}/{len(states)} episodes "
                    f"(last: {episodes[-1]['state_id']} {episodes[-1]['termination_cause']} "
                    f"steps={episodes[-1]['steps']})",
                    flush=True,
                )
    finally:
        close = getattr(policy, "close", None)
        if callable(close):
            close()
        save_results()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
