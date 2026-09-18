"""Record GPU-rendered oracle rollout videos across Gamma on the current scene.

Frames are rendered through MuJoCo's EGL backend, so this must run on a host
with a usable NVIDIA EGL device (the device nodes are hidden from the Codex
sandbox; run it unsandboxed). Camera pixels never enter the oracle
observation, action, or success metric, and the output is qualitative only.

Example:

    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python -m \
        shakebench.scripts.oracle_gamma_videos \
        --output-dir out/oracle_gamma_videos
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

import shakebench
from robosuite.controllers import load_composite_controller_config
from shakebench import models
from shakebench.demos.demo_oracle_video import VideoObserver
from shakebench.utils.artifacts import write_json
from shakebench.utils.calibration import build_vibration_program
from shakebench.utils.expert import oracle_observation
from shakebench.utils.geometry import geometry_scene_path
from shakebench.utils.oracle import (
    OracleControllerProfile,
    ShakeBenchOracleController,
    ShakeBenchOracleError,
    WorktableTaskContext,
)
from shakebench.utils.outcomes import resolve_termination_cause, validate_outcome
from shakebench.utils.scene import load_scene_visual_config
from shakebench.utils.state_schema import normalize_state

SCHEMA_ID = "shakebench.oracle_gamma_videos"
SCHEMA_VERSION = 1
GEOMETRY_PROFILE = "world_fixed_arm_v1"
TIER = "V0"
DEFAULT_GAMMAS = (0.0, 0.15, 0.30, 0.45, 0.60, 0.75, 0.95)
DEFAULT_STATES = Path(models.assets_root, "shakebench_states_dev.json")
ZERO_GAMMA_SAMPLE_S = (0.0, 0.5, 1.0, 2.0)


def parse_gammas(text):
    gammas = []
    for item in str(text).split(","):
        gamma = float(item)
        if not np.isfinite(gamma) or gamma < 0.0:
            raise argparse.ArgumentTypeError("gammas must be finite and non-negative")
        if gamma not in gammas:
            gammas.append(gamma)
    if not gammas:
        raise argparse.ArgumentTypeError("at least one Gamma is required")
    return tuple(gammas)


def load_state(path, state_id):
    payload = json.loads(Path(path).read_text())
    matches = [state for state in payload["states"] if state["state_id"] == state_id]
    if len(matches) != 1:
        available = ", ".join(state["state_id"] for state in payload["states"])
        raise ValueError(f"unknown state-id {state_id!r}; available: {available}")
    return matches[0]


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record_episode(state, *, gamma, mode, args, output):
    """Run one oracle episode and write its EGL-rendered video."""

    state = normalize_state(state)
    profile = OracleControllerProfile()
    seed = int(state.get("excitation_seed", state.get("seed", 0)))
    t0_s = float(state.get("t0_s", 0.0))
    vibration = {"mode": mode, "gamma": gamma, "seed": seed, "t0_s": t0_s}
    if gamma == 0.0:
        program = build_vibration_program(vibration)
        for sample_time_s in ZERO_GAMMA_SAMPLE_S:
            sample = program.evaluate(sample_time_s)
            if max(np.max(np.abs(sample.q)), np.max(np.abs(sample.qdot)), np.max(np.abs(sample.qdd))) != 0.0:
                raise AssertionError(f"Gamma=0 must command zero excitation, got {sample.q} at t={sample_time_s}s")
    env = shakebench.make(
        "VibrationPickPlace",
        robots="Panda",
        controller_configs=load_composite_controller_config(robot="Panda"),
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        use_object_obs=True,
        physics_profile="official",
        geometry_profile=GEOMETRY_PROFILE,
        vibration=vibration,
        object_start_xy=tuple(state["object_xy_m"]),
        imu_seed=int(state.get("imu_seed", seed)),
        horizon=args.horizon_steps,
        ignore_done=True,
        seed=seed,
        hard_reset=False,
    )
    observer = VideoObserver(
        output,
        camera=args.camera,
        width=args.width,
        height=args.height,
        fps=args.fps,
        tier=TIER,
        gamma=gamma,
        state_id=state["state_id"],
        policy_rate_hz=profile.policy_rate_hz,
        scene_config=load_scene_visual_config(geometry_scene_path(GEOMETRY_PROFILE)),
        raw=args.raw,
    )
    frames = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    try:
        observation = oracle_observation(env, env.reset())
        controller = ShakeBenchOracleController(
            profile, task_context=WorktableTaskContext.from_mapping(env.get_policy_task_context()["task_context"])
        )
        worktable_id = env.sim.model.body_name2id(env.arena.worktable_body_name)
        reference = np.asarray(env.sim.data.xpos[worktable_id], dtype=float).copy()
        worktable_peak_m = 0.0
        termination_cause = None
        for step in range(args.horizon_steps):
            try:
                normalized = controller.action(observation, time_s=step / profile.policy_rate_hz)
            except ShakeBenchOracleError:
                termination_cause = "policy_error"
                break
            observation = oracle_observation(env, env.step(np.clip(normalized, -1.0, 1.0))[0])
            worktable_peak_m = max(
                worktable_peak_m, float(np.max(np.abs(np.asarray(env.sim.data.xpos[worktable_id]) - reference)))
            )
            observer(env, step, observation, controller)
            frames += 1
            metrics = env.get_metrics()
            termination_cause = resolve_termination_cause(
                prior_cause=None,
                task_rule_violation=False,
                success_latched=bool(metrics["success"]["passed"]),
                policy_abort=controller.abort_requested,
                horizon_exhausted=False,
            )
            if termination_cause is not None:
                break
        else:
            termination_cause = "horizon_exhausted"
        success = termination_cause == "success_latched"
        if termination_cause in ("policy_error",):
            validate_outcome(
                episode_validity="valid", score_outcome="unsuccessful", termination_cause=termination_cause
            )
        else:
            validate_outcome(
                episode_validity="valid",
                score_outcome="success" if success else "unsuccessful",
                termination_cause=termination_cause,
            )
        observer.close(success=success, hold_seconds=args.hold_seconds)
        frames += max(1, round(args.fps * args.hold_seconds))
    except BaseException:
        observer.close(success=False, hold_seconds=0.0)
        raise
    finally:
        env.close()
    return {
        "gamma": gamma,
        "state_id": state["state_id"],
        "success": success,
        "termination_cause": termination_cause,
        "steps": frames - max(1, round(args.fps * args.hold_seconds)),
        "frames": frames,
        "worktable_peak_m": worktable_peak_m,
        "abort_reason": controller.executive.abort_reason,
        "final_phase": controller.executive.phase.value,
        "recovery_count": int(controller.executive.recovery_count),
        "wall_time_s": time.perf_counter() - started,
        "video": output.name,
        "video_sha256": _sha256(output),
        "video_bytes": output.stat().st_size,
    }


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--states", type=Path, default=DEFAULT_STATES)
    parser.add_argument("--state-id", default="shakebench-dev-v0-000")
    parser.add_argument("--gammas", type=parse_gammas, default=DEFAULT_GAMMAS)
    parser.add_argument("--mode", default="multisine_v1")
    parser.add_argument("--camera", default="task_close")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--hold-seconds", type=float, default=1.5)
    parser.add_argument("--raw", action="store_true", help="write unannotated camera frames")
    parser.add_argument("--horizon-steps", type=int, default=1200)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if min(args.width, args.height, args.fps, args.horizon_steps) <= 0 or args.hold_seconds < 0.0:
        raise ValueError("video dimensions, frame rate, horizon, and hold time must be positive")
    gl_backend = os.environ.get("MUJOCO_GL")
    if gl_backend != "egl":
        raise ValueError("set MUJOCO_GL=egl to render on the GPU; refusing a non-GPU recording")
    state = load_state(args.states, args.state_id)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_id": SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "qualitative_only": True,
        "scoreable": False,
        "geometry_profile": GEOMETRY_PROFILE,
        "tier": TIER,
        "mode": args.mode,
        "camera": args.camera,
        "width": args.width,
        "height": args.height,
        "fps": args.fps,
        "gl_backend": gl_backend,
        "egl_vendor_libraries": os.environ.get("__EGL_VENDOR_LIBRARY_FILENAMES"),
        "episodes": [],
    }
    for gamma in args.gammas:
        output = args.output_dir / f"gamma_{gamma:.2f}.mp4"
        episode = record_episode(state, gamma=gamma, mode=args.mode, args=args, output=output)
        summary["episodes"].append(episode)
        write_json(args.output_dir / "summary.json", summary)
        print(json.dumps(episode, sort_keys=True), flush=True)
    return 0 if all(episode["success"] for episode in summary["episodes"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
