"""Collect state/action demonstrations with MJWarp and the existing CPU oracle.

Each NPZ stores T actions and T+1 observations/physics states; JSON metadata
records backend provenance and outcome. These artifacts are not CPU benchmark
artifacts and are always non-scoreable. No rendering is performed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from robosuite.scripts.shakebench_run_oracle import _json_ready, load_state_asset
from robosuite.utils.shakebench_artifacts import write_json_atomic
from robosuite.utils.shakebench_metrics import DEFAULT_SUCCESS_THRESHOLDS
from robosuite.utils.shakebench_oracle import (
    OracleControllerProfile, ShakeBenchOracleController, ShakeBenchOracleError, WorktableTaskContext,
)
from robosuite.utils.shakebench_outcomes import resolve_termination_cause, validate_outcome
from robosuite.utils.shakebench_calibration import vibration_record
from robosuite.utils.shakebench_providers import TABLE_IMU_POLICY_KEYS


def make_environment(state, *, gamma, horizon, mode="multisine_v1", physics_profile="official"):
    """Use the same inputs and reset sequence as the CPU oracle runner."""
    import robosuite
    from robosuite.controllers import load_composite_controller_config
    from robosuite.utils.shakebench_tasks import task_env_kwargs

    seed = int(state.get("excitation_seed", state.get("seed", 0)))
    t0 = float(state.get("t0_s", 0.0))
    variant = "task" in state
    kwargs = task_env_kwargs(state) if variant else {"object_start_xy": tuple(state["object_xy_m"])}
    env = robosuite.make(
        "VibrationPickPlace", robots="Panda",
        controller_configs=load_composite_controller_config(robot="Panda"), has_renderer=False,
        has_offscreen_renderer=False, use_camera_obs=False, use_object_obs=False,
        physics_profile=physics_profile, geometry_profile="world_fixed_arm_v1",
        imu_mode="canonical_noisy_v1", vibration={"mode": mode, "gamma": gamma, "seed": seed, "t0_s": t0}, **kwargs,
        imu_seed=int(state.get("imu_seed", seed)), horizon=horizon, ignore_done=True, seed=seed, hard_reset=False,
    )
    try:
        env.reset()
    except BaseException:
        env.close()
        raise
    return env, env.deck_driver.trajectory


def _write_npz(path, arrays):
    """Publish complete episodes atomically, never overwrite an existing file."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)  # exclusive publication on the same filesystem
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def collect_batch(batch, states, *, horizon, profile=None):
    """Run closed-loop oracle episodes; completed worlds are no longer recorded."""
    profile = profile or OracleControllerProfile()
    if len(states) != batch.nworld or isinstance(horizon, bool) or int(horizon) != horizon or horizon < 1:
        raise ValueError("one state per world and a positive integer horizon are required")
    if profile.policy_rate_hz != 20:
        raise ValueError("GPU collector requires policy_rate_hz=20")
    observations = batch.reset()
    controllers = [ShakeBenchOracleController(profile, task_context=WorktableTaskContext.from_mapping(
        env.get_policy_task_context().get("task_context"))) for env in batch.envs]
    # ponytail: worlds finish as a batch; use masked resets only if tail waste
    # becomes a measured throughput bottleneck. Terminal snapshots are retained.
    records = [{"observations": [obs], "actions": [], "qpos": [batch.initial_arrays["qpos"][w].copy()],
                "qvel": [batch.initial_arrays["qvel"][w].copy()], "ctrl": [], "actuator_force": [],
                "contacts": [], "phases": [], "termination_cause": None, "episode_validity": "valid",
                "complete_event_recorded": False}
               for w, obs in enumerate(observations)]
    start = time.perf_counter()
    for step in range(horizon):
        active = [w for w, record in enumerate(records) if record["termination_cause"] is None]
        if not active:
            break
        if step % 100 == 0:
            print(f"Rollout step {step}/{horizon}: {len(active)} active worlds", flush=True)
        actions = np.zeros((len(states), 7))
        for w in active:
            try:
                actions[w] = controllers[w].action(observations[w], time_s=step / 20)
                if not np.all(np.isfinite(actions[w])):
                    raise ShakeBenchOracleError("non-finite oracle action")
                if controllers[w].executive.phase.value == "complete" and not records[w]["complete_event_recorded"]:
                    controllers[w].executive.record_evaluator_not_latched(observations[w], step / 20)
                    records[w]["complete_event_recorded"] = True
            except ShakeBenchOracleError:
                records[w]["termination_cause"] = "policy_error"
                actions[w] = 0
        active = [w for w in active if records[w]["termination_cause"] is None]
        if not active:
            continue
        observations, metrics = batch.step(np.clip(actions, -1, 1))
        for w in active:
            record = records[w]
            if metrics["invalid"][w]:
                record["episode_validity"] = "invalid"
                record["termination_cause"] = "invalid_execution"
                continue
            record["observations"].append(observations[w])
            record["actions"].append(actions[w].copy())
            for field in ("qpos", "qvel", "ctrl", "actuator_force", "contacts"):
                record[field].append(metrics[field][w].copy())
            record["phases"].append(controllers[w].last_trace["phase"])
            record["termination_cause"] = resolve_termination_cause(
                prior_cause=None, task_rule_violation=bool(
                    metrics["contacts"][w, 0] >= DEFAULT_SUCCESS_THRESHOLDS.max_illegal_penetration_m),
                success_latched=bool(metrics["success"][w]), policy_abort=controllers[w].abort_requested,
                horizon_exhausted=step + 1 == horizon,
            )
    elapsed = time.perf_counter() - start
    for record, controller in zip(records, controllers):
        record["termination_cause"] = record["termination_cause"] or "horizon_exhausted"
        record["score_outcome"] = (None if record["episode_validity"] == "invalid" else
                                   "success" if record["termination_cause"] == "success_latched" else "unsuccessful")
        record["controller_events"] = controller.executive.controller_events
        validate_outcome(**{key: record[key] for key in ("episode_validity", "score_outcome", "termination_cause")})
    return records, elapsed


def _positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--tier", choices=("V0",), default="V0",
        help="deprecated alias kept for old commands; the current expert is the single-lane one",
    )
    gamma_args = parser.add_mutually_exclusive_group()
    gamma_args.add_argument("--gamma", type=float, help="one Gamma (default: 0)")
    gamma_args.add_argument("--gammas", type=float, nargs="+", help="Gamma sweep, e.g. 0 0.15 0.3 0.6")
    parser.add_argument("--mode", default="multisine_v1", choices=("multisine_v1", "single_sine_v1"))
    parser.add_argument("--physics-profile", choices=("official", "probe"), default="official",
                        help="official retains frozen hard contacts; probe uses the existing exploratory contact profile")
    parser.add_argument("--num-worlds", type=_positive, default=16)
    parser.add_argument("--horizon-steps", type=_positive, default=1200)
    parser.add_argument("--limit", type=_positive)
    parser.add_argument("--state-id", action="append", help="repeat to select specific states")
    parser.add_argument("--device", default="cuda:0", help="cuda:N for collection; cpu for slow debugging")
    parser.add_argument("--nconmax", type=_positive, default=128)
    parser.add_argument("--njmax", type=_positive, default=512)
    parser.add_argument("--no-capture", action="store_true", help="disable CUDA graphs for debugging")
    args = parser.parse_args(argv)
    gammas = list(dict.fromkeys(args.gammas or [0.0 if args.gamma is None else args.gamma]))
    if any(not np.isfinite(gamma) or gamma < 0 for gamma in gammas):
        parser.error("Gamma values must be finite and non-negative")
    try:
        import mujoco
        import mujoco_warp as mjw
        import warp as wp
        from robosuite.utils.shakebench_mjwarp import MJWarpBatch
    except ImportError as exc:
        parser.error(f"GPU dependencies unavailable: {exc}. Install requirements-gpu.txt in your collection environment.")
    wp.init()
    try:
        device = wp.get_device(args.device)
    except Exception as exc:
        parser.error(f"cannot use device {args.device}: {exc}")
    if not device.is_cuda and args.device != "cpu":
        parser.error("no CUDA device available; --device cpu is only for debugging")
    asset = load_state_asset(args.states)
    states = asset["states"]
    if args.state_id:
        missing = set(args.state_id) - {s["state_id"] for s in states}
        if missing:
            parser.error(f"unknown state IDs: {sorted(missing)}")
        states = [s for s in states if s["state_id"] in args.state_id]
    if args.limit is not None:
        states = states[:args.limit]
    if not states:
        parser.error("no states selected")
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        parser.error("output must be a new or empty directory; existing episodes are never overwritten")
    args.output.mkdir(parents=True, exist_ok=True)
    grouped = defaultdict(list)
    for index, (state, gamma) in enumerate((state, gamma) for state in states for gamma in gammas):
        grouped[json.dumps(state.get("task"), sort_keys=True)].append((index, state, gamma))
    manifest = {"schema_id": "shakebench.mjwarp.collection", "schema_version": 2, "scoreable": False,
                "backend": "mujoco_warp", "device": str(device), "mujoco_version": mujoco.__version__,
                "mujoco_warp_version": mjw.__version__, "warp_version": wp.__version__,
                "expert_contract": "privileged_current_state_v1", "geometry_profile": "world_fixed_arm_v1",
                "gammas": gammas, "mode": args.mode, "state_authority": asset["authority"],
                "physics_profile": args.physics_profile,
                "horizon_steps": args.horizon_steps, "episodes": [], "complete": False,
                "requested_num_worlds": args.num_worlds, "nconmax": args.nconmax, "njmax": args.njmax,
                "cuda_graph": bool(device.is_cuda and not args.no_capture),
                "contact_columns": ["penetration_m", "bottom_support_force_N", "bottom_contact", "finger_contact"]}
    started = time.perf_counter()
    profile = OracleControllerProfile()
    manifest["controller_profile"] = profile.to_dict()
    write_json_atomic(args.output / "manifest.json", manifest)
    for group in grouped.values():
        for begin in range(0, len(group), args.num_worlds):
            selected = group[begin:begin + args.num_worlds]
            envs, programs = [], []
            try:
                for _, state, gamma in selected:
                    print(f"Preparing {state['state_id']} Gamma={gamma:g}", flush=True)
                    env, program = make_environment(state, gamma=gamma, mode=args.mode, horizon=args.horizon_steps,
                                                    physics_profile=args.physics_profile)
                    envs.append(env)
                    programs.append(program)
                batch = MJWarpBatch(envs, programs, device=str(device), nconmax=args.nconmax, njmax=args.njmax,
                                    capture=not args.no_capture)
                records, elapsed = collect_batch(batch, [s for _, s, _ in selected], horizon=args.horizon_steps, profile=profile)
                solver_tolerance = float(batch.model.opt.tolerance.numpy().reshape(-1)[0])
                model_path = args.output / f"model_{batch.model_sha256}.xml"
                if not model_path.exists():
                    with model_path.open("x") as stream:
                        stream.write(envs[0].sim.model.get_xml())
                for w, ((index, state, gamma), record) in enumerate(zip(selected, records)):
                    name = f"episode_{index:06d}.npz"
                    meta = {"state": state, "expert_contract": manifest["expert_contract"], "gamma_commanded": gamma,
                            "schema_version": 2, "geometry_profile": envs[w].geometry_profile,
                            "vibration": vibration_record(programs[w]),
                            "imu_mount": envs[w].table_imu_provider.audit_compiled_mount(envs[w].sim),
                            "scoreable": False, "backend": "mujoco_warp", "model_sha256": batch.model_sha256,
                            "physics_profile_sha256": envs[w].physics_profile.profile_sha256,
                            "physics_profile": envs[w].physics_profile.to_dict(),
                            "effective_solver_tolerance": solver_tolerance,
                            "program": programs[w].to_dict(), "steps": len(record["actions"]),
                            **{key: record[key] for key in ("episode_validity", "score_outcome", "termination_cause", "controller_events")}}
                    arrays = {("observations/" if key in TABLE_IMU_POLICY_KEYS else "privileged_oracle/") + key:
                              np.stack([o[key] for o in record["observations"]])
                              for key in record["observations"][0]}
                    for key, width in (("actions", 7), ("ctrl", batch.raw_model.nu), ("actuator_force", batch.raw_model.nu),
                                       ("contacts", 4), ("qpos", batch.raw_model.nq), ("qvel", batch.raw_model.nv)):
                        arrays[key] = np.asarray(record[key]).reshape(-1, width)
                    arrays["clipped_actions"] = np.clip(arrays["actions"], -1, 1)
                    arrays["time_s"] = np.arange(len(record["observations"])) / 20
                    arrays["phase"] = np.asarray(record["phases"], dtype="U64")
                    arrays["metadata_json"] = np.asarray(json.dumps(_json_ready(meta), sort_keys=True, allow_nan=False))
                    _write_npz(args.output / name, arrays)
                    manifest["episodes"].append({"file": name, "sha256": hashlib.sha256((args.output / name).read_bytes()).hexdigest(),
                                                 "state_id": state["state_id"], "gamma": gamma, "steps": meta["steps"],
                                                 **{key: meta[key] for key in ("episode_validity", "score_outcome", "termination_cause")}})
                write_json_atomic(args.output / "manifest.json", manifest)
                print(f"Collected {len(manifest['episodes'])}/{len(states) * len(gammas)} episodes; batch rollout {elapsed:.2f}s", flush=True)
                del batch
            finally:
                for env in envs:
                    env.close()
    manifest["complete"] = True
    manifest["wall_time_s"] = time.perf_counter() - started
    manifest["valid_episodes_per_hour"] = sum(e["episode_validity"] == "valid" for e in manifest["episodes"]) * 3600 / manifest["wall_time_s"]
    write_json_atomic(args.output / "manifest.json", manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
