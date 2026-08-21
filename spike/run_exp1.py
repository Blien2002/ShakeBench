"""Experiment 1 entry point; currently enforces the mandatory 1a gate first."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import platform

import mujoco
import numpy as np
import robosuite

from env_shakedeck import ANCHOR, make_env
from metrics import EpisodeMetrics, contact_snapshot
from policies import ReactiveScriptedPolicy
from vibration import calibrated_vibration


DEFAULT_SEED = 17
DEFAULT_TIMESTEP = 2.0e-4
DEFAULT_WINDOW_S = 6.0
DEFAULT_CONTROL_FREQ = 20


def _frame_position(env, body_id: int, point_w: np.ndarray) -> np.ndarray:
    rotation_w = np.asarray(env.sim.data.body_xmat[body_id]).reshape(3, 3)
    origin_w = np.asarray(env.sim.data.body_xpos[body_id])
    return rotation_w.T @ (point_w - origin_w)


def _warning_count(env) -> int:
    return int(sum(item.number for item in env.sim.data.warning))


def _eef_site_id(env) -> int:
    ids = env.robots[0].eef_site_id
    if isinstance(ids, dict):
        return int(ids.get("right", next(iter(ids.values()))))
    return int(ids)


def _robot_qpos(env) -> np.ndarray:
    addresses = [env.sim.model.get_joint_qpos_addr(f"robot0_joint{i}") for i in range(1, 8)]
    return np.asarray([env.sim.data.qpos[int(address)] for address in addresses], dtype=np.float64)


def _cube_table_penetration(env) -> float | None:
    cube_geom = env.sim.model.geom_name2id("cube_g0")
    table_geom = env.sim.model.geom_name2id("table_collision")
    pair = {cube_geom, table_geom}
    distances = [
        float(contact.dist)
        for contact in env.sim.data.contact[: env.sim.data.ncon]
        if {int(contact.geom1), int(contact.geom2)} == pair
    ]
    if not distances:
        return None
    return max(0.0, -min(distances))


def _rollout(seed: int, timestep: float, window_s: float, vibration) -> dict:
    control_freq = DEFAULT_CONTROL_FREQ
    steps = int(round(window_s * control_freq))
    env = make_env(
        seed=seed,
        physics_timestep=timestep,
        motion_sampler=None if vibration is None else vibration.sample,
        control_freq=control_freq,
        horizon=steps + 1,
    )
    try:
        action = np.zeros(env.action_dim, dtype=np.float64)
        eef_site_id = _eef_site_id(env)
        base_body_id = env.sim.model.body_name2id("robot0_base")
        cube_body_id = env.sim.model.body_name2id("cube_main")
        start_qpos = _robot_qpos(env)
        start_eef_w = np.asarray(env.sim.data.site_xpos[eef_site_id]).copy()
        start_eef_b = _frame_position(env, base_body_id, start_eef_w)
        start_cube_b = _frame_position(
            env, base_body_id, np.asarray(env.sim.data.body_xpos[cube_body_id]).copy()
        )

        qpos_trace = []
        eef_world_drift = []
        eef_base_delta = []
        cube_base_delta = []
        command_displacement = []
        weld_error = []
        penetrations = []
        for _ in range(steps):
            env.step(action)
            eef_w = np.asarray(env.sim.data.site_xpos[eef_site_id]).copy()
            cube_w = np.asarray(env.sim.data.body_xpos[cube_body_id]).copy()
            eef_b = _frame_position(env, base_body_id, eef_w)
            cube_b = _frame_position(env, base_body_id, cube_w)
            command_pos, _command_quat = env.commanded_deck_pose()
            actual_deck_pos = np.asarray(env.sim.data.site_xpos[env.deck_site_id]).copy()
            penetration = _cube_table_penetration(env)

            qpos_trace.append(np.asarray(env.sim.data.qpos).copy())
            eef_world_drift.append(float(np.linalg.norm(eef_w - start_eef_w)))
            eef_base_delta.append(float(np.linalg.norm(eef_b - start_eef_b)))
            cube_base_delta.append(float(np.linalg.norm(cube_b - start_cube_b)))
            command_displacement.append(float(np.linalg.norm(command_pos - ANCHOR)))
            weld_error.append(float(np.linalg.norm(actual_deck_pos - command_pos)))
            if penetration is not None:
                penetrations.append(penetration)

        final_window = max(1, control_freq)
        return {
            "qpos_trace": np.asarray(qpos_trace),
            "ee_world_drift_max_m": max(eef_world_drift),
            "joint_drift_max_rad": float(np.max(np.abs(_robot_qpos(env) - start_qpos))),
            "command_displacement_peak_m": max(command_displacement),
            "weld_tracking_error_peak_m": max(weld_error),
            "weld_tracking_error_p90_m": float(np.percentile(weld_error, 90)),
            "ee_wobble_base_m": max(eef_base_delta),
            "obj_slip_on_table_m": max(cube_base_delta),
            "static_penetration_median_m": (
                float(np.median(penetrations[-final_window:])) if penetrations else None
            ),
            "static_penetration_max_m": max(penetrations) if penetrations else None,
            "contact_samples": len(penetrations),
            "mujoco_warning_count": _warning_count(env),
            "compiled_timestep_s": float(env.sim.model.opt.timestep),
            "mocap_body_count": int(env.sim.model.nmocap),
        }
    finally:
        env.close()


def run_selfcheck(seed: int, timestep: float, window_s: float) -> dict:
    physics_hz = int(round(1.0 / timestep))
    vibration, calibration = calibrated_vibration(
        0.5,
        seed=seed,
        physics_hz=physics_hz,
        episode_s=16.0,
    )
    zero = _rollout(seed, timestep, window_s, vibration=None)
    shaken_a = _rollout(seed, timestep, window_s, vibration=vibration)
    shaken_b = _rollout(seed, timestep, window_s, vibration=vibration)
    determinism_error = float(np.max(np.abs(shaken_a["qpos_trace"] - shaken_b["qpos_trace"])))
    command_peak = shaken_a["command_displacement_peak_m"]
    tracking_ratio = shaken_a["weld_tracking_error_peak_m"] / command_peak if command_peak else None

    checks = {
        "zero_trajectory": {
            "pass": zero["ee_world_drift_max_m"] < 5.0e-5
            and zero["joint_drift_max_rad"] < 1.0e-3,
            "ee_drift_m": zero["ee_world_drift_max_m"],
            "joint_drift_rad": zero["joint_drift_max_rad"],
            "limits": {"ee_drift_m_lt": 5.0e-5, "joint_drift_rad_lt": 1.0e-3},
        },
        "weld_tracking": {
            "pass": tracking_ratio is not None and tracking_ratio < 0.20,
            "command_peak_m": command_peak,
            "error_peak_m": shaken_a["weld_tracking_error_peak_m"],
            "error_p90_m": shaken_a["weld_tracking_error_p90_m"],
            "error_over_command_peak": tracking_ratio,
            "limit_ratio_lt": 0.20,
        },
        "static_penetration": {
            "pass": zero["static_penetration_median_m"] is not None
            and zero["static_penetration_median_m"] < 5.0e-6,
            "median_m": zero["static_penetration_median_m"],
            "max_m": zero["static_penetration_max_m"],
            "contact_samples": zero["contact_samples"],
            "diagnostic_limit_m_lt": 5.0e-6,
            "expected_order_m": 2.0e-7,
        },
        "mujoco_warnings": {
            "pass": zero["mujoco_warning_count"] == 0
            and shaken_a["mujoco_warning_count"] == 0
            and shaken_b["mujoco_warning_count"] == 0,
            "counts": [
                zero["mujoco_warning_count"],
                shaken_a["mujoco_warning_count"],
                shaken_b["mujoco_warning_count"],
            ],
        },
        "determinism": {
            "pass": determinism_error == 0.0,
            "max_abs_qpos_difference": determinism_error,
            "expected": 0.0,
        },
    }
    return {
        "schema": "shakebench_spike.exp1_selfcheck.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "versions": {
            "python": platform.python_version(),
            "robosuite": robosuite.__version__,
            "mujoco": mujoco.__version__,
            "numpy": np.__version__,
        },
        "configuration": {
            "seed": seed,
            "gamma": 0.5,
            "physics_timestep_s": timestep,
            "physics_hz": physics_hz,
            "control_freq_hz": DEFAULT_CONTROL_FREQ,
            "measurement_window_s": window_s,
            "calibration_episode_s": 16.0,
        },
        "calibration": calibration,
        "checks": checks,
        "all_passed": all(item["pass"] for item in checks.values()),
        "reference_scale": {
            "level_scale": 1.35567,
            "gamma_realized": 0.5,
            "full_episode_peak_deck_displacement_m": 0.002160,
            "full_episode_peak_deck_velocity_m_s": 0.0958,
            "six_second_command_peak_m": 0.00181,
            "six_second_weld_error_m": 8.88e-5,
            "six_second_ee_wobble_base_m": 0.00120,
            "six_second_obj_slip_on_table_m": 2.6e-5,
        },
        "zero_rollout": {key: value for key, value in zero.items() if key != "qpos_trace"},
        "shaken_rollout": {key: value for key, value in shaken_a.items() if key != "qpos_trace"},
    }


def _runtime_warning_count(env) -> int:
    return int(sum(item.number for item in env.sim.data.warning))


def run_task_episode(seed: int, timestep: float, episode_s: float, gamma: float) -> dict:
    physics_hz = int(round(1.0 / timestep))
    vibration, calibration = calibrated_vibration(
        gamma,
        seed=seed,
        physics_hz=physics_hz,
        episode_s=episode_s,
    )
    max_policy_steps = int(round(episode_s * DEFAULT_CONTROL_FREQ))
    env = make_env(
        seed=seed,
        physics_timestep=timestep,
        motion_sampler=vibration.sample,
        control_freq=DEFAULT_CONTROL_FREQ,
        horizon=max_policy_steps + 1,
        direct_gripper=True,
    )
    try:
        policy = ReactiveScriptedPolicy(env)
        metrics = EpisodeMetrics.start(env)
        actions: list[list[float]] = []
        for _ in range(max_policy_steps):
            before = contact_snapshot(env)
            action = policy.command(before)
            actions.append(action.tolist())
            env.step(action)
            metrics.update(env, policy, contact_snapshot(env))
            if policy.finished:
                break
        result = metrics.result(policy, _runtime_warning_count(env))
        result.update(
            {
                "schema": "shakebench_spike.exp1_episode.v1",
                "seed": seed,
                "gamma": gamma,
                "physics_timestep_s": timestep,
                "physics_hz": physics_hz,
                "control_freq_hz": DEFAULT_CONTROL_FREQ,
                "episode_limit_s": episode_s,
                "elapsed_s": float(env.sim.data.time),
                "policy_steps": len(actions),
                "calibration": calibration,
                # Exact post-latch commands are retained for experiment 2 tapes.
                "actions": actions,
            }
        )
        return result
    finally:
        env.close()


def run_experiment1(
    *,
    seed_start: int,
    episodes: int,
    timestep: float,
    episode_s: float,
    gamma: float,
    output_dir: Path,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    episode_results = []
    for seed in range(seed_start, seed_start + episodes):
        result = run_task_episode(seed, timestep, episode_s, gamma)
        episode_results.append(result)
        episode_path = output_dir / f"exp1_seed_{seed:03d}.json"
        episode_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(
            f"seed={seed:03d} success={result['success']} phase={result['phase_history'][-1]} "
            f"failure={result['failure_reason']} slip_mm={1000.0 * result['max_grasp_slip_m']:.3f} "
            f"lift_mm={1000.0 * result['max_object_lift_m']:.1f}"
        )

    failures = Counter(result["failure_reason"] for result in episode_results if not result["success"])
    slips = np.asarray([result["max_grasp_slip_m"] for result in episode_results], dtype=np.float64)
    success_count = sum(bool(result["success"]) for result in episode_results)
    summary = {
        "schema": "shakebench_spike.exp1_summary.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "versions": {
            "python": platform.python_version(),
            "robosuite": robosuite.__version__,
            "mujoco": mujoco.__version__,
            "numpy": np.__version__,
        },
        "configuration": {
            "seed_start": seed_start,
            "episodes": episodes,
            "seeds": list(range(seed_start, seed_start + episodes)),
            "gamma": gamma,
            "physics_timestep_s": timestep,
            "control_freq_hz": DEFAULT_CONTROL_FREQ,
            "episode_limit_s": episode_s,
            "grasp_slip_tolerance_m": 0.010,
            "grasp_assist": False,
        },
        "success_count": success_count,
        "success_rate": success_count / episodes,
        "failure_reason_histogram": dict(sorted(failures.items())),
        "max_grasp_slip_distribution_m": {
            "median": float(np.median(slips)),
            "p90": float(np.percentile(slips, 90)),
            "max": float(np.max(slips)),
        },
        "grasp_slip_exceeded_count": sum(bool(result["grasp_slip_exceeded"]) for result in episode_results),
        "grasp_slip_exceeded_reproduced": any(
            bool(result["grasp_slip_exceeded"]) for result in episode_results
        ),
        "experiment1_gate_passed": success_count / episodes >= 0.80,
        "episode_files": [f"exp1_seed_{result['seed']:03d}.json" for result in episode_results],
    }
    (output_dir / "exp1_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true", help="run mandatory experiment 1a gate")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--timestep", type=float, default=DEFAULT_TIMESTEP)
    parser.add_argument("--window-s", type=float, default=DEFAULT_WINDOW_S)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--episode-s", type=float, default=16.0)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    args = parser.parse_args()
    default_out = Path(__file__).resolve().parent / "out"
    if args.self_check:
        result = run_selfcheck(args.seed, args.timestep, args.window_s)
        output = args.output or default_out / "exp1_selfcheck.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["all_passed"] else 2
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    selfcheck_path = default_out / "exp1_selfcheck.json"
    if not selfcheck_path.is_file() or not json.loads(selfcheck_path.read_text())["all_passed"]:
        parser.error("experiment 1a gate has not passed; run --self-check first")
    result = run_experiment1(
        seed_start=args.seed,
        episodes=args.episodes,
        timestep=args.timestep,
        episode_s=args.episode_s,
        gamma=args.gamma,
        output_dir=args.output or default_out,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["experiment1_gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
