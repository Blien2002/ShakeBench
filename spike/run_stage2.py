"""Stage 2 entry point; this revision intentionally implements Stage A only."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform

import mujoco
import numpy as np
import robosuite

from env_shakedeck import ANCHOR, make_env
from metrics import EpisodeMetrics, contact_snapshot, eef_position_b, object_position_b
from policies import ReactiveConfig, ReactiveScriptedPolicy
from vibration import calibrated_vibration


GAMMA_LADDER = (0.15, 0.30, 0.50, 0.75, 0.95)
REFERENCE_EE_WOBBLE_M = {
    0.15: 0.00033,
    0.30: 0.00070,
    0.50: 0.00120,
    0.75: 0.00180,
    0.95: 0.00227,
}
POLICY_HZ = 20
DIAGNOSTIC_SAMPLE_HZ = 200
GRIP_FORCE_LIMIT_N_PER_FINGER = 3.0
FINGER_PAD_MU = 2.0
WORKPIECE_SIDE_MU = 1.5
GAMMA_095_LOAD_MULTIPLIER = 1.95
MIN_CUBE_MASS_KG = 0.07011070102818318
MAX_CUBE_MASS_KG = 0.0830013231861482


def _versions() -> dict:
    return {
        "python": platform.python_version(),
        "robosuite": robosuite.__version__,
        "mujoco": mujoco.__version__,
        "numpy": np.__version__,
    }


def _warning_count(env) -> int:
    return int(sum(item.number for item in env.sim.data.warning))


def _sample_decimation(timestep: float, sample_hz: int) -> int:
    physics_hz = int(round(1.0 / timestep))
    if physics_hz % sample_hz:
        raise ValueError("physics frequency must be divisible by diagnostic sample frequency")
    return physics_hz // sample_hz


def measure_ee_wobble(
    *,
    gamma: float,
    seed: int,
    timestep: float,
    window_s: float,
    sample_hz: int,
) -> dict:
    physics_hz = int(round(1.0 / timestep))
    vibration, calibration = calibrated_vibration(
        gamma,
        seed=seed,
        physics_hz=physics_hz,
        episode_s=16.0,
    )
    control_steps = int(round(window_s * POLICY_HZ))
    env = make_env(
        seed=seed,
        physics_timestep=timestep,
        motion_sampler=vibration.sample,
        control_freq=POLICY_HZ,
        horizon=control_steps + 1,
    )
    try:
        initial_eef_b = eef_position_b(env).copy()
        initial_object_b = object_position_b(env).copy()
        decimation = _sample_decimation(timestep, sample_hz)
        physics_steps = 0
        eef_delta_m: list[float] = []
        object_delta_m: list[float] = []
        command_displacement_m: list[float] = []
        weld_error_m: list[float] = []

        def sample(stepped_env) -> None:
            nonlocal physics_steps
            physics_steps += 1
            if physics_steps % decimation:
                return
            eef_delta_m.append(float(np.linalg.norm(eef_position_b(stepped_env) - initial_eef_b)))
            object_delta_m.append(
                float(np.linalg.norm(object_position_b(stepped_env) - initial_object_b))
            )
            command_pos, _ = stepped_env.commanded_deck_pose()
            actual_pos = np.asarray(
                stepped_env.sim.data.site_xpos[stepped_env.deck_site_id]
            ).copy()
            command_displacement_m.append(float(np.linalg.norm(command_pos - ANCHOR)))
            weld_error_m.append(float(np.linalg.norm(actual_pos - command_pos)))

        env.physics_step_callback = sample
        action = np.zeros(env.action_dim, dtype=np.float64)
        for _ in range(control_steps):
            env.step(action)
        return {
            "gamma": gamma,
            "seed": seed,
            "measurement_window_s": window_s,
            "policy_frequency_hz": POLICY_HZ,
            "diagnostic_sample_frequency_hz": sample_hz,
            "sample_count": len(eef_delta_m),
            "ee_wobble_base_m": max(eef_delta_m),
            "object_motion_base_m": max(object_delta_m),
            "command_displacement_peak_m": max(command_displacement_m),
            "weld_tracking_error_peak_m": max(weld_error_m),
            "mujoco_warning_count": _warning_count(env),
            "calibration": calibration,
        }
    finally:
        env.close()


def run_wobble_ladder(
    *,
    seed: int,
    timestep: float,
    window_s: float,
    sample_hz: int,
) -> dict:
    rows = []
    for gamma in GAMMA_LADDER:
        row = measure_ee_wobble(
            gamma=gamma,
            seed=seed,
            timestep=timestep,
            window_s=window_s,
            sample_hz=sample_hz,
        )
        row["reference_ee_wobble_m"] = REFERENCE_EE_WOBBLE_M[gamma]
        row["measured_over_reference"] = (
            row["ee_wobble_base_m"] / row["reference_ee_wobble_m"]
        )
        rows.append(row)
        print(
            f"gamma={gamma:.2f} ee_wobble_mm={1000 * row['ee_wobble_base_m']:.4f} "
            f"reference_mm={1000 * row['reference_ee_wobble_m']:.4f} "
            f"samples={row['sample_count']}"
        )
    return {
        "schema": "shakebench_spike.stage2.ee_wobble_ladder.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "versions": _versions(),
        "configuration": {
            "seed": seed,
            "physics_timestep_s": timestep,
            "physics_hz": int(round(1.0 / timestep)),
            "policy_frequency_hz": POLICY_HZ,
            "diagnostic_sample_frequency_hz": sample_hz,
            "measurement_window_s": window_s,
            "gamma_values": list(GAMMA_LADDER),
        },
        "rows": rows,
    }


def run_force_grip_design_check(
    *,
    seed: int,
    gamma: float,
    timestep: float,
    episode_s: float,
    sample_hz: int,
) -> dict:
    physics_hz = int(round(1.0 / timestep))
    vibration, calibration = calibrated_vibration(
        gamma,
        seed=seed,
        physics_hz=physics_hz,
        episode_s=episode_s,
    )
    policy_steps = int(round(episode_s * POLICY_HZ))
    env = make_env(
        seed=seed,
        physics_timestep=timestep,
        motion_sampler=vibration.sample,
        control_freq=POLICY_HZ,
        horizon=policy_steps + 1,
        direct_gripper=True,
        gripper_force_limit_n=GRIP_FORCE_LIMIT_N_PER_FINGER,
    )
    try:
        policy = ReactiveScriptedPolicy(
            env,
            ReactiveConfig(
                control_freq_hz=POLICY_HZ,
                grip_mode="force_limited_close",
            ),
        )
        metrics = EpisodeMetrics.start(env)
        decimation = _sample_decimation(timestep, sample_hz)
        physics_steps = 0

        def sample(stepped_env) -> None:
            nonlocal physics_steps
            physics_steps += 1
            if physics_steps % decimation:
                return
            metrics.update(stepped_env, policy, contact_snapshot(stepped_env))

        env.physics_step_callback = sample
        actions = []
        for _ in range(policy_steps):
            action = policy.command(contact_snapshot(env))
            actions.append(action.tolist())
            env.step(action)
            if policy.finished:
                break
        result = metrics.result(policy, _warning_count(env))
        result.update(
            {
                "schema": "shakebench_spike.stage2.force_grip_design_check.v1",
                "seed": seed,
                "gamma": gamma,
                "physics_timestep_s": timestep,
                "policy_frequency_hz": POLICY_HZ,
                "diagnostic_sample_frequency_hz": sample_hz,
                "elapsed_s": float(env.sim.data.time),
                "policy_steps": len(actions),
                "grip_mode": policy.config.grip_mode,
                "full_close_target_m": 0.0,
                "gripper_force_limit_n_per_finger": GRIP_FORCE_LIMIT_N_PER_FINGER,
                "gripper_actuators": env.gripper_actuator_configuration(),
                "contact_configuration": env.contact_configuration(),
                "object_mass_kg": float(
                    env.sim.model.body_subtreemass[
                        env.sim.model.body_name2id("cube_main")
                    ]
                ),
                "calibration": calibration,
                "actions": actions,
            }
        )
        return result
    finally:
        env.close()


def grip_design_document(check: dict) -> dict:
    limiting_mu = min(FINGER_PAD_MU, WORKPIECE_SIDE_MU)
    static_min = MIN_CUBE_MASS_KG * 9.81 / (2.0 * limiting_mu)
    static_max = MAX_CUBE_MASS_KG * 9.81 / (2.0 * limiting_mu)
    gamma_095_min = static_min * GAMMA_095_LOAD_MULTIPLIER
    gamma_095_max = static_max * GAMMA_095_LOAD_MULTIPLIER
    return {
        "schema": "shakebench_spike.stage2.grip_design.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "versions": _versions(),
        "selection": {
            "stage_a_path": "A2_force_limited_close",
            "position_command_after_bilateral_contact_m": 0.0,
            "force_limit_n_per_finger": GRIP_FORCE_LIMIT_N_PER_FINGER,
            "stock_position_actuator_kp_n_per_m": 1000.0,
            "stock_force_limit_n_per_finger": 20.0,
            "description": (
                "Command both fingers fully closed after bilateral contact; "
                "the symmetric actuator forcerange, not a frozen opening, caps force."
            ),
        },
        "derivation": {
            "cube_mass_range_kg": [MIN_CUBE_MASS_KG, MAX_CUBE_MASS_KG],
            "workpiece_side_mu": WORKPIECE_SIDE_MU,
            "finger_pad_mu": FINGER_PAD_MU,
            "limiting_mu": limiting_mu,
            "static_required_normal_force_n_per_finger": [static_min, static_max],
            "gamma_0p95_load_multiplier": GAMMA_095_LOAD_MULTIPLIER,
            "gamma_0p95_required_normal_force_n_per_finger": [
                gamma_095_min,
                gamma_095_max,
            ],
            "selected_force_over_worst_case_requirement": (
                GRIP_FORCE_LIMIT_N_PER_FINGER / gamma_095_max
            ),
        },
        "main_repository_difference": {
            "committed_gripper_contact_preload_m": 0.0003,
            "validation_upper_bound_m": 0.002,
            "stage2_gamma_0p95_reference_ee_wobble_m": 0.00227,
            "finding": (
                "The committed position latch preload and even its validation upper "
                "bound are below the reference Gamma=0.95 EE wobble. Stage A2 is an "
                "intentional diagnostic deviation and does not modify src/shakebench."
            ),
        },
        "single_seed_engineering_check_not_a_gate": check,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--timestep", type=float, default=2.0e-4)
    parser.add_argument("--window-s", type=float, default=6.0)
    parser.add_argument("--sample-hz", type=int, default=DIAGNOSTIC_SAMPLE_HZ)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "out" / "stage2")
    args = parser.parse_args()
    if args.sample_hz < 200:
        parser.error("--sample-hz must be at least 200")
    args.output.mkdir(parents=True, exist_ok=True)

    ladder = run_wobble_ladder(
        seed=args.seed,
        timestep=args.timestep,
        window_s=args.window_s,
        sample_hz=args.sample_hz,
    )
    (args.output / "ee_wobble_ladder.json").write_text(
        json.dumps(ladder, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    check = run_force_grip_design_check(
        seed=0,
        gamma=0.95,
        timestep=args.timestep,
        episode_s=16.0,
        sample_hz=args.sample_hz,
    )
    design = grip_design_document(check)
    (args.output / "grip_design.json").write_text(
        json.dumps(design, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"force-design check: success={check['success']} "
        f"slip_mm={1000 * check['max_grasp_slip_m']:.3f} "
        f"force_samples={check['post_latch_finger_force_n']['sample_count']}"
    )
    return 0 if check["success"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
