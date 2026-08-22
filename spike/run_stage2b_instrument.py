"""Stage B3 decoupled-table instrument; this file runs no task policy."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from env_shakedeck import ANCHOR, make_env
from metrics import eef_position_b, point_in_body_frame
from vibration import calibrated_vibration


GAMMAS = (0.15, 0.30, 0.50, 0.75, 0.95)
POLICY_HZ = 20
DIAGNOSTIC_HZ = 200
PHASE_OFFSET_RAD = 0.5 * np.pi


def _warning_count(env) -> int:
    return int(sum(item.number for item in env.sim.data.warning))


def measure(
    gamma: float,
    *,
    seed: int,
    timestep_s: float,
    window_s: float,
) -> dict:
    physics_hz = int(round(1.0 / timestep_s))
    if physics_hz % DIAGNOSTIC_HZ:
        raise ValueError("physics frequency must divide 200 Hz exactly")
    base_vibration, calibration = calibrated_vibration(
        gamma,
        seed=seed,
        physics_hz=physics_hz,
        episode_s=16.0,
    )
    table_vibration, table_calibration = calibrated_vibration(
        gamma,
        seed=seed,
        physics_hz=physics_hz,
        episode_s=16.0,
        phase_offset_rad=PHASE_OFFSET_RAD,
    )
    steps = int(round(window_s * POLICY_HZ))
    env = make_env(
        seed=seed,
        physics_timestep=timestep_s,
        motion_sampler=base_vibration.sample,
        table_motion_sampler=table_vibration.sample,
        control_freq=POLICY_HZ,
        horizon=steps + 1,
    )
    try:
        base_id = env.sim.model.body_name2id("robot0_base")
        cube_id = env.sim.model.body_name2id("cube_main")
        initial_eef_b = eef_position_b(env).copy()
        initial_cube_t = point_in_body_frame(
            env,
            env.table_deck_body_id,
            np.asarray(env.sim.data.body_xpos[cube_id]).copy(),
        )
        initial_table_b = point_in_body_frame(
            env,
            base_id,
            np.asarray(env.sim.data.body_xpos[env.table_deck_body_id]).copy(),
        )
        decimation = physics_hz // DIAGNOSTIC_HZ
        physics_steps = 0
        ee_wobble = []
        object_table_slip = []
        table_base_motion = []
        base_weld_error = []
        table_weld_error = []

        def sample(stepped_env) -> None:
            nonlocal physics_steps
            physics_steps += 1
            if physics_steps % decimation:
                return
            cube_w = np.asarray(stepped_env.sim.data.body_xpos[cube_id]).copy()
            cube_t = point_in_body_frame(
                stepped_env, stepped_env.table_deck_body_id, cube_w
            )
            table_w = np.asarray(
                stepped_env.sim.data.body_xpos[stepped_env.table_deck_body_id]
            ).copy()
            table_b = point_in_body_frame(stepped_env, base_id, table_w)
            ee_wobble.append(
                float(np.linalg.norm(eef_position_b(stepped_env) - initial_eef_b))
            )
            object_table_slip.append(float(np.linalg.norm(cube_t - initial_cube_t)))
            table_base_motion.append(float(np.linalg.norm(table_b - initial_table_b)))
            base_command, _ = stepped_env.commanded_deck_pose()
            table_command, _ = stepped_env.commanded_table_pose()
            base_actual = np.asarray(
                stepped_env.sim.data.site_xpos[stepped_env.deck_site_id]
            )
            table_actual = np.asarray(
                stepped_env.sim.data.site_xpos[stepped_env.table_deck_site_id]
            )
            base_weld_error.append(float(np.linalg.norm(base_actual - base_command)))
            table_weld_error.append(float(np.linalg.norm(table_actual - table_command)))

        env.physics_step_callback = sample
        action = np.zeros(env.action_dim, dtype=np.float64)
        for _ in range(steps):
            env.step(action)
        return {
            "gamma": gamma,
            "seed": seed,
            "measurement_window_s": window_s,
            "physics_timestep_s": timestep_s,
            "physics_frequency_hz": physics_hz,
            "policy_frequency_hz": POLICY_HZ,
            "diagnostic_frequency_hz": DIAGNOSTIC_HZ,
            "sample_count": len(ee_wobble),
            "base_frame_ee_wobble_m": max(ee_wobble),
            "table_frame_object_slip_m": max(object_table_slip),
            "base_frame_table_motion_m": max(table_base_motion),
            "base_weld_tracking_error_peak_m": max(base_weld_error),
            "table_weld_tracking_error_peak_m": max(table_weld_error),
            "mujoco_warning_count": _warning_count(env),
            "mocap_body_count": int(env.sim.model.nmocap),
            "calibration": calibration,
            "table_calibration": table_calibration,
        }
    finally:
        env.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--timestep", type=float, default=2.0e-4)
    parser.add_argument("--window-s", type=float, default=6.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "out" / "stage2b" / "decoupled_instrument",
    )
    args = parser.parse_args()
    rows = []
    for gamma in GAMMAS:
        row = measure(
            gamma,
            seed=args.seed,
            timestep_s=args.timestep,
            window_s=args.window_s,
        )
        rows.append(row)
        print(
            f"gamma={gamma:.2f} ee_mm={1000*row['base_frame_ee_wobble_m']:.4f} "
            f"object_table_mm={1000*row['table_frame_object_slip_m']:.4f} "
            f"table_base_mm={1000*row['base_frame_table_motion_m']:.4f}",
            flush=True,
        )
    result = {
        "schema": "shakebench_spike.stage2b.decoupled_instrument.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "B3",
        "policy_rollouts_run": False,
        "structural_change": (
            "The Panda base remains on deck; table is attached to an independent "
            "mocap+weld support with equal spectral amplitudes and a uniform pi/2 "
            "phase offset on every spectral line."
        ),
        "phase_offset_rad": PHASE_OFFSET_RAD,
        "excluded_from_b1_b2_and_c_gates": True,
        "rows": rows,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "decoupled_instrument.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
