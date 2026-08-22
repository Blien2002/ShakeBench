"""Stage 1 instrument for the compliant-worktable ShakeBench spike.

This runner deliberately executes no task policy.  It enforces the 1a gate
before producing the eight-configuration, five-Gamma stage 1b sweep.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform

import mujoco
import numpy as np
import robosuite

from env_shakedeck import make_env
from isolator import IsolatorParameters, absolute_transmissibility
from metrics import eef_position_b, point_in_body_frame
from vibration import calibrated_vibration


SEED = 17
TIMESTEP_S = 2.0e-4
POLICY_HZ = 20
DIAGNOSTIC_HZ = 200
SWEEP_WINDOW_S = 6.0
CALIBRATION_EPISODE_S = 16.0
GAMMAS = (0.15, 0.30, 0.50, 0.75, 0.95)
MUS = (0.2, 0.3, 0.5, 0.8, 1.5)
ISOLATOR_CONFIGS = (
    ("fn30_zeta0p10", 30.0, 0.10),
    ("fn15_zeta0p10", 15.0, 0.10),
    ("fn8_zeta0p10", 8.0, 0.10),
    ("fn5_zeta0p10", 5.0, 0.10),
    ("fn3_zeta0p10", 3.0, 0.10),
    ("fn5_zeta0p05", 5.0, 0.05),
    ("fn5_zeta0p20", 5.0, 0.20),
)
OUTPUT_ROOT = Path(__file__).resolve().parent / "out" / "iso"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _warning_count(env) -> int:
    return int(sum(item.number for item in env.sim.data.warning))


def _peak_about_mean(trace: list[np.ndarray]) -> float:
    values = np.asarray(trace, dtype=np.float64)
    center = np.mean(values, axis=0)
    return float(np.max(np.linalg.norm(values - center, axis=1)))


def _half_peak_to_peak(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    return 0.5 * float(np.max(array) - np.min(array))


class HarmonicMotion:
    def __init__(self, frequency_hz: float, amplitude_m: float):
        self.omega = 2.0 * np.pi * frequency_hz
        self.amplitude_m = amplitude_m

    def sample(self, time_s: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        q = np.zeros(6, dtype=np.float64)
        qd = np.zeros(6, dtype=np.float64)
        qdd = np.zeros(6, dtype=np.float64)
        angle = self.omega * time_s
        q[0] = self.amplitude_m * np.sin(angle)
        qd[0] = self.amplitude_m * self.omega * np.cos(angle)
        qdd[0] = -(self.amplitude_m * self.omega**2) * np.sin(angle)
        return q, qd, qdd


def _run_policy_steps(env, duration_s: float) -> None:
    action = np.zeros(env.action_dim, dtype=np.float64)
    for _ in range(int(round(duration_s * POLICY_HZ))):
        env.step(action)


def measure_transmissibility(
    parameters: IsolatorParameters,
    *,
    excitation_frequency_hz: float = 5.0,
    excitation_amplitude_m: float = 5.0e-4,
    duration_s: float = 8.0,
    analysis_window_s: float = 2.0,
) -> dict:
    motion = HarmonicMotion(excitation_frequency_hz, excitation_amplitude_m)
    env = make_env(
        seed=SEED,
        physics_timestep=TIMESTEP_S,
        motion_sampler=motion.sample,
        control_freq=POLICY_HZ,
        horizon=int(round(duration_s * POLICY_HZ)) + 1,
        table_isolator=parameters,
    )
    try:
        physics_hz = int(round(1.0 / TIMESTEP_S))
        decimation = physics_hz // DIAGNOSTIC_HZ
        physics_steps = 0
        times: list[float] = []
        deck_x: list[float] = []
        table_x: list[float] = []
        command_x: list[float] = []
        deck_site_id = env.deck_site_id
        table_body_id = env.sim.model.body_name2id("table")

        def sample(stepped_env) -> None:
            nonlocal physics_steps
            physics_steps += 1
            if physics_steps % decimation:
                return
            time_s = float(stepped_env.sim.data.time)
            times.append(time_s)
            deck_x.append(float(stepped_env.sim.data.site_xpos[deck_site_id][0]))
            table_x.append(float(stepped_env.sim.data.body_xpos[table_body_id][0]))
            command_x.append(float(motion.sample(time_s)[0][0]))

        env.physics_step_callback = sample
        _run_policy_steps(env, duration_s)
        keep = np.asarray(times) >= duration_s - analysis_window_s
        deck_analysis = np.asarray(deck_x)[keep].tolist()
        table_analysis = np.asarray(table_x)[keep].tolist()
        command_analysis = np.asarray(command_x)[keep].tolist()
        deck_amplitude = _half_peak_to_peak(deck_analysis)
        table_amplitude = _half_peak_to_peak(table_analysis)
        measured = table_amplitude / deck_amplitude
        analytic = absolute_transmissibility(
            parameters.natural_frequency_hz,
            parameters.damping_ratio,
            excitation_frequency_hz,
        )
        relative_error = abs(measured - analytic) / analytic
        return {
            "parameters": parameters.as_dict(),
            "excitation_frequency_hz": excitation_frequency_hz,
            "excitation_amplitude_m": excitation_amplitude_m,
            "duration_s": duration_s,
            "analysis_window_s": analysis_window_s,
            "diagnostic_frequency_hz": DIAGNOSTIC_HZ,
            "analysis_sample_count": len(deck_analysis),
            "command_amplitude_measured_m": _half_peak_to_peak(command_analysis),
            "deck_amplitude_measured_m": deck_amplitude,
            "table_absolute_amplitude_measured_m": table_amplitude,
            "transmissibility_measured": measured,
            "transmissibility_analytic": analytic,
            "relative_error": relative_error,
            "pass": relative_error < 0.05,
            "limit_relative_error_lt": 0.05,
            "mujoco_warning_count": _warning_count(env),
        }
    finally:
        env.close()


def measure_static_height(
    parameters: IsolatorParameters | None,
    *,
    duration_s: float = 3.0,
    analysis_window_s: float = 0.5,
) -> dict:
    env = make_env(
        seed=SEED,
        physics_timestep=TIMESTEP_S,
        motion_sampler=None,
        control_freq=POLICY_HZ,
        horizon=int(round(duration_s * POLICY_HZ)) + 1,
        table_isolator=parameters,
    )
    try:
        physics_hz = int(round(1.0 / TIMESTEP_S))
        decimation = physics_hz // DIAGNOSTIC_HZ
        physics_steps = 0
        samples: list[tuple[float, float]] = []
        table_geom_id = env.sim.model.geom_name2id("table_collision")

        def sample(stepped_env) -> None:
            nonlocal physics_steps
            physics_steps += 1
            if physics_steps % decimation:
                return
            samples.append(
                (
                    float(stepped_env.sim.data.time),
                    float(stepped_env.sim.data.geom_xpos[table_geom_id][2]),
                )
            )

        env.physics_step_callback = sample
        _run_policy_steps(env, duration_s)
        heights = [height for time_s, height in samples if time_s >= duration_s - analysis_window_s]
        table_id = env.sim.model.body_name2id("table")
        result = {
            "height_median_m": float(np.median(heights)),
            "height_peak_to_peak_m": float(np.ptp(heights)),
            "analysis_sample_count": len(heights),
            "mujoco_warning_count": _warning_count(env),
            "compiled_table_mass_kg": float(env.sim.model.body_mass[table_id]),
            "compiled_table_inertia_kg_m2": np.asarray(
                env.sim.model.body_inertia[table_id]
            ).tolist(),
        }
        if parameters is not None:
            result["parameters"] = parameters.as_dict()
            result["compiled_joints"] = {
                name: {
                    "stiffness": float(
                        env.sim.model.jnt_stiffness[env.sim.model.joint_name2id(name)]
                    ),
                    "damping": float(
                        env.sim.model.dof_damping[
                            env.sim.model.jnt_dofadr[env.sim.model.joint_name2id(name)]
                        ]
                    ),
                    "springref": float(
                        env.sim.model.qpos_spring[
                            env.sim.model.jnt_qposadr[env.sim.model.joint_name2id(name)]
                        ]
                    ),
                }
                for name in (
                    "table_iso_tx",
                    "table_iso_ty",
                    "table_iso_tz",
                    "table_iso_rx",
                    "table_iso_ry",
                    "table_iso_rz",
                )
            }
        return result
    finally:
        env.close()


def measure_spectral(
    *,
    gamma: float,
    parameters: IsolatorParameters | None,
    capture_qpos: bool = False,
    cube_table_sliding_mu: float = 1.5,
) -> dict:
    physics_hz = int(round(1.0 / TIMESTEP_S))
    vibration, calibration = calibrated_vibration(
        gamma,
        seed=SEED,
        physics_hz=physics_hz,
        episode_s=CALIBRATION_EPISODE_S,
    )
    env = make_env(
        seed=SEED,
        physics_timestep=TIMESTEP_S,
        motion_sampler=vibration.sample,
        control_freq=POLICY_HZ,
        horizon=int(round(SWEEP_WINDOW_S * POLICY_HZ)) + 1,
        table_isolator=parameters,
        cube_table_sliding_mu=cube_table_sliding_mu,
    )
    try:
        decimation = physics_hz // DIAGNOSTIC_HZ
        physics_steps = 0
        base_id = env.sim.model.body_name2id("robot0_base")
        table_id = env.sim.model.body_name2id("table")
        cube_id = env.sim.model.body_name2id("cube_main")
        deck_site_id = env.deck_site_id
        initial_cube_t = point_in_body_frame(
            env, table_id, np.asarray(env.sim.data.body_xpos[cube_id]).copy()
        )
        table_b_trace: list[np.ndarray] = []
        table_w_trace: list[np.ndarray] = []
        eef_b_trace: list[np.ndarray] = []
        command_trace: list[np.ndarray] = []
        deck_w_trace: list[np.ndarray] = []
        object_slip: list[float] = []
        qpos_trace: list[np.ndarray] = []

        def sample(stepped_env) -> None:
            nonlocal physics_steps
            physics_steps += 1
            if physics_steps % decimation:
                return
            table_w = np.asarray(stepped_env.sim.data.body_xpos[table_id]).copy()
            cube_w = np.asarray(stepped_env.sim.data.body_xpos[cube_id]).copy()
            table_b_trace.append(point_in_body_frame(stepped_env, base_id, table_w))
            table_w_trace.append(table_w)
            eef_b_trace.append(eef_position_b(stepped_env).copy())
            cube_t = point_in_body_frame(stepped_env, table_id, cube_w)
            object_slip.append(float(np.linalg.norm(cube_t - initial_cube_t)))
            command_trace.append(
                np.asarray(vibration.sample(float(stepped_env.sim.data.time))[0][:3]).copy()
            )
            deck_w_trace.append(
                np.asarray(stepped_env.sim.data.site_xpos[deck_site_id]).copy()
            )
            if capture_qpos:
                qpos_trace.append(np.asarray(stepped_env.sim.data.qpos).copy())

        env.physics_step_callback = sample
        _run_policy_steps(env, SWEEP_WINDOW_S)
        result = {
            "gamma": gamma,
            "seed": SEED,
            "measurement_window_s": SWEEP_WINDOW_S,
            "physics_timestep_s": TIMESTEP_S,
            "physics_frequency_hz": physics_hz,
            "diagnostic_frequency_hz": DIAGNOSTIC_HZ,
            "sample_count": len(table_b_trace),
            "base_frame_table_motion_peak_about_mean_m": _peak_about_mean(table_b_trace),
            "table_absolute_motion_peak_about_mean_m": _peak_about_mean(table_w_trace),
            "base_frame_ee_wobble_peak_about_mean_m": _peak_about_mean(eef_b_trace),
            "table_frame_object_slip_max_m": max(object_slip),
            "deck_command_peak_about_mean_m": _peak_about_mean(command_trace),
            "deck_actual_peak_about_mean_m": _peak_about_mean(deck_w_trace),
            "mujoco_warning_count": _warning_count(env),
            "calibration": calibration,
            "isolator": parameters.as_dict() if parameters is not None else None,
            "cube_table_sliding_mu": cube_table_sliding_mu,
        }
        if capture_qpos:
            result["_qpos_trace"] = np.asarray(qpos_trace)
        return result
    finally:
        env.close()


def _relative_difference(value: float, reference: float) -> float:
    return abs(value - reference) / max(abs(reference), np.finfo(float).eps)


def run_selfcheck(output_root: Path) -> dict:
    transfer_rows = []
    for name, natural_frequency_hz, damping_ratio in ISOLATOR_CONFIGS:
        print(f"1a transmissibility {name}", flush=True)
        row = measure_transmissibility(
            IsolatorParameters(natural_frequency_hz, damping_ratio)
        )
        row["configuration_name"] = name
        transfer_rows.append(row)

    print("1a static-height checks", flush=True)
    rigid_height = measure_static_height(None)
    height_rows = []
    for name, natural_frequency_hz, damping_ratio in ISOLATOR_CONFIGS:
        row = measure_static_height(
            IsolatorParameters(natural_frequency_hz, damping_ratio)
        )
        error = abs(row["height_median_m"] - rigid_height["height_median_m"])
        row.update(
            {
                "configuration_name": name,
                "height_error_vs_rigid_m": error,
                "pass": error < 5.0e-5,
                "limit_abs_error_m_lt": 5.0e-5,
            }
        )
        height_rows.append(row)

    print("1a f_n=30 Hz rigid-degeneration checks", flush=True)
    rigid = measure_spectral(gamma=0.5, parameters=None)
    high_a = measure_spectral(
        gamma=0.5,
        parameters=IsolatorParameters(30.0, 0.10),
        capture_qpos=True,
    )
    high_b = measure_spectral(
        gamma=0.5,
        parameters=IsolatorParameters(30.0, 0.10),
        capture_qpos=True,
    )
    compared_metrics = (
        "table_absolute_motion_peak_about_mean_m",
        "base_frame_ee_wobble_peak_about_mean_m",
        "table_frame_object_slip_max_m",
        "deck_command_peak_about_mean_m",
        "deck_actual_peak_about_mean_m",
    )
    comparisons = {
        metric: {
            "rigid": rigid[metric],
            "fn30": high_a[metric],
            "relative_difference": _relative_difference(high_a[metric], rigid[metric]),
            "pass": _relative_difference(high_a[metric], rigid[metric]) < 0.10,
            "limit_relative_difference_lt": 0.10,
        }
        for metric in compared_metrics
    }
    relative_motion_check = {
        "fn30_base_frame_table_motion_m": high_a[
            "base_frame_table_motion_peak_about_mean_m"
        ],
        "deck_command_peak_m": high_a["deck_command_peak_about_mean_m"],
        "ratio": high_a["base_frame_table_motion_peak_about_mean_m"]
        / high_a["deck_command_peak_about_mean_m"],
        "pass": high_a["base_frame_table_motion_peak_about_mean_m"]
        < 0.10 * high_a["deck_command_peak_about_mean_m"],
        "limit_ratio_lt": 0.10,
    }
    determinism_error = float(
        np.max(np.abs(high_a.pop("_qpos_trace") - high_b.pop("_qpos_trace")))
    )
    warning_count = sum(row["mujoco_warning_count"] for row in transfer_rows)
    warning_count += rigid_height["mujoco_warning_count"]
    warning_count += sum(row["mujoco_warning_count"] for row in height_rows)
    warning_count += rigid["mujoco_warning_count"]
    warning_count += high_a["mujoco_warning_count"] + high_b["mujoco_warning_count"]

    checks = {
        "transmissibility_vs_analytic": {
            "pass": all(row["pass"] for row in transfer_rows),
            "rows": transfer_rows,
        },
        "static_table_height": {
            "pass": all(row["pass"] for row in height_rows),
            "rigid": rigid_height,
            "rows": height_rows,
        },
        "fn30_degenerates_to_rigid": {
            "pass": all(row["pass"] for row in comparisons.values())
            and relative_motion_check["pass"],
            "metric_comparisons": comparisons,
            "relative_table_motion": relative_motion_check,
            "rigid": rigid,
            "fn30": high_a,
        },
        "determinism": {
            "pass": determinism_error == 0.0,
            "max_abs_qpos_difference": determinism_error,
            "expected": 0.0,
        },
    }
    result = {
        "schema": "shakebench_spike.iso.transmissibility_check.v1",
        "created_utc": _now(),
        "policy_rollouts_run": False,
        "versions": {
            "python": platform.python_version(),
            "robosuite": robosuite.__version__,
            "mujoco": mujoco.__version__,
            "numpy": np.__version__,
        },
        "configuration": {
            "seed": SEED,
            "physics_timestep_s": TIMESTEP_S,
            "diagnostic_frequency_hz": DIAGNOSTIC_HZ,
            "gravity_compensation": (
                "table_iso_tz springref = +(M_table + M_payload) g / k"
            ),
        },
        "checks": checks,
        "mujoco_warnings": {
            "diagnostic_only": True,
            "total_count": warning_count,
        },
        "all_passed": all(check["pass"] for check in checks.values()),
    }
    _write(output_root / "transmissibility_check.json", result)
    return result


def run_sweep(output_root: Path) -> dict:
    selfcheck_path = output_root / "transmissibility_check.json"
    if not selfcheck_path.is_file():
        raise RuntimeError("stage 1a output is missing")
    selfcheck = json.loads(selfcheck_path.read_text(encoding="utf-8"))
    if not selfcheck.get("all_passed"):
        raise RuntimeError("stage 1a gate did not pass; stage 1b is forbidden")

    configurations = [("RIGID", None)] + [
        (name, IsolatorParameters(frequency, damping))
        for name, frequency, damping in ISOLATOR_CONFIGS
    ]
    rows = []
    for name, parameters in configurations:
        for gamma in GAMMAS:
            print(f"1b {name} gamma={gamma:.2f}", flush=True)
            row = measure_spectral(gamma=gamma, parameters=parameters)
            row["configuration_name"] = name
            rows.append(row)
            print(
                f"  table_base_mm={1000*row['base_frame_table_motion_peak_about_mean_m']:.4f} "
                f"ee_base_mm={1000*row['base_frame_ee_wobble_peak_about_mean_m']:.4f} "
                f"object_table_um={1e6*row['table_frame_object_slip_max_m']:.4f}",
                flush=True,
            )
    result = {
        "schema": "shakebench_spike.iso.sweep.v1",
        "created_utc": _now(),
        "policy_rollouts_run": False,
        "authorized_design_change": (
            "six-DOF compliant worktable support; cube-table and finger-pad contact "
            "parameters unchanged"
        ),
        "configuration": {
            "seed": SEED,
            "physics_timestep_s": TIMESTEP_S,
            "measurement_window_s": SWEEP_WINDOW_S,
            "diagnostic_frequency_hz": DIAGNOSTIC_HZ,
            "gammas": list(GAMMAS),
            "configuration_count": len(configurations),
        },
        "rows": rows,
        "mujoco_warning_count": sum(row["mujoco_warning_count"] for row in rows),
    }
    _write(output_root / "iso_sweep.json", result)
    return result


def select_operating_points(output_root: Path) -> dict:
    sweep_path = output_root / "iso_sweep.json"
    if not sweep_path.is_file():
        raise RuntimeError("stage 1b output is missing")
    sweep = json.loads(sweep_path.read_text(encoding="utf-8"))
    gamma_half = {
        row["configuration_name"]: row
        for row in sweep["rows"]
        if row["gamma"] == 0.5
    }
    candidates = [
        row
        for name, row in gamma_half.items()
        if name != "RIGID"
        and row["base_frame_table_motion_peak_about_mean_m"] <= 0.012
    ]

    def choose(low_m: float, high_m: float, target_m: float) -> tuple[dict, bool]:
        in_range = [
            row
            for row in candidates
            if low_m
            <= row["base_frame_table_motion_peak_about_mean_m"]
            <= high_m
        ]
        pool = in_range or candidates
        selected = min(
            pool,
            key=lambda row: abs(
                row["base_frame_table_motion_peak_about_mean_m"] - target_m
            ),
        )
        return selected, bool(in_range)

    mild, mild_in_range = choose(0.0008, 0.0020, 0.0015)
    strong, strong_in_range = choose(0.0035, 0.0080, 0.0050)

    def point(
        label: str,
        row: dict,
        interval_m: tuple[float, float] | None,
        target_m: float | None,
        in_range: bool,
    ) -> dict:
        value = row["base_frame_table_motion_peak_about_mean_m"]
        return {
            "label": label,
            "configuration_name": row["configuration_name"],
            "isolator": row["isolator"],
            "gamma_selection": 0.5,
            "base_frame_table_motion_m": value,
            "target_m": target_m,
            "allowed_interval_m": list(interval_m) if interval_m else None,
            "within_predeclared_interval": in_range,
            "deviation_from_interval_m": (
                0.0
                if interval_m is None or in_range
                else min(abs(value - interval_m[0]), abs(value - interval_m[1]))
            ),
            "under_12_mm_limit": value <= 0.012,
        }

    result = {
        "schema": "shakebench_spike.iso.operating_points.v1",
        "created_utc": _now(),
        "selection_was_predeclared": True,
        "selection_metric": "Gamma=0.50 base-frame table motion peak about mean",
        "points": {
            "RIGID": point("RIGID", gamma_half["RIGID"], None, None, True),
            "MILD": point(
                "MILD", mild, (0.0008, 0.0020), 0.0015, mild_in_range
            ),
            "STRONG": point(
                "STRONG", strong, (0.0035, 0.0080), 0.0050, strong_in_range
            ),
        },
        "excluded_over_12_mm_at_gamma_0p5": [
            {
                "configuration_name": name,
                "base_frame_table_motion_m": row[
                    "base_frame_table_motion_peak_about_mean_m"
                ],
            }
            for name, row in gamma_half.items()
            if name != "RIGID"
            and row["base_frame_table_motion_peak_about_mean_m"] > 0.012
        ],
    }
    _write(output_root / "operating_points.json", result)
    return result


def run_mu_sweep(output_root: Path) -> dict:
    operating_path = output_root / "operating_points.json"
    if not operating_path.is_file():
        raise RuntimeError("stage 1c operating points are missing")
    operating = json.loads(operating_path.read_text(encoding="utf-8"))
    strong_data = operating["points"]["STRONG"]["isolator"]
    strong = IsolatorParameters(
        strong_data["natural_frequency_hz"], strong_data["damping_ratio"]
    )
    rows = []
    for label, parameters in (("RIGID", None), ("STRONG", strong)):
        for mu in MUS:
            for gamma in GAMMAS:
                print(f"mu sweep {label} mu={mu:.1f} gamma={gamma:.2f}", flush=True)
                row = measure_spectral(
                    gamma=gamma,
                    parameters=parameters,
                    cube_table_sliding_mu=mu,
                )
                row["operating_point"] = label
                rows.append(row)
                print(
                    f"  object_table_mm={1000*row['table_frame_object_slip_max_m']:.4f} "
                    f"table_base_mm={1000*row['base_frame_table_motion_peak_about_mean_m']:.4f}",
                    flush=True,
                )

    selection_rows = [
        row
        for row in rows
        if row["operating_point"] == "STRONG" and row["gamma"] == 0.5
    ]
    exceeding = [
        row for row in selection_rows if row["table_frame_object_slip_max_m"] > 0.0005
    ]
    if exceeding:
        selected = max(exceeding, key=lambda row: row["cube_table_sliding_mu"])
        selection_case = "largest_mu_exceeding_0p5_mm"
    else:
        selected = min(selection_rows, key=lambda row: row["cube_table_sliding_mu"])
        selection_case = "none_exceeded_0p5_mm_friction_not_limiting"
    result = {
        "schema": "shakebench_spike.iso.mu_sweep.v1",
        "created_utc": _now(),
        "policy_rollouts_run": False,
        "authorized_design_change": "cube-table explicit-pair sliding friction only",
        "unchanged": [
            "Panda finger-pad friction",
            "Panda finger-pad solref",
            "cube-table solref",
            "contact margin and gap",
        ],
        "configuration": {
            "seed": SEED,
            "physics_timestep_s": TIMESTEP_S,
            "measurement_window_s": SWEEP_WINDOW_S,
            "diagnostic_frequency_hz": DIAGNOSTIC_HZ,
            "mus": list(MUS),
            "gammas": list(GAMMAS),
            "operating_points": ["RIGID", "STRONG"],
        },
        "rows": rows,
        "selection": {
            "rule": (
                "largest mu whose STRONG/Gamma=0.50 table-frame object slip "
                "first exceeds 0.5 mm"
            ),
            "case": selection_case,
            "selected_mu": selected["cube_table_sliding_mu"],
            "selected_slip_m": selected["table_frame_object_slip_max_m"],
            "selected_row": selected,
        },
        "mujoco_warning_count": sum(row["mujoco_warning_count"] for row in rows),
    }
    _write(output_root / "mu_sweep.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--select", action="store_true")
    parser.add_argument("--mu-sweep", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    if sum((args.self_check, args.sweep, args.all, args.select, args.mu_sweep)) != 1:
        parser.error(
            "select exactly one of --self-check, --sweep, --select, --mu-sweep, or --all"
        )
    if args.self_check or args.all:
        selfcheck = run_selfcheck(args.output)
        print(json.dumps({"all_passed": selfcheck["all_passed"]}, indent=2))
        if not selfcheck["all_passed"]:
            print("stage 1a failed; stopping before stage 1b", flush=True)
            return 2
    if args.sweep or args.all:
        sweep = run_sweep(args.output)
        print(
            json.dumps(
                {
                    "row_count": len(sweep["rows"]),
                    "mujoco_warning_count": sweep["mujoco_warning_count"],
                },
                indent=2,
            )
        )
    if args.select or args.all:
        selected = select_operating_points(args.output)
        print(json.dumps(selected["points"], indent=2))
    if args.mu_sweep or args.all:
        mu = run_mu_sweep(args.output)
        print(json.dumps(mu["selection"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
