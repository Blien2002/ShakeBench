"""Run the minimal E1/E2/E3 vibration-arm diagnostic.

Example:
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python -m robosuite.scripts.shakebench_probe_arm_effect
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

import robosuite
import robosuite.utils.transform_utils as T
from robosuite.controllers import load_composite_controller_config
from robosuite.utils.shakebench_calibration import calibrate_gamma, level_scale_for_gamma
from robosuite.utils.shakebench_excitation import MotionSample, build_excitation_program, quintic_smoothstep
from robosuite.utils.shakebench_tasks import TaskSpec

SEEDS = (17, 18, 19)
GAMMA = 0.60
MEASUREMENT_S = 6.0
CONTROL_HZ = 20.0


class GatedProgram:
    """Keep the existing program at zero until the prepared measurement starts."""

    def __init__(self, program):
        self.program = program
        self.start_s: float | None = None

    def evaluate(self, time_s):
        time = np.asarray(time_s, dtype=float)
        if self.start_s is None:
            zeros = np.zeros(time.shape + (6,), dtype=float)
            return MotionSample(q=zeros, qdot=zeros, qdd=zeros)
        return self.program.evaluate(np.maximum(time - self.start_s, 0.0))


def _rotation_angle(rotation: np.ndarray) -> float:
    return float(np.arccos(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)))


def _pose_in(env, body_id: int, frame_id: int) -> tuple[np.ndarray, np.ndarray]:
    frame_position = np.asarray(env.sim.data.xpos[frame_id], dtype=float)
    frame_rotation = np.asarray(env.sim.data.xmat[frame_id], dtype=float).reshape(3, 3)
    position = frame_rotation.T @ (np.asarray(env.sim.data.xpos[body_id], dtype=float) - frame_position)
    rotation = frame_rotation.T @ np.asarray(env.sim.data.xmat[body_id], dtype=float).reshape(3, 3)
    return position, rotation


def _contact_pairs(env) -> set[frozenset[int]]:
    return {
        frozenset((int(env.sim.data.contact[index].geom1), int(env.sim.data.contact[index].geom2)))
        for index in range(env.sim.data.ncon)
    }


def _has_contact(env, names_a, names_b) -> bool:
    ids_a = {env.sim.model.geom_name2id(name) for name in names_a}
    ids_b = {env.sim.model.geom_name2id(name) for name in names_b}
    return any(pair & ids_a and pair & ids_b for pair in _contact_pairs(env))


def _eef_pose_base(env) -> tuple[np.ndarray, np.ndarray]:
    robot = env.robots[0]
    site_id = robot.eef_site_id["right"]
    base_position = np.asarray(env.sim.data.xpos[env.robot_base_body_id], dtype=float)
    base_rotation = np.asarray(env.sim.data.xmat[env.robot_base_body_id], dtype=float).reshape(3, 3)
    position = base_rotation.T @ (np.asarray(env.sim.data.site_xpos[site_id], dtype=float) - base_position)
    rotation = base_rotation.T @ np.asarray(env.sim.data.site_xmat[site_id], dtype=float).reshape(3, 3)
    return position, rotation


def _action(env, target_position: np.ndarray, target_rotation: np.ndarray, gripper: float) -> np.ndarray:
    position, rotation = _eef_pose_base(env)
    action = np.zeros(7, dtype=float)
    action[:3] = np.clip(4.0 * (target_position - position), -0.70, 0.70)
    action[3:6] = np.clip(2.0 * T.get_orientation_error(T.mat2quat(target_rotation), T.mat2quat(rotation)), -0.70, 0.70)
    action[6] = gripper
    return action


def _drive(env, target_position, target_rotation, gripper, steps: int) -> None:
    for _ in range(steps):
        env.step(_action(env, target_position, target_rotation, gripper))


def _snapshot(env, *, include_object=False) -> dict:
    robot = env.robots[0]
    result = {
        "qpos": np.asarray(env.sim.data.qpos, dtype=float).copy(),
        "qvel": np.asarray(env.sim.data.qvel, dtype=float).copy(),
        "eef_base_position": _eef_pose_base(env)[0].copy(),
        "gripper_qpos": np.asarray(
            [env.sim.data.qpos[index] for index in robot._ref_gripper_joint_pos_indexes["right"]], dtype=float
        ),
    }
    if include_object:
        result["object_gripper_position"], result["object_gripper_rotation"] = _pose_in(
            env, env.can_body_id, env.sim.model.body_name2id(env.gripper_body_name)
        )
    return result


def _snapshot_delta(left: dict, right: dict) -> dict[str, float]:
    values = {}
    for key in left:
        if key in right:
            values[key] = float(np.max(np.abs(np.asarray(left[key]) - np.asarray(right[key]))))
    return values


@dataclass
class TorqueStats:
    count: int = 0
    total: np.ndarray = field(default_factory=lambda: np.zeros(7))
    square_total: np.ndarray = field(default_factory=lambda: np.zeros(7))

    def add(self, torque: np.ndarray) -> None:
        self.count += 1
        self.total += torque
        self.square_total += torque * torque

    def summary(self) -> dict:
        mean = self.total / self.count
        return {
            "mean_Nm": mean.tolist(),
            "std_Nm": np.sqrt(np.maximum(self.square_total / self.count - mean * mean, 0.0)).tolist(),
            "rms_Nm": np.sqrt(self.square_total / self.count).tolist(),
        }


class Recorder:
    def __init__(
        self, env, experiment: str, reference: Callable[[float], tuple[np.ndarray, np.ndarray]], duration_s: float
    ):
        self.env, self.experiment, self.reference, self.duration_s = env, experiment, reference, duration_s
        self.start_s: float | None = None
        self.rows: list[dict] = []
        self.stats = TorqueStats()
        self.initial_object: tuple[np.ndarray, np.ndarray] | None = None
        self.drop_elapsed_s = 0.0
        self.dropped = False
        self.external_contact = False
        self._last_saved = -np.inf
        self.base_initial: tuple[np.ndarray, np.ndarray] | None = None

    def start(self, time_s: float) -> None:
        self.start_s = time_s
        self.base_initial = (
            np.asarray(self.env.sim.data.xpos[self.env.robot_base_body_id], dtype=float).copy(),
            np.asarray(self.env.sim.data.xmat[self.env.robot_base_body_id], dtype=float).reshape(3, 3).copy(),
        )
        if self.experiment == "E3":
            self.initial_object = _pose_in(
                self.env, self.env.can_body_id, self.env.sim.model.body_name2id(self.env.gripper_body_name)
            )

    def __call__(self, sample_time_s: float, policy_step=False) -> None:
        if self.start_s is None:
            return
        time_s = sample_time_s - self.start_s
        if time_s < -1e-12 or time_s > self.duration_s + self.env.model_timestep / 2:
            return
        target_position, target_rotation = self.reference(time_s)
        position, rotation = _eef_pose_base(self.env)
        error = target_position - position
        torque = np.asarray(self.env.sim.data.qfrc_actuator[self.env.robots[0]._ref_joint_vel_indexes], dtype=float)[:7]
        if time_s >= min(1.0, self.duration_s) - self.env.model_timestep:
            self.stats.add(torque)
        finger_object = _has_contact(self.env, self.env.finger_pad_geom_names, self.env.can.contact_geoms)
        external = _has_contact(
            self.env,
            self.env.finger_pad_geom_names,
            tuple(self.env.can.contact_geoms)
            + tuple(self.env.table_contact_geom_names)
            + tuple(self.env.target_collision_geom_names),
        )
        self.external_contact |= external
        object_position = np.full(3, np.nan)
        object_angle = np.nan
        if self.initial_object is not None:
            object_position, object_rotation = _pose_in(
                self.env, self.env.can_body_id, self.env.sim.model.body_name2id(self.env.gripper_body_name)
            )
            downward_m = float(np.asarray(self.env.sim.data.xpos[self.env.can_body_id])[2] - self._object_world_start_z)
            if not finger_object and downward_m < -0.020:
                self.drop_elapsed_s += self.env.model_timestep
            else:
                self.drop_elapsed_s = 0.0
            self.dropped |= self.drop_elapsed_s >= 0.1
            object_angle = _rotation_angle(self.initial_object[1].T @ object_rotation)
            object_position = object_position - self.initial_object[0]
        if time_s - self._last_saved < 0.005 - self.env.model_timestep / 2:
            return
        self._last_saved = time_s
        assert self.base_initial is not None
        base_position = np.asarray(self.env.sim.data.xpos[self.env.robot_base_body_id], dtype=float)
        base_rotation = np.asarray(self.env.sim.data.xmat[self.env.robot_base_body_id], dtype=float).reshape(3, 3)
        self.rows.append(
            {
                "experiment": self.experiment,
                "time_s": time_s,
                "reference_frame": "robot_base",
                "eef_error_mm": float(np.linalg.norm(error) * 1000),
                "eef_error_peak_component_mm": float(np.max(np.abs(error)) * 1000),
                "eef_orientation_error_deg": float(np.degrees(_rotation_angle(target_rotation.T @ rotation))),
                "base_displacement_mm": float(np.linalg.norm(base_position - self.base_initial[0]) * 1000),
                "base_rotation_deg": float(np.degrees(_rotation_angle(self.base_initial[1].T @ base_rotation))),
                "reference_x_m": target_position[0],
                "reference_y_m": target_position[1],
                "reference_z_m": target_position[2],
                "finger_object_contact": int(finger_object),
                "external_gripper_contact": int(external),
                "object_relative_drift_mm": float(np.linalg.norm(object_position) * 1000),
                "object_relative_angle_deg": float(np.degrees(object_angle)),
                "drop_confirmed": int(self.dropped),
                **{f"joint_{index + 1}_torque_Nm": torque[index] for index in range(7)},
            }
        )

    @property
    def _object_world_start_z(self) -> float:
        # This is assigned at measurement start, separately from the relative pose.
        return self._initial_object_world_z

    def set_object_world_start(self) -> None:
        self._initial_object_world_z = float(self.env.sim.data.xpos[self.env.can_body_id][2])

    def summary(self) -> dict:
        errors = np.asarray(
            [
                row["eef_error_mm"]
                for row in self.rows
                if row["time_s"] >= min(1.0, self.duration_s) - self.env.model_timestep
            ],
            dtype=float,
        )
        if not errors.size:
            errors = np.asarray([row["eef_error_mm"] for row in self.rows], dtype=float)
        drift = np.asarray([row["object_relative_drift_mm"] for row in self.rows], dtype=float)
        angles = np.asarray([row["object_relative_angle_deg"] for row in self.rows], dtype=float)
        if self.stats.count:
            torque = self.stats.summary()
        else:
            values = np.asarray(
                [[row[f"joint_{index}_torque_Nm"] for index in range(1, 8)] for row in self.rows], dtype=float
            )
            torque = {
                "mean_Nm": np.mean(values, axis=0).tolist(),
                "std_Nm": np.std(values, axis=0).tolist(),
                "rms_Nm": np.sqrt(np.mean(values * values, axis=0)).tolist(),
            }
        return {
            "eef_rms_error_mm_1_to_6_s": float(np.sqrt(np.mean(errors * errors))),
            "eef_peak_error_mm_1_to_6_s": float(np.max(errors)),
            "torque_1_to_6_s": torque,
            "external_gripper_contact": self.external_contact,
            "object_peak_relative_drift_mm_0_to_6_s": float(np.nanmax(drift)) if np.any(np.isfinite(drift)) else None,
            "object_peak_relative_angle_deg_0_to_6_s": (
                float(np.nanmax(angles)) if np.any(np.isfinite(angles)) else None
            ),
            "drop_confirmed": self.dropped,
        }


def _make_env(seed: int, program: GatedProgram):
    return robosuite.make(
        "VibrationPickPlace",
        robots="Panda",
        controller_configs=load_composite_controller_config(robot="Panda"),
        task=TaskSpec(object_id="wood_cube", surface_id="metal"),
        initialization_noise=None,
        use_camera_obs=False,
        use_object_obs=False,
        physics_profile="official",
        geometry_profile="direct_mount_v1",
        excitation_program=program,
        has_renderer=False,
        has_offscreen_renderer=False,
        hard_reset=False,
        horizon=100000,
        ignore_done=True,
        seed=seed,
    )


def _prepare_empty(env) -> tuple[np.ndarray, np.ndarray]:
    position, rotation = _eef_pose_base(env)
    target = position + np.array((0.0, 0.0, 0.12))
    _drive(env, target, rotation, -1.0, 80)
    if np.linalg.norm(_eef_pose_base(env)[0] - target) > 0.018:
        raise RuntimeError("E1/E2 safe target was not reached")
    return target, rotation


def _prepare_grasp(env) -> tuple[np.ndarray, np.ndarray]:
    """Follow the established Panda OSC grasp geometry; no object pose is overwritten."""
    eef_position, rotation = _eef_pose_base(env)
    object_position, _ = _pose_in(env, env.can_body_id, env.robot_base_body_id)
    base_position = np.asarray(env.sim.data.xpos[env.robot_base_body_id], dtype=float)
    base_rotation = np.asarray(env.sim.data.xmat[env.robot_base_body_id], dtype=float).reshape(3, 3)
    pad_midpoint = np.mean(
        [
            base_rotation.T @ (np.asarray(env.sim.data.geom_xpos[env.sim.model.geom_name2id(name)]) - base_position)
            for name in env.finger_pad_geom_names
        ],
        axis=0,
    )
    grasp = eef_position + object_position - pad_midpoint
    above = grasp + np.array((0.0, 0.0, 0.16))
    _drive(env, above, rotation, -1.0, 60)
    _drive(env, grasp, rotation, -1.0, 80)
    _drive(env, grasp, rotation, 1.0, 30)
    lifted = grasp + np.array((0.0, 0.0, 0.08))
    _drive(env, lifted, rotation, 1.0, 60)
    finger_object = _has_contact(env, env.finger_pad_geom_names, env.can.contact_geoms)
    object_position_world = np.asarray(env.sim.data.xpos[env.can_body_id], dtype=float)
    table_z = float(env.arena.table_top_abs[2])
    if not finger_object or object_position_world[2] - table_z < 0.055:
        raise RuntimeError(
            "E3 static grasp preparation failed: "
            f"finger_contact={finger_object}, object_lift_m={object_position_world[2] - table_z:.6f}"
        )
    return lifted, rotation


def _run_trial(
    experiment: str, seed: int, gamma: float, duration_s: float, *, env=None, gated: GatedProgram | None = None
) -> tuple[list[dict], dict, dict]:
    level_scale = 0.0 if gamma == 0.0 else level_scale_for_gamma(gamma, seed=seed, t0=0.0)
    owns_env = env is None
    if gated is None:
        gated = GatedProgram(build_excitation_program(seed=seed, t0=0.0, level_scale=level_scale))
    if env is None:
        env = _make_env(seed, gated)
    gated.program = build_excitation_program(seed=seed, t0=0.0, level_scale=level_scale)
    gated.start_s = None
    recorder = None
    try:
        env.reset()
        if experiment == "E3":
            center, rotation = _prepare_grasp(env)
        else:
            center, rotation = _prepare_empty(env)

        def reference(time_s: float):
            if experiment != "E2":
                return center, rotation
            ramp = float(quintic_smoothstep(time_s, 0.5))
            return center + np.array((0.010 * ramp * np.sin(np.pi * time_s), 0.0, 0.0)), rotation

        recorder = Recorder(env, experiment, reference, duration_s)
        env.add_post_physics_step_hook(recorder)
        start_s = float(env.sim.data.time)
        start_snapshot = _snapshot(env, include_object=experiment == "E3")
        recorder.start(start_s)
        if experiment == "E3":
            recorder.set_object_world_start()
        gated.start_s = start_s
        steps = int(round(duration_s * CONTROL_HZ))
        for step in range(steps):
            time_s = step / CONTROL_HZ
            target_position, target_rotation = reference(time_s)
            env.step(_action(env, target_position, target_rotation, 1.0 if experiment == "E3" else -1.0))
        if not recorder.rows:
            raise RuntimeError("physics-step recorder did not receive samples")
        summary = recorder.summary()
        summary.update(
            {
                "experiment": experiment,
                "seed": seed,
                "condition": "vibration" if gamma else "no_vibration",
                "gamma_commanded": gamma,
                "level_scale": level_scale,
                "gamma_calibrated_2s": calibrate_gamma(gated.program).gamma_commanded,
                "sample_rate_hz": 200,
                "measurement_s": duration_s,
            }
        )
        return recorder.rows, summary, start_snapshot
    finally:
        if recorder is not None and recorder in env._post_physics_step_hooks:
            env._post_physics_step_hooks.remove(recorder)
        if owns_env:
            env.close()


def _write_csv(path: Path, rows: list[dict], summary: dict) -> None:
    fields = ("seed", "condition", "gamma_commanded", *rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "seed": summary["seed"],
                    "condition": summary["condition"],
                    "gamma_commanded": summary["gamma_commanded"],
                    **row,
                }
            )


def _svg(path: Path, title: str, series: dict[str, tuple[np.ndarray, np.ndarray]], unit: str) -> None:
    width, height, margin = 900, 430, 55
    all_x = np.concatenate([value[0] for value in series.values()])
    all_y = np.concatenate([value[1] for value in series.values()])
    if not all_x.size:
        return
    xmin, xmax = float(np.min(all_x)), float(np.max(all_x))
    ymin, ymax = float(np.min(all_y)), float(np.max(all_y))
    if np.isclose(ymin, ymax):
        ymin, ymax = ymin - 1.0, ymax + 1.0
    colors = ("#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2", "#4f46e5")

    def xy(x, y):
        return margin + (x - xmin) / (xmax - xmin) * (width - 2 * margin), height - margin - (y - ymin) / (
            ymax - ymin
        ) * (height - 2 * margin)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{margin}" y="25" font-family="sans-serif" font-size="18">{title}</text>',
        f'<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#111"/>',
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="#111"/>',
        f'<text x="{margin}" y="{height-15}" font-family="sans-serif">time (s)</text>',
        f'<text x="8" y="{margin}" font-family="sans-serif">{unit}</text>',
    ]
    for index, (name, (xs, ys)) in enumerate(series.items()):
        points = " ".join(f"{x:.1f},{y:.1f}" for x, y in (xy(float(x), float(y)) for x, y in zip(xs, ys)))
        y = 48 + index * 18
        parts.extend(
            (
                f'<polyline fill="none" stroke="{colors[index % len(colors)]}" stroke-width="1.5" points="{points}"/>',
                f'<text x="{width-220}" y="{y}" fill="{colors[index % len(colors)]}" font-family="sans-serif" font-size="13">{name}</text>',
            )
        )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def _plots(output: Path, trials: list[tuple[list[dict], dict]]) -> None:
    for experiment in ("E1", "E2", "E3"):
        selected = [(rows, summary) for rows, summary in trials if summary["experiment"] == experiment]
        for metric, suffix, unit in (("eef_error_mm", "error", "mm"), ("object_relative_drift_mm", "drift", "mm")):
            if metric == "object_relative_drift_mm" and experiment != "E3":
                continue
            series = {}
            for condition in ("no_vibration", "vibration"):
                rows = [row for values, summary in selected if summary["condition"] == condition for row in values]
                by_time: dict[float, list[float]] = {}
                for row in rows:
                    by_time.setdefault(round(row["time_s"], 3), []).append(row[metric])
                xs = np.array(sorted(by_time))
                series[condition] = (xs, np.array([np.mean(by_time[x]) for x in xs]))
            _svg(output / f"{experiment}_{suffix}.svg", f"{experiment}: {suffix}", series, unit)
        if experiment in ("E1", "E2"):
            series = {}
            for condition in ("no_vibration", "vibration"):
                rows = [row for values, summary in selected if summary["condition"] == condition for row in values]
                by_time: dict[float, list[list[float]]] = {}
                for row in rows:
                    by_time.setdefault(round(row["time_s"], 3), []).append(
                        [row[f"joint_{joint}_torque_Nm"] for joint in range(1, 8)]
                    )
                xs = np.array(sorted(by_time))
                values = np.array([np.mean(by_time[x], axis=0) for x in xs])
                for joint in range(7):
                    series[f"{condition} J{joint + 1}"] = (xs, values[:, joint])
            _svg(output / f"{experiment}_torque.svg", f"{experiment}: all joint torque", series, "N m")


def _self_check() -> None:
    program = GatedProgram(build_excitation_program(seed=17, level_scale=1.0))
    assert np.allclose(program.evaluate(1.0).q, 0.0)
    program.start_s = 1.0
    assert np.allclose(program.evaluate(1.0).q, 0.0)
    assert np.allclose(quintic_smoothstep(0.0, 0.5), 0.0)
    assert np.allclose(quintic_smoothstep(0.5, 0.5), 1.0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("out/vibration_arm_effect_minimal"))
    parser.add_argument("--experiment", choices=("E1", "E2", "E3", "all"), default="all")
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS, help="defaults to the prescribed 17 18 19")
    parser.add_argument("--duration-s", type=float, default=MEASUREMENT_S)
    parser.add_argument("--check", action="store_true", help="run the no-simulator invariant check")
    args = parser.parse_args(argv)
    _self_check()
    if args.check:
        return 0
    if args.duration_s <= 0:
        parser.error("--duration-s must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    experiments = ("E1", "E2", "E3") if args.experiment == "all" else (args.experiment,)
    trials: list[tuple[list[dict], dict]] = []
    preparation_checks = []
    if any(seed < 0 for seed in args.seeds):
        parser.error("--seeds must be non-negative")
    for experiment in experiments:
        for seed in args.seeds:
            pair = []
            gated = GatedProgram(build_excitation_program(seed=seed, t0=0.0, level_scale=0.0))
            env = _make_env(seed, gated)
            try:
                for gamma in (0.0, GAMMA):
                    rows, summary, start = _run_trial(experiment, seed, gamma, args.duration_s, env=env, gated=gated)
                    _write_csv(args.output_dir / f"{experiment}_seed{seed}_{summary['condition']}.csv", rows, summary)
                    trials.append((rows, summary))
                    pair.append(start)
                    print(json.dumps(summary, sort_keys=True), flush=True)
            finally:
                env.close()
            differences = _snapshot_delta(*pair)
            preparation_checks.append(
                {
                    "experiment": experiment,
                    "seed": seed,
                    "max_abs_difference": differences,
                    "consistent_at_1e-9": all(value <= 1.0e-9 for value in differences.values()),
                }
            )
    _plots(args.output_dir, trials)
    try:
        revision = subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = "unavailable"
    report = {
        "schema_id": "shakebench.vibration_arm_effect_minimal.v1",
        "purpose": "diagnostic result, not a formal benchmark score",
        "reproduce": f"python -m robosuite.scripts.shakebench_probe_arm_effect --output-dir {args.output_dir}",
        "parameters": {
            "seeds": list(args.seeds),
            "gamma": GAMMA,
            "measurement_s": args.duration_s,
            "sample_rate_hz": 200,
            "geometry_profile": "direct_mount_v1",
            "physics_profile": "official",
            "control_hz": CONTROL_HZ,
        },
        "code_revision": revision,
        "trials": [summary for _, summary in trials],
        "preparation_checks": preparation_checks,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
