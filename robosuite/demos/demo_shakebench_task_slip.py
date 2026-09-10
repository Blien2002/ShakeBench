"""Record equal-condition vibration/slip demos for all six task variants.

The clips deliberately keep the gripper open and stationary. They show only
surface-relative object motion, not pick/place policy skill. Gamma uses the
same two-second authored-program calibration as the formal oracle runner.

Example:
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python -m \
        robosuite.demos.demo_shakebench_task_slip --output-dir out/task_slip_demos
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import mujoco
import numpy as np

from robosuite.demos.demo_shakebench_oracle_video import FFmpegVideoWriter
from robosuite.utils.shakebench_calibration import calibrate_gamma, level_scale_for_gamma
from robosuite.utils.shakebench_excitation import build_excitation_program
from robosuite.utils.shakebench_tasks import OBJECTS, SURFACES, TaskSpec, make_task_env, task_variants


def _camera(env) -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = np.asarray(env.arena.table_top_abs, dtype=float) + (0.0, 0.02, 0.02)
    camera.distance = 1.02
    camera.azimuth = 140.0
    camera.elevation = -47.0
    return camera


def _frame(
    renderer,
    env,
    camera,
    option,
    spec: TaskSpec,
    sample: dict,
    *,
    gamma: float,
    level_scale: float,
) -> np.ndarray:
    renderer.update_scene(env.sim.data._data, camera=camera, scene_option=option)
    image = renderer.render().copy()
    lines = (
        f"{spec.surface_id.upper()} / {spec.object_id.upper()}     table-object mu={spec.table_sliding_mu:.2f}",
        f"official Gamma={gamma:.3f}     level scale={level_scale:.6f}     t={sample['time_s']:.2f} s",
        f"surface-relative slip={sample['slip_distance_m'] * 1000:.2f} mm     speed={sample['slip_speed_m_s'] * 1000:.2f} mm/s",
    )
    for index, line in enumerate(lines):
        y = 30 + 25 * index
        cv2.putText(image, line, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (15, 20, 22), 3, cv2.LINE_AA)
        cv2.putText(image, line, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (235, 242, 240), 1, cv2.LINE_AA)
    return image


def _sample(env, spec: TaskSpec) -> dict:
    report = env.get_metrics(update=True)
    worktable = report["can"]["worktable"]
    # The object z-axis in the worktable frame exposes tipping / rolling, so
    # translational displacement after a tip is never mislabeled pure slide.
    body_id = env.can_body_id
    worktable_id = env.worktable_body_id
    body_rotation = np.asarray(env.sim.data.xmat[body_id], dtype=float).reshape(3, 3)
    table_rotation = np.asarray(env.sim.data.xmat[worktable_id], dtype=float).reshape(3, 3)
    upright_cosine = float(table_rotation[:, 2].dot(body_rotation[:, 2]))
    contacts = report["contacts"]
    return {
        "variant_id": spec.variant_id,
        "object_id": spec.object_id,
        "surface_id": spec.surface_id,
        "table_object_sliding_mu": spec.table_sliding_mu,
        "time_s": float(report["time_s"]),
        "slip_distance_m": float(report["table_slip_distance_m"]),
        "slip_speed_m_s": float(report["table_slip_speed_m_s"]),
        "object_x_worktable_m": float(worktable["position_m"][0]),
        "object_y_worktable_m": float(worktable["position_m"][1]),
        "object_z_worktable_m": float(worktable["position_m"][2]),
        "object_vx_worktable_m_s": float(worktable["linear_velocity_m_s"][0]),
        "object_vy_worktable_m_s": float(worktable["linear_velocity_m_s"][1]),
        "object_upright_cosine": upright_cosine,
        "table_contact_present": bool(contacts["table_contact_present"]),
    }


def record_variant(
    spec: TaskSpec,
    *,
    output_dir: Path,
    duration_s: float,
    fps: int,
    width: int,
    height: int,
    seed: int,
    gamma: float | None,
    level_scale: float,
    playback_slowdown: int,
) -> tuple[list[dict], dict]:
    program = build_excitation_program(seed=seed, t0=0.0, level_scale=level_scale)
    gamma_2s = calibrate_gamma(program).gamma_commanded
    gamma_rollout = calibrate_gamma(program, duration_s=duration_s).gamma_commanded
    env = make_task_env(
        spec,
        observation_tier="V0",
        excitation_program=program,
        physics_profile="official",
        geometry_profile="direct_mount_v1",
        imu_mode="canonical_noisy_v1",
        hard_reset=False,
        horizon=int(np.ceil(duration_s * 20.0)) + 2,
        seed=seed,
    )
    output = output_dir / f"{spec.variant_id}.mp4"
    records: list[dict] = []
    try:
        env.reset()
        camera = _camera(env)
        option = mujoco.MjvOption()
        option.geomgroup[0] = 0
        option.geomgroup[1] = 1
        with mujoco.Renderer(env.sim.model._model, height=height, width=width) as renderer:
            writer = FFmpegVideoWriter(output, width=width, height=height, fps=fps)
            try:
                for _ in range(int(round(duration_s * fps))):
                    env.step(np.zeros(env.action_dim, dtype=float))
                    sample = _sample(env, spec)
                    records.append(sample)
                    frame = _frame(
                        renderer,
                        env,
                        camera,
                        option,
                        spec,
                        sample,
                        gamma=gamma_2s,
                        level_scale=level_scale,
                    )
                    for _ in range(playback_slowdown):
                        writer.append_data(frame)
            finally:
                writer.close()
    finally:
        env.close()
    slips = np.asarray([row["slip_distance_m"] for row in records], dtype=float)
    speeds = np.asarray([row["slip_speed_m_s"] for row in records], dtype=float)
    upright = np.asarray([row["object_upright_cosine"] for row in records], dtype=float)
    summary = {
        "variant_id": spec.variant_id,
        "object_id": spec.object_id,
        "surface_id": spec.surface_id,
        "table_object_sliding_mu": spec.table_sliding_mu,
        "video": str(output),
        "gamma_requested": gamma,
        "gamma_calibrated_2s": gamma_2s,
        "gamma_observed_over_rollout_window": gamma_rollout,
        "level_scale": level_scale,
        "sample_count": len(records),
        "peak_slip_mm": float(np.max(slips) * 1000),
        "final_slip_mm": float(slips[-1] * 1000),
        "rms_slip_mm": float(np.sqrt(np.mean(slips**2)) * 1000),
        "peak_slip_speed_mm_s": float(np.max(speeds) * 1000),
        "first_slip_time_s": next((row["time_s"] for row in records if row["slip_speed_m_s"] > 1e-3), None),
        "minimum_upright_cosine": float(np.min(upright)),
        "tip_time_s": next((row["time_s"] for row in records if row["object_upright_cosine"] < 0.95), None),
    }
    return records, summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("out/task_slip_demos"))
    parser.add_argument("--duration-s", type=float, default=5.0)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--object-id", choices=tuple(OBJECTS), default=None)
    parser.add_argument("--surface-id", choices=SURFACES, default=None)
    excitation = parser.add_mutually_exclusive_group(required=True)
    excitation.add_argument("--gamma", type=float, help="formal two-second calibrated Gamma command")
    excitation.add_argument("--level-scale", type=float, help="explicit exploratory scale; cannot be called Gamma")
    parser.add_argument(
        "--playback-slowdown",
        type=int,
        default=1,
        help="repeat rendered frames for qualitative slow motion; physics samples are unchanged",
    )
    args = parser.parse_args(argv)
    if args.duration_s <= 0 or args.fps <= 0 or args.width <= 0 or args.height <= 0:
        raise ValueError("duration, fps, width, and height must be positive")
    if args.fps != 20:
        raise ValueError("fps must equal the 20 Hz control rate so every rendered frame is one rollout step")
    if args.seed < 0 or args.playback_slowdown <= 0:
        raise ValueError("seed must be non-negative; playback-slowdown must be positive")
    if args.gamma is not None and args.gamma < 0:
        raise ValueError("gamma must be non-negative")
    if args.level_scale is not None and args.level_scale < 0:
        raise ValueError("level-scale must be non-negative")
    if (args.object_id is None) != (args.surface_id is None):
        raise ValueError("object-id and surface-id must be supplied together")
    level_scale = (
        level_scale_for_gamma(args.gamma, seed=args.seed, t0=0.0) if args.gamma is not None else float(args.level_scale)
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    summaries: list[dict] = []
    specs = (
        task_variants() if args.object_id is None else (TaskSpec(object_id=args.object_id, surface_id=args.surface_id),)
    )
    for spec in specs:
        samples, summary = record_variant(
            spec,
            output_dir=args.output_dir,
            duration_s=args.duration_s,
            fps=args.fps,
            width=args.width,
            height=args.height,
            seed=args.seed,
            gamma=args.gamma,
            level_scale=level_scale,
            playback_slowdown=args.playback_slowdown,
        )
        rows.extend(samples)
        summaries.append(summary)
        print(json.dumps(summary, sort_keys=True), flush=True)
    with (args.output_dir / "slip_timeseries.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "schema_id": "shakebench.task_slip_demo.v1",
        "purpose": "formal-physics diagnostic of ungrasped table-relative slip; not a Phase 09 score",
        "excitation": {
            "seed": args.seed,
            "gamma_requested": args.gamma,
            "gamma_calibration_window_s": 2.0 if args.gamma is not None else None,
            "level_scale": level_scale,
            "duration_s": args.duration_s,
            "physics_profile": "official",
            "geometry_profile": "direct_mount_v1",
        },
        "video_playback": {"fps": args.fps, "slowdown": args.playback_slowdown},
        "summary": summaries,
        "timeseries": "slip_timeseries.csv",
    }
    (args.output_dir / "slip_summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
