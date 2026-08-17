#!/usr/bin/env python3
"""Run the standalone asset-backed vibrating pick-and-place benchmark."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch
import yaml

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene

from vibration_benchmark_v2 import AssetConfig, BenchmarkConfig, VibrationBenchmarkTask, VibrationConfig
from vibration_benchmark_v2.config import AXES, SpectralBand
from vibration_benchmark_v2.controller import ScriptedPickPlaceController, grasp_feasibility_table
from vibration_benchmark_v2.diagnostics import (
    collision_shape_geometry,
    configure_mujoco_contact_solref,
    print_contact_snapshot,
)
from vibration_benchmark_v2.recording import BenchmarkRecorder
from vibration_benchmark_v2.scene import make_scene_cfg, make_sim_cfg
from vibration_benchmark_v2.vibration import validate_impulsive_timestep
from vibration_benchmark_v2.wrist_camera import (
    WRIST_CAMERA_EYE_H,
    WRIST_CAMERA_FORWARD_H,
    WRIST_CAMERA_UP_H,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", help="Load a named scenario from configs/scenarios.yaml")
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "out" / "benchmark_v2_demo.mp4")
    parser.add_argument("--camera-preset", choices=("main", "stewart_side"), default="main")
    parser.add_argument("--no-overlays", action="store_true", help="Record clean NewtonGL frames without telemetry panels")
    parser.add_argument("--workpiece", choices=("cracker_box", "sugar_box", "soup_can", "mustard_bottle"), default="sugar_box")
    parser.add_argument("--workpiece-scale", type=float, default=0.75)
    parser.add_argument("--vibration", choices=("off", "sine", "spectral"), default="spectral")
    parser.add_argument("--sine-axis", choices=("tx", "ty", "tz", "rx", "ry", "rz"), default="tz")
    parser.add_argument("--sine-amplitude", type=float, default=0.0015)
    parser.add_argument("--sine-frequency-hz", type=float, default=5.0)
    parser.add_argument(
        "--spectral-scale",
        type=float,
        default=1.0,
        help="Scale every enabled spectral axis together; 0.15 is safe for the 240 Hz full-6DOF profile",
    )
    parser.add_argument(
        "--vibration-axes",
        default=",".join(AXES),
        help="Comma-separated spectral axes, for example tx,tz for longitudinal vehicle motion",
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--episode-s", type=float, default=16.0)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--physics-profile", choices=("official", "training"), default="official")
    parser.add_argument("--physics-hz", type=int, help="Override the profile rate; official scoring requires >=1000 Hz")
    parser.add_argument(
        "--gripper-closing-speed",
        type=float,
        help="Scenario override in m/s; the config default remains backward compatible",
    )
    parser.add_argument(
        "--grasp-timeout-s",
        type=float,
        help="Scenario override for the bilateral-contact timeout",
    )
    grasp = parser.add_mutually_exclusive_group()
    grasp.add_argument("--grasp-assist", dest="grasp_assist", action="store_true", help="Opt in to the disclosed demo stabilizer")
    grasp.add_argument("--no-grasp-assist", dest="grasp_assist", action="store_false", help="Deprecated compatibility spelling; assistance is already off by default")
    parser.set_defaults(grasp_assist=False)
    parser.add_argument("--metrics-output", type=Path, default=PROJECT_ROOT / "out" / "last_metrics.json")
    return parser.parse_args()


def load_scenario(name: str | None) -> dict:
    if name is None:
        return {}
    scenarios_path = PROJECT_ROOT / "configs" / "scenarios.yaml"
    payload = yaml.safe_load(scenarios_path.read_text(encoding="utf-8"))
    scenarios = payload.get("scenarios", {})
    if name not in scenarios:
        raise ValueError(f"unknown scenario {name!r}; available={sorted(scenarios)}")
    return dict(scenarios[name])


def scenario_bands(payload: dict) -> dict[str, tuple[SpectralBand, ...]] | None:
    configured = payload.get("bands")
    if configured is None:
        return None
    return {
        axis: tuple(
            SpectralBand(
                center_hz=float(band["center_hz"]),
                rms=float(band["rms"] if "rms" in band else band["rms_m"]),
                bandwidth_ratio=float(band.get("bandwidth_ratio", 0.08)),
                tones=int(band.get("tones", 12)),
            )
            for band in bands
        )
        for axis, bands in configured.items()
    }


def main() -> int:
    args = parse_args()
    scenario = load_scenario(args.scenario)
    vibration_mode = scenario.get("vibration", args.vibration)
    spectral_scale = float(scenario.get("spectral_scale", args.spectral_scale))
    configured_axes = scenario.get("axes", args.vibration_axes)
    if isinstance(configured_axes, str):
        active_axes = tuple(axis.strip() for axis in configured_axes.split(",") if axis.strip())
    else:
        active_axes = tuple(configured_axes)
    spectral_bands = scenario_bands(scenario)
    workpiece = scenario.get("workpiece", args.workpiece)
    workpiece_scale = float(scenario.get("workpiece_scale", args.workpiece_scale))
    episode_s = float(scenario.get("episode_s", args.episode_s))
    physics_profile = scenario.get("physics_profile", args.physics_profile)
    physics_hz = args.physics_hz or (1000 if physics_profile == "official" else 240)
    if physics_hz <= 0:
        raise ValueError("physics_hz must be positive")
    benchmark_overrides = {}
    closing_speed = (
        args.gripper_closing_speed
        if args.gripper_closing_speed is not None
        else scenario.get("gripper_closing_speed_m_s")
    )
    grasp_timeout_s = (
        args.grasp_timeout_s
        if args.grasp_timeout_s is not None
        else scenario.get("grasp_timeout_s")
    )
    if closing_speed is not None:
        benchmark_overrides["gripper_closing_speed_m_s"] = float(closing_speed)
    if grasp_timeout_s is not None:
        benchmark_overrides["grasp_timeout_s"] = float(grasp_timeout_s)
    cfg = BenchmarkConfig(
        dt=1.0 / physics_hz,
        episode_s=episode_s,
        num_envs=args.num_envs,
        assets=AssetConfig(workpiece=workpiece, workpiece_scale=workpiece_scale),
        vibration=VibrationConfig(
            mode=vibration_mode,
            seed=args.seed,
            sine_axis=args.sine_axis,
            sine_amplitude=args.sine_amplitude,
            sine_frequency_hz=args.sine_frequency_hz,
            spectral_scale=spectral_scale,
            active_axes=active_axes,
            **({"bands": spectral_bands} if spectral_bands is not None else {}),
        ),
        contact_solref=(0.00060, 1.0) if physics_profile == "official" else (0.0025, 1.0),
        grasp_assist=args.grasp_assist,
        **benchmark_overrides,
    )
    estimated_substep_displacement_m = validate_impulsive_timestep(
        cfg.vibration,
        cfg.physics_hz,
        cfg.solver_substeps,
        cfg.max_substep_displacement_m,
    )
    if physics_profile == "official" and cfg.physics_hz < 1000:
        raise ValueError("official profile requires physics_hz >= 1000")
    with sim_utils.build_simulation_context(sim_cfg=make_sim_cfg(cfg, args.device), auto_add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None
        scene = InteractiveScene(make_scene_cfg(cfg))
        sim.reset()
        task = VibrationBenchmarkTask(sim, scene, cfg)
        contact_response = configure_mujoco_contact_solref(cfg.contact_solref)
        contact_response["requested_margin_m"] = cfg.contact_margin_m
        obs = task.reset()
        contact_start = print_contact_snapshot("episode_start")
        workpiece_collision_shapes = collision_shape_geometry("/Workpiece/")
        left_finger_collision_shapes = collision_shape_geometry("/Robot/panda_leftfinger")
        controller = ScriptedPickPlaceController(task)
        recorder = (
            BenchmarkRecorder(
                args.output,
                camera_preset=args.camera_preset,
                overlays=not args.no_overlays,
            )
            if args.record
            else None
        )
        frame_stride = max(1, round((1.0 / cfg.dt) / 30.0))
        shaker_leg_spread_max_m = float(
            (task._shaker_leg_lengths.max(dim=1).values - task._shaker_leg_lengths.min(dim=1).values)
            .max()
            .item()
        )
        step = 0
        wall_start = time.perf_counter()
        print(f"[ASSETS] robot=Franka Panda table=phenolic_worktable_c2 target=shallow_bin workpiece=YCB/{cfg.assets.workpiece}")
        print(f"[COLLIDER] workpiece={json.dumps(workpiece_collision_shapes, sort_keys=True)}")
        print(f"[COLLIDER] left_finger={json.dumps(left_finger_collision_shapes, sort_keys=True)}")
        print(f"[ROBOT] fixed_base={task.robot.is_fixed_base} joints={task.robot.joint_names}")
        print(f"[SENSOR] wrench_bodies={task.wrist_wrench.body_names}")
        print("[SENSOR] wrist_camera=fixed panda_hand top mount RGB 384x240 vfov=75deg full_6dof_extrinsic=True")
        print(
            f"[PHYSICS] profile={physics_profile} hz={cfg.physics_hz} "
            f"substeps={cfg.solver_substeps} effective_hz={cfg.effective_substep_hz} "
            f"estimated_peak_substep={1000.0 * estimated_substep_displacement_m:.3f}mm"
        )
        print(f"[USD] work_thread_limit={os.environ.get('PXR_WORK_THREAD_LIMIT', 'unset')}")
        print(f"[CONTACT_RESPONSE] {json.dumps(contact_response, sort_keys=True)}")
        print("[GRASP_FEASIBILITY] workpiece scale min_horizontal_mm feasible")
        for row in grasp_feasibility_table():
            print(
                f"[GRASP_FEASIBILITY] {row['workpiece']} {row['scale']:.2f} "
                f"{1000.0 * row['min_horizontal_m']:.1f} {row['feasible']}"
            )
        try:
            while task.time_s < cfg.episode_s and not controller.finished:
                with torch.inference_mode():
                    arm, fingers = controller.command(obs)
                    obs = task.step(arm, fingers)
                    shaker_leg_spread_max_m = max(
                        shaker_leg_spread_max_m,
                        float(
                            (obs["shaker_leg_lengths_m"].max(dim=1).values - obs["shaker_leg_lengths_m"].min(dim=1).values)
                            .max()
                            .item()
                        ),
                    )
                if recorder is not None and step % frame_stride == 0:
                    recorder.add_frame(task, controller, obs)
                if step % 240 == 0:
                    obj = obs["workpiece_pose_w"][0, :3].tolist()
                    ee = obs["ee_pose_w"][0, :3].tolist()
                    camera_eye = obs["wrist_camera_eye_w"][0]
                    camera_forward = torch.nn.functional.normalize(
                        obs["wrist_camera_target_w"][0] - camera_eye,
                        dim=0,
                    )
                    camera_to_object = torch.nn.functional.normalize(
                        obs["workpiece_pose_w"][0, :3] - camera_eye,
                        dim=0,
                    )
                    camera_error_deg = float(
                        torch.rad2deg(
                            torch.acos(torch.clamp(torch.dot(camera_forward, camera_to_object), -1.0, 1.0))
                        ).item()
                    )
                    left_n = float(obs["left_finger_contact_n"][0, 0].item())
                    right_n = float(obs["right_finger_contact_n"][0, 0].item())
                    delta_z_mm = float(obs["mount_delta_z"][0, 0].item()) * 1000.0
                    finger_q = obs["joint_pos"][0, task.finger_joint_ids].tolist()
                    left_pos = obs["left_finger_pose_w"][0, :3].tolist()
                    right_pos = obs["right_finger_pose_w"][0, :3].tolist()
                    print(
                        f"[STEP] t={task.time_s:5.2f}s phase={controller.name:10s} "
                        f"object={obj} ee={ee} contact=({left_n:.3f},{right_n:.3f})N "
                        f"hold={bool(obs['grasped'][0])} delta_z={delta_z_mm:+.3f}mm "
                        f"camera_error={camera_error_deg:.2f}deg "
                        f"camera_eye={camera_eye.tolist()} camera_forward={camera_forward.tolist()} "
                        f"object_ray={camera_to_object.tolist()} "
                        f"finger_q={finger_q} pads=({left_pos},{right_pos})"
                    )
                step += 1
        finally:
            if recorder is not None:
                recorder.close()
        wall_elapsed_s = max(time.perf_counter() - wall_start, 1.0e-9)
        vibration_times, vibration_values = task.vibration_history()
        if vibration_values.shape[0] > 0:
            observed_rms = torch.sqrt(torch.mean(vibration_values * vibration_values, dim=0))
            observed_peak = torch.max(torch.abs(vibration_values), dim=0).values
            observed_axis_rms = {
                axis: float(observed_rms[index].item()) for index, axis in enumerate(AXES)
            }
            observed_axis_peak_abs = {
                axis: float(observed_peak[index].item()) for index, axis in enumerate(AXES)
            }
            observation_window_s = float((vibration_times[-1] - vibration_times[0]).item()) if vibration_times.shape[0] > 1 else 0.0
        else:
            observed_axis_rms = {axis: 0.0 for axis in AXES}
            observed_axis_peak_abs = {axis: 0.0 for axis in AXES}
            observation_window_s = 0.0
        print(f"[RESULT] {task.metrics}")
        contact_end = print_contact_snapshot("episode_end")
        args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_output.write_text(
            json.dumps(
                {
                    "backend": "Isaac Lab + Newton/MJWarp",
                    "robot": cfg.assets.robot,
                    "table": cfg.assets.table,
                    "target": "shallow_storage_bin",
                    "workpiece": cfg.assets.workpiece,
                    "vibration_mode": cfg.vibration.mode,
                    "sine_axis": cfg.vibration.sine_axis if cfg.vibration.mode == "sine" else None,
                    "sine_amplitude": cfg.vibration.sine_amplitude if cfg.vibration.mode == "sine" else None,
                    "sine_frequency_hz": cfg.vibration.sine_frequency_hz if cfg.vibration.mode == "sine" else None,
                    "spectral_scale": cfg.vibration.spectral_scale if cfg.vibration.mode == "spectral" else None,
                    "vibration_axes": list(cfg.vibration.active_axes) if cfg.vibration.mode == "spectral" else ([cfg.vibration.sine_axis] if cfg.vibration.mode == "sine" else []),
                    "observed_vibration_window_s": observation_window_s,
                    "observed_vibration_displacement_rms": observed_axis_rms,
                    "observed_vibration_displacement_peak_abs": observed_axis_peak_abs,
                    "seed": cfg.vibration.seed,
                    "scenario": args.scenario,
                    "physics_profile": physics_profile,
                    "physics_hz": cfg.physics_hz,
                    "solver_substeps": cfg.solver_substeps,
                    "effective_substep_hz": cfg.effective_substep_hz,
                    "usd_work_thread_limit": os.environ.get("PXR_WORK_THREAD_LIMIT", "unset"),
                    "estimated_peak_substep_displacement_mm": 1000.0 * estimated_substep_displacement_m,
                    "wall_elapsed_s": wall_elapsed_s,
                    "physics_steps_per_wall_s": step / wall_elapsed_s,
                    "grasp_assist_enabled": cfg.grasp_assist,
                    "controller_motion_limits": {
                        "approach_clearance_m": cfg.approach_clearance_m,
                        "descend_clearance_m": cfg.descend_clearance_m,
                        "finger_table_clearance_m": cfg.finger_table_clearance_m,
                        "runtime_finger_downward_reach_m": task.finger_downward_reach_m,
                        "arm_linear_speed_m_s": cfg.arm_linear_speed_m_s,
                        "lift_takeoff_speed_m_s": cfg.lift_takeoff_speed_m_s,
                        "lift_takeoff_duration_s": cfg.lift_takeoff_duration_s,
                        "descend_linear_speed_m_s": cfg.descend_linear_speed_m_s,
                        "place_linear_speed_m_s": cfg.place_linear_speed_m_s,
                        "gripper_closing_speed_m_s": cfg.gripper_closing_speed_m_s,
                        "gripper_contact_recovery_speed_m_s": cfg.gripper_contact_recovery_speed_m_s,
                        "gripper_opening_speed_m_s": cfg.gripper_opening_speed_m_s,
                        "gripper_contact_preload_m": cfg.gripper_contact_preload_m,
                        "grasp_timeout_s": cfg.grasp_timeout_s,
                        "grasp_contact_loss_timeout_s": cfg.grasp_contact_loss_timeout_s,
                        "grasp_slip_tolerance_m": cfg.grasp_slip_tolerance_m,
                    },
                    "controller_failure_reason": controller.failure_reason,
                    "contact_diagnostics": {
                        "episode_start": contact_start.to_dict(),
                        "episode_end": contact_end.to_dict(),
                    },
                    "contact_response": contact_response,
                    "workpiece_collision_shapes": workpiece_collision_shapes,
                    "left_finger_collision_shapes": left_finger_collision_shapes,
                    "shaker": {
                        "geometry": asdict(cfg.shaker),
                        "leg_length_min_m": float(task._shaker_leg_lengths.min().item()),
                        "leg_length_max_m": float(task._shaker_leg_lengths.max().item()),
                        "max_instantaneous_leg_spread_m": shaker_leg_spread_max_m,
                    },
                    "wrist_camera": {
                        "physical_mount": "panda_hand/WristCamera (fixed top-offset D415-style geometry)",
                        "rgb_resolution": [384, 240],
                        "vertical_fov_deg": 75.0,
                        "eye_in_hand_m": list(WRIST_CAMERA_EYE_H),
                        "forward_in_hand": list(WRIST_CAMERA_FORWARD_H),
                        "up_in_hand": list(WRIST_CAMERA_UP_H),
                        "world_stabilization": False,
                    },
                    "metrics": asdict(task.metrics),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"[METRICS] {args.metrics_output.resolve()}")
        if recorder is not None:
            print(f"[VIDEO] {args.output.resolve()} frames={recorder.frames}")
        return 0 if task.metrics.success else 2


if __name__ == "__main__":
    raise SystemExit(main())
