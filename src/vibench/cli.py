#!/usr/bin/env python3
"""ViBench command-line interface: run the embodied vibration-perturbed pick-and-place benchmark."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
import time
from pathlib import Path

from .paths import PROJECT_ROOT

import torch
import yaml

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene

from vibench import AssetConfig, BenchmarkConfig, PanelConfig, VibrationBenchmarkTask, VibrationConfig
from vibench.config import AXES, CONTROL_KINDS, SpectralBand
from vibench.controller import ScriptedPickPlaceController, grasp_feasibility_table
from vibench.diagnostics import (
    collision_shape_geometry,
    configure_mujoco_contact_solref,
    print_contact_snapshot,
)
from vibench.panel_controller import ScriptedPanelController
from vibench.panel_task import PanelBenchmarkTask
from vibench.recording import BenchmarkRecorder
from vibench.scene import install_clite_model_constraints, make_scene_cfg, make_sim_cfg
from vibench.supports import install_structural_collision_exclusions, support_group_geometries
from vibench.vibration import offline_support_travel_report
from vibench.wrist_camera import (
    WRIST_CAMERA_EYE_H,
    WRIST_CAMERA_FORWARD_H,
    WRIST_CAMERA_UP_H,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="vibench",
        description="Run a ViBench vibration-perturbed pick-and-place episode on Isaac Lab + Newton/MJWarp.",
    )
    parser.add_argument("--scenario", help="Load a named scenario from configs/scenarios.yaml")
    parser.add_argument(
        "--task",
        choices=("pick_place", "panel_operation"),
        default="pick_place",
        help="Benchmark task; panel_operation uses the fixed control panel on the existing worktable",
    )
    parser.add_argument(
        "--panel-sequence",
        help=f"Comma-separated ordered controls for panel_operation, e.g. knob,lever,button; omit to sample from --panel-seed",
    )
    parser.add_argument(
        "--panel-seed",
        type=int,
        help="Seed for sampling a random ordered control subset when --panel-sequence is not given",
    )
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "out" / "benchmark_v2_demo.mp4")
    parser.add_argument(
        "--camera-preset",
        choices=("main", "stewart_side", "panel_review"),
        default="main",
    )
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
    parser.add_argument("--solver-substeps", type=int, help="Override profile substeps for solver experiments")
    parser.add_argument("--contact-solref-timeconst", type=float, help="Override contact solref time constant for solver experiments")
    parser.add_argument("--solver-iterations", type=int, help="Override Newton solver main iterations for solver experiments")
    parser.add_argument(
        "--support-config",
        choices=("C2", "C2_CLITE"),
        default=None,
        help="C2 keeps kinematic teleport supports; C2_CLITE drives dynamic supports through mocap + weld constraints. Official defaults to C2_CLITE",
    )
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
    task_kind = scenario.get("task", args.task)
    if task_kind not in ("pick_place", "panel_operation"):
        raise ValueError(f"unknown task {task_kind!r}; expected pick_place or panel_operation")
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
    support_config = str(
        scenario.get(
            "support_config",
            args.support_config or ("C2_CLITE" if physics_profile == "official" else "C2"),
        )
    )
    if support_config not in ("C2", "C2_CLITE"):
        raise ValueError(f"unknown support_config {support_config!r}")
    # C2_CLITE at 4 substeps / 50 iterations passes the five-seed numerical
    # floor; ordinary C2 keeps the original 5-substep official gate.
    solver_substeps = args.solver_substeps or (
        5 if physics_profile == "official" and support_config == "C2" else 4
    )
    solver_iterations = args.solver_iterations or (
        50 if physics_profile == "official" and support_config == "C2_CLITE" else None
    )
    panel_sequence_value = (
        args.panel_sequence if args.panel_sequence is not None else scenario.get("panel_sequence")
    )
    if panel_sequence_value is None:
        panel_sequence = ()
    elif isinstance(panel_sequence_value, str):
        panel_sequence = tuple(
            kind.strip() for kind in panel_sequence_value.split(",") if kind.strip()
        )
    else:
        panel_sequence = tuple(str(kind) for kind in panel_sequence_value)
    unknown_panel = set(panel_sequence) - set(CONTROL_KINDS)
    if unknown_panel:
        raise ValueError(
            f"unknown panel controls {sorted(unknown_panel)}; expected {CONTROL_KINDS}"
        )
    panel_seed = (
        args.panel_seed
        if args.panel_seed is not None
        else int(scenario.get("panel_seed", args.seed))
    )
    panel_config = PanelConfig(seed=panel_seed, sequence=panel_sequence)
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
        solver_substeps=solver_substeps,
        episode_s=episode_s,
        num_envs=args.num_envs,
        task=task_kind,
        assets=AssetConfig(workpiece=workpiece, workpiece_scale=workpiece_scale),
        panel=panel_config,
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
        contact_solref=(
            (args.contact_solref_timeconst or 0.00060, 1.0)
            if physics_profile == "official"
            else (0.0025, 1.0)
        ),
        support_config=support_config,
        grasp_assist=args.grasp_assist,
        **benchmark_overrides,
    )
    support_groups = support_group_geometries(cfg)
    support_travel = offline_support_travel_report(
        cfg, support_groups, cfg.physics_hz, cfg.solver_substeps
    )
    if (
        cfg.vibration.mode != "off"
        and support_travel.max_substep_travel_m > cfg.max_substep_displacement_m
    ):
        raise ValueError(
            "unsafe support excitation: replayed substep travel "
            f"{1000.0 * support_travel.max_substep_travel_m:.3f} mm exceeds "
            f"fixed geometric limit {1000.0 * cfg.max_substep_displacement_m:.3f} mm; "
            "raise solver_substeps globally (never drop a seed)"
        )
    if physics_profile == "official" and cfg.physics_hz < 1000:
        raise ValueError("official profile requires physics_hz >= 1000")
    install_structural_collision_exclusions()
    with sim_utils.build_simulation_context(
        sim_cfg=make_sim_cfg(cfg, args.device, solver_iterations),
        auto_add_lighting=True,
    ) as sim:
        sim._app_control_on_stop_handle = None
        if cfg.use_clite_support:
            install_clite_model_constraints(cfg)
        scene = InteractiveScene(make_scene_cfg(cfg))
        sim.reset()
        if task_kind == "panel_operation":
            task = PanelBenchmarkTask(sim, scene, cfg)
            controller = ScriptedPanelController(task)
        else:
            task = VibrationBenchmarkTask(sim, scene, cfg)
            controller = ScriptedPickPlaceController(task)
        obs = task.reset()
        contact_response = configure_mujoco_contact_solref(cfg.contact_solref)
        contact_response["requested_margin_m"] = cfg.contact_margin_m
        contact_start = print_contact_snapshot("episode_start")
        workpiece_collision_shapes = (
            [] if task_kind == "panel_operation" else collision_shape_geometry("/Workpiece/")
        )
        panel_collision_shapes = (
            collision_shape_geometry("/Control")
            if task_kind == "panel_operation"
            else []
        )
        left_finger_collision_shapes = collision_shape_geometry("/Robot/panda_leftfinger")
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
        if task_kind == "panel_operation":
            print(
                f"[ASSETS] robot=Franka Panda table=phenolic_worktable_c2 "
                f"panel=control_panel controls=knob|lever|button"
            )
            print(f"[PANEL] instruction={list(task.panel_sequence)} panel_seed={cfg.panel.seed}")
        else:
            print(f"[ASSETS] robot=Franka Panda table=phenolic_worktable_c2 target=shallow_bin workpiece=YCB/{cfg.assets.workpiece}")
        print(f"[COLLIDER] workpiece={json.dumps(workpiece_collision_shapes, sort_keys=True)}")
        if task_kind == "panel_operation":
            print(f"[COLLIDER] panel={json.dumps(panel_collision_shapes, sort_keys=True)}")
        print(f"[COLLIDER] left_finger={json.dumps(left_finger_collision_shapes, sort_keys=True)}")
        print(f"[ROBOT] fixed_base={task.robot.is_fixed_base} joints={task.robot.joint_names}")
        print(
            f"[SENSOR] wrench_bodies="
            f"{task.wrist_wrench.body_names if task.wrist_wrench is not None else 'disabled_multi_articulation'}"
        )
        print("[SENSOR] wrist_camera=fixed panda_hand top mount RGB 384x240 vfov=75deg full_6dof_extrinsic=True")
        print(
            f"[PHYSICS] profile={physics_profile} hz={cfg.physics_hz} "
            f"substeps={cfg.solver_substeps} effective_hz={cfg.effective_substep_hz} "
            f"replayed_peak_substep={1000.0 * support_travel.max_substep_travel_m:.3f}mm"
        )
        print(f"[USD] work_thread_limit={os.environ.get('PXR_WORK_THREAD_LIMIT', 'unset')}")
        print(f"[CONTACT_RESPONSE] {json.dumps(contact_response, sort_keys=True)}")
        if task_kind != "panel_operation":
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
                    if task_kind == "pick_place":
                        task.record_ee_tracking_error(
                            controller.commanded_position_b,
                            obs["ee_pose_b"],
                        )
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
                    ee = obs["ee_pose_w"][0, :3].tolist()
                    camera_eye = obs["wrist_camera_eye_w"][0]
                    camera_forward = torch.nn.functional.normalize(
                        obs["wrist_camera_target_w"][0] - camera_eye,
                        dim=0,
                    )
                    camera_to_object = torch.nn.functional.normalize(
                        (
                            obs["workpiece_pose_w"][0, :3]
                            if "workpiece_pose_w" in obs
                            else obs["panel_pose_w"][0, :3]
                        )
                        - camera_eye,
                        dim=0,
                    )
                    camera_error_deg = float(
                        torch.rad2deg(
                            torch.acos(torch.clamp(torch.dot(camera_forward, camera_to_object), -1.0, 1.0))
                        ).item()
                    )
                    delta_z_mm = float(obs["mount_delta_z"][0, 0].item()) * 1000.0
                    finger_q = obs["joint_pos"][0, task.finger_joint_ids].tolist()
                    left_pos = obs["left_finger_pose_w"][0, :3].tolist()
                    right_pos = obs["right_finger_pose_w"][0, :3].tolist()
                    if task_kind == "panel_operation":
                        state = obs["panel_state"][0].tolist()
                        print(
                            f"[STEP] t={task.time_s:5.2f}s phase={controller.name:10s} "
                            f"ee={ee} state={[round(float(v), 3) for v in state]} "
                            f"delta_z={delta_z_mm:+.3f}mm camera_error={camera_error_deg:.2f}deg "
                            f"finger_q={finger_q} pads=({left_pos},{right_pos})"
                        )
                    else:
                        obj = obs["workpiece_pose_w"][0, :3].tolist()
                        left_n = float(obs["left_finger_contact_n"][0, 0].item())
                        right_n = float(obs["right_finger_contact_n"][0, 0].item())
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
        if task_kind == "pick_place":
            task.finalize_episode_metrics()
        if task_kind == "pick_place" and cfg.use_clite_support:
            timing = task._support_step_timing
            print(
                "[TIMING] support_substeps "
                + " ".join(f"{key}={timing[key]:.3f}s" for key in timing)
                + f" steps={task._support_step_count}"
            )
        print(f"[RESULT] {task.metrics}")
        contact_end = print_contact_snapshot("episode_end")
        metrics_dict = asdict(task.metrics)
        metrics_dict.pop("_control_contact_streak", None)
        args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_output.write_text(
            json.dumps(
                {
                    "backend": "Isaac Lab + Newton/MJWarp",
                    "task": cfg.task,
                    "robot": cfg.assets.robot,
                    "table": cfg.assets.table,
                    "target": "control_panel" if cfg.task == "panel_operation" else "shallow_storage_bin",
                    "workpiece": None if cfg.task == "panel_operation" else cfg.assets.workpiece,
                    "panel_instruction": (
                        list(task.panel_sequence) if cfg.task == "panel_operation" else None
                    ),
                    "panel_seed": cfg.panel.seed if cfg.task == "panel_operation" else None,
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
                    "support_config": cfg.support_config,
                    "solver_substeps": cfg.solver_substeps,
                    "effective_substep_hz": cfg.effective_substep_hz,
                    "usd_work_thread_limit": os.environ.get("PXR_WORK_THREAD_LIMIT", "unset"),
                    "replayed_max_substep_travel_mm": 1000.0 * support_travel.max_substep_travel_m,
                    "replayed_max_v_dt_mm": 1000.0 * support_travel.max_v_dt_m,
                    "replayed_max_half_a_dt2_mm": 1000.0 * support_travel.max_half_a_dt2_m,
                    "replayed_worst_member": support_travel.worst_member,
                    "replayed_worst_time_s": support_travel.worst_time_s,
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
                    "panel_collision_shapes": panel_collision_shapes,
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
                    "metrics": metrics_dict,
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
