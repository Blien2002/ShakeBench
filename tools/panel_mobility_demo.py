"""Control-panel mobility demo without arm operation.

The Panda arm stays parked at its reset posture.  Each panel control is
exercised directly through its own simulated articulation:

- knob:  revolute joint, effort-ramped PD torque toward +72 deg and back
- lever: revolute joint, effort-ramped PD torque toward +30 deg and back
- button: prismatic joint, its authored linear drive toward -4 mm and back

Joint positions are never written by this tool.  Progress is read from the
simulated articulation state, exactly as the benchmark task does.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene

from vibench import BenchmarkConfig, PanelConfig, VibrationConfig
from vibench.panel_task import PanelBenchmarkTask
from vibench.recording import BenchmarkRecorder
from vibench.scene import make_scene_cfg, make_sim_cfg
from vibench.supports import install_structural_collision_exclusions

KNOB = 0
LEVER = 1
BUTTON = 2


class DemoController:
    """Recorder-facing stub: this demo does not command the robot arm."""

    def __init__(self) -> None:
        self.failure_reason = None
        self._finished = False

    @property
    def name(self) -> str:
        return "control_mobility_demo"

    @property
    def finished(self) -> bool:
        return self._finished


def ramp_goal(now_s: float, start_s: float, ramp_s: float, goal: float) -> float:
    if ramp_s <= 0.0:
        return float(goal)
    phase = max(0.0, min(1.0, (now_s - start_s) / ramp_s))
    return phase * float(goal)


def main() -> int:
    parser = argparse.ArgumentParser(description="Record panel-control mobility without arm operation")
    parser.add_argument("--output", type=Path, default=Path("out/panel_mobility_demo.mp4"))
    parser.add_argument("--metrics-output", type=Path, default=Path("out/panel_mobility_demo.json"))
    parser.add_argument("--physics-profile", choices=("training", "official"), default="training")
    parser.add_argument("--episode-s", type=float, default=14.0)
    parser.add_argument("--capture-fps", type=float, default=30.0)
    parser.add_argument("--no-overlays", action="store_true")
    args = parser.parse_args()
    if args.episode_s <= 0.0:
        parser.error("--episode-s must be positive")
    if args.capture_fps <= 0.0:
        parser.error("--capture-fps must be positive")

    physics_hz = 240 if args.physics_profile == "training" else 1000
    solver_substeps = 4 if args.physics_profile == "training" else 5
    contact_solref = (0.0025, 1.0) if args.physics_profile == "training" else (0.00060, 1.0)
    cfg = BenchmarkConfig(
        dt=1.0 / physics_hz,
        solver_substeps=solver_substeps,
        episode_s=args.episode_s,
        task="panel_operation",
        panel=PanelConfig(sequence=("knob", "lever", "button")),
        vibration=VibrationConfig(mode="off"),
        contact_solref=contact_solref,
    )

    # Phase table: (start_s, ramp_s, hold_s).  Ramp out, hold, ramp back.
    phases = {
        "knob": (0.8, 1.2, 0.8),
        "lever": (3.8, 1.5, 0.8),
        "button": (7.6, 1.0, 0.8),
    }
    goals = {
        "knob": cfg.panel.knob_goal_rad,
        "lever": cfg.panel.lever_goal_rad,
        "button": -cfg.panel.button_travel_m,
    }
    # Low-gain P effort for the two zero-stiffness revolute joints.  Their
    # intrinsic damping is already enough for a well-behaved first-order
    # response; a velocity feedback term here only chatters at the control
    # rate on these milligram-class inertias.
    pd = {
        "knob": {"kp": 0.4, "kd": 0.0, "limit": 0.6},
        "lever": {"kp": 0.3, "kd": 0.0, "limit": 0.5},
    }

    install_structural_collision_exclusions()
    with sim_utils.build_simulation_context(
        sim_cfg=make_sim_cfg(cfg, "cuda:0"), auto_add_lighting=True
    ) as sim:
        sim._app_control_on_stop_handle = None
        scene = InteractiveScene(make_scene_cfg(cfg))
        sim.reset()
        task = PanelBenchmarkTask(sim, scene, cfg)
        controller = DemoController()
        recorder = BenchmarkRecorder(
            args.output,
            fps=round(args.capture_fps),
            camera_preset="panel_review",
            overlays=not args.no_overlays,
        )
        obs = task.reset()

        arm = task.robot.data.default_joint_pos.torch[:, task.arm_joint_ids].clone()
        fingers = torch.full((1, len(task.finger_joint_ids)), 0.04, device=task.device)
        control_assets = {"knob": task.knob, "lever": task.lever, "button": task.button}
        control_joint_ids = {
            "knob": task.knob_joint_id,
            "lever": task.lever_joint_id,
            "button": task.button_joint_id,
        }

        stats = {
            kind: {
                "goal": float(goals[kind]),
                "q_min": 0.0,
                "q_max": 0.0,
                "peak_abs_qd": 0.0,
                "peak_progress": 0.0,
                "final_q": 0.0,
                "max_abs_effort_target": 0.0,
            }
            for kind in ("knob", "lever", "button")
        }

        def drive_control(kind: str, goal: float) -> float:
            asset = control_assets[kind]
            joint_id = control_joint_ids[kind]
            if kind == "button":
                # Authored prismatic spring drive: command its internal target.
                target = asset.data.default_joint_pos.torch.clone()
                target[:, joint_id] = goal
                asset.set_joint_position_target_index(target=target, joint_ids=[joint_id])
                return 0.0
            q = asset.data.joint_pos.torch[:, joint_id]
            qd = asset.data.joint_vel.torch[:, joint_id]
            gains = pd[kind]
            effort = (gains["kp"] * (goal - q) - gains["kd"] * qd).clamp(
                -gains["limit"], gains["limit"]
            )
            effort = effort.reshape(-1, 1)
            asset.set_joint_effort_target_index(target=effort, joint_ids=[joint_id])
            return float(effort[0, 0].item())

        def update_stats(kind: str, obs, effort_target: float) -> None:
            index = {"knob": KNOB, "lever": LEVER, "button": BUTTON}[kind]
            asset = control_assets[kind]
            joint_id = control_joint_ids[kind]
            q = float(asset.data.joint_pos.torch[0, joint_id].item())
            qd = float(asset.data.joint_vel.torch[0, joint_id].item())
            progress = float(obs["panel_state"][0, index].item())
            entry = stats[kind]
            entry["q_min"] = min(entry["q_min"], q)
            entry["q_max"] = max(entry["q_max"], q)
            entry["peak_abs_qd"] = max(entry["peak_abs_qd"], abs(qd))
            entry["peak_progress"] = max(entry["peak_progress"], progress)
            entry["final_q"] = q
            entry["max_abs_effort_target"] = max(entry["max_abs_effort_target"], abs(effort_target))

        frame_stride = max(1, round((1.0 / cfg.dt) / args.capture_fps))
        step = 0
        wall_start = time.perf_counter()
        print(
            "[DEMO] panel-control mobility: robot arm parked, controls driven through their own articulations"
        )
        print(f"[DEMO] physics={args.physics_profile} hz={cfg.physics_hz} substeps={cfg.solver_substeps}")
        while task.time_s < cfg.episode_s:
            now = task.time_s
            if now < phases["knob"][0]:
                drive_goal = 0.0
            else:
                drive_goal = 0.0
                for kind, (start_s, ramp_s, hold_s) in phases.items():
                    end_s = start_s + 2.0 * ramp_s + hold_s
                    if now < start_s or now >= end_s:
                        continue
                    local = now - start_s
                    if local < ramp_s:
                        drive_goal = ramp_goal(local, 0.0, ramp_s, goals[kind])
                    elif local < ramp_s + hold_s:
                        drive_goal = goals[kind]
                    else:
                        drive_goal = ramp_goal(
                            local - ramp_s - hold_s, 0.0, ramp_s, 0.0
                        )
                    break
            efforts = {
                kind: drive_control(kind, drive_goal if kind == active_kind(now, phases) else 0.0)
                for kind in ("knob", "lever", "button")
            }
            obs = task.step(arm, fingers)
            for kind in ("knob", "lever", "button"):
                update_stats(kind, obs, efforts[kind])
            if step % frame_stride == 0:
                recorder.add_frame(task, controller, obs)
            if step % 240 == 0:
                state = obs["panel_state"][0].tolist()
                print(
                    f"[DEMO] t={task.time_s:5.2f}s state={[round(float(v), 3) for v in state]} "
                    f"q=({stats['knob']['final_q']:+.4f},{stats['lever']['final_q']:+.4f},"
                    f"{stats['button']['final_q']*1000:+.3f}mm)"
                )
            step += 1
        recorder.close()
        wall_elapsed_s = max(time.perf_counter() - wall_start, 1.0e-9)

        for kind in ("knob", "lever", "button"):
            entry = stats[kind]
            entry["returned_to_rest"] = abs(entry["final_q"]) < max(
                1.0e-3, 0.02 * abs(entry["goal"])
            )
        payload = {
            "demo": "panel_control_mobility_without_arm_operation",
            "physics_profile": args.physics_profile,
            "physics_hz": cfg.physics_hz,
            "solver_substeps": cfg.solver_substeps,
            "episode_s": cfg.episode_s,
            "controls": stats,
            "max_penetration_mm": task.metrics.max_penetration_mm,
            "max_penetration_pair": task.metrics.max_penetration_pair,
            "penetration_frames_over_0p5mm": task.metrics.penetration_frames_over_0p5mm,
            "max_finger_contact_n": max(
                task.metrics.max_left_finger_contact_n,
                task.metrics.max_right_finger_contact_n,
            ),
            "wall_elapsed_s": wall_elapsed_s,
        }
        args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"[DEMO] metrics={args.metrics_output.resolve()}")
        print(f"[DEMO] video={args.output.resolve()} frames={recorder.frames}")
        print("[DEMO] summary:", json.dumps(stats, indent=2))
        return 0


def active_kind(now: float, phases) -> str:
    for kind, (start_s, ramp_s, hold_s) in phases.items():
        end_s = start_s + 2.0 * ramp_s + hold_s
        if start_s <= now < end_s:
            return kind
    return "knob"


if __name__ == "__main__":
    raise SystemExit(main())
