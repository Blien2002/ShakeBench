"""Shared public State-tier fixture builder for Phase 07 tests."""

from __future__ import annotations

import numpy as np


def build_tier_observation(
    tier: str = "V0",
    *,
    goal_frame_pos=(0.0, 0.0, 0.0),
    can_pos=(0.1, -0.1, 0.04),
    eef_pos=(0.0, 0.0, 0.4),
    gripper_state=(0.03, -0.03, 0.0, 0.0),
    fingertip_pos=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    table_twist=None,
    table_accel=None,
    imu_window=None,
):
    if tier not in {"V0", "V1", "V2", "V3"}:
        raise ValueError(tier)
    values = {
        "robot0_joint_pos": np.zeros(7),
        "robot0_joint_vel": np.zeros(7),
        "robot0_eef_pos_robot_base": np.asarray(eef_pos, dtype=np.float32),
        "robot0_eef_quat_robot_base": np.array((0.0, 0.0, 0.0, 1.0), dtype=np.float32),
        "robot0_gripper_state": np.asarray(gripper_state, dtype=np.float32),
        "robot0_wrist_force": np.zeros(3, dtype=np.float32),
        "robot0_wrist_torque": np.zeros(3, dtype=np.float32),
        "robot0_fingertip_pos_robot_base": np.asarray(fingertip_pos, dtype=np.float32),
        "can_pos_robot_base": np.asarray(can_pos, dtype=np.float32),
        "can_quat_robot_base": np.array((0.0, 0.0, 0.0, 1.0), dtype=np.float32),
        "goal_frame_pos_robot_base": np.asarray(goal_frame_pos, dtype=np.float32),
        "goal_frame_quat_robot_base": np.array((0.0, 0.0, 0.0, 1.0), dtype=np.float32),
        "goal_inner_half_extents_target": np.array((0.082, 0.072), dtype=np.float32),
        "goal_z_bounds_target": np.array((0.0, 0.035), dtype=np.float32),
        "goal_orientation_mask": np.zeros(3, dtype=np.bool_),
    }
    if tier in {"V1", "V2", "V3"}:
        values.update(
            {
                "deck_imu_window": (
                    np.asarray(imu_window, dtype=np.float32)
                    if imu_window is not None
                    else np.tile((0.0, 0.0, 9.81, 0.0, 0.0, 0.0), (10, 1)).astype(np.float32)
                ),
                "deck_imu_dt_s": np.float32(0.005),
            }
        )
    if tier in {"V2", "V3"}:
        values.update(
            {
                "deck_pose_in_nominal_frame": np.array((0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0), dtype=np.float32),
                "deck_twist_in_nominal_frame": np.zeros(6, dtype=np.float32),
                "deck_accel_in_nominal_frame": np.zeros(6, dtype=np.float32),
                "table_pose_in_deck_frame": np.array((0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0), dtype=np.float32),
                "table_twist_in_deck_frame": (
                    np.zeros(6, dtype=np.float32) if table_twist is None else np.asarray(table_twist, dtype=np.float32)
                ),
                "table_accel_in_deck_frame": (
                    np.zeros(6, dtype=np.float32) if table_accel is None else np.asarray(table_accel, dtype=np.float32)
                ),
            }
        )
    if tier == "V3":
        values.update(
            {
                "line_accel_amplitude": np.zeros((6, 12), dtype=np.float32),
                "line_omega_rad_s": np.ones((6, 12), dtype=np.float32),
                "line_phase_at_episode_zero": np.zeros((6, 12), dtype=np.float32),
                "line_mask": np.ones((6, 12), dtype=np.bool_),
                "episode_time_s": np.float32(0.0),
                "ramp_type": np.asarray("quintic_smoothstep"),
                "ramp_duration_s": np.float32(1.0),
                "program_frame": np.asarray("deck"),
            }
        )
    return values
