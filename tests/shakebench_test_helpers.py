"""Shared public observation fixture builder for the current oracle contract."""

from __future__ import annotations

import numpy as np


def build_current_observation(
    *,
    goal_frame_pos=(0.0, 0.0, 0.0),
    can_pos=(0.1, -0.1, 0.04),
    eef_pos=(0.0, 0.0, 0.4),
    gripper_state=(0.03, -0.03, 0.0, 0.0),
    fingertip_pos=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    imu_window=None,
):
    """Build the single current contract: task fields plus the worktable IMU."""

    values = {
        "robot0_joint_pos": np.zeros(7),
        "robot0_joint_vel": np.zeros(7),
        "robot0_eef_pos_robot_base": np.asarray(eef_pos, dtype=np.float32),
        "robot0_eef_quat_robot_base": np.array((0.0, 0.0, 0.0, 1.0), dtype=np.float32),
        "robot0_gripper_state": np.asarray(gripper_state, dtype=np.float32),
        "robot0_wrist_force": np.zeros(3, dtype=np.float32),
        "robot0_wrist_torque": np.zeros(3, dtype=np.float32),
        "robot0_fingertip_pos_robot_base": np.asarray(fingertip_pos, dtype=np.float32),
        "object_pos_robot_base": np.asarray(can_pos, dtype=np.float32),
        "object_quat_robot_base": np.array((0.0, 0.0, 0.0, 1.0), dtype=np.float32),
        "goal_frame_pos_robot_base": np.asarray(goal_frame_pos, dtype=np.float32),
        "goal_frame_quat_robot_base": np.array((0.0, 0.0, 0.0, 1.0), dtype=np.float32),
        "goal_inner_half_extents_target": np.array((0.082, 0.072), dtype=np.float32),
        "goal_z_bounds_target": np.array((0.0, 0.035), dtype=np.float32),
        "goal_orientation_mask": np.zeros(3, dtype=np.bool_),
        "table_imu_window": (
            np.asarray(imu_window, dtype=np.float32)
            if imu_window is not None
            else np.tile((0.0, 0.0, 9.81, 0.0, 0.0, 0.0), (10, 1)).astype(np.float32)
        ),
        "table_imu_dt_s": np.float32(0.005),
        "table_imu_timestamps_s": np.arange(10, dtype=np.float64) * 0.005,
    }
    return values
