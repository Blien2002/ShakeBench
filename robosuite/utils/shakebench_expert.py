"""Privileged collection-expert state, separate from learner observations."""

import numpy as np

from robosuite.utils import transform_utils as T
from robosuite.utils.shakebench_providers import COMMON_STATE_KEYS, POLICY_FIELD_CONTRACT

ORACLE_STATE_KEYS = COMMON_STATE_KEYS


def oracle_observation(env, observation=None):
    """Read current compiled state for the expert without changing learner inputs.

    ``observation`` is accepted for video callbacks; all expert values come from
    the live simulation, including object-neutral task variants. No future
    excitation or evaluator outcome is included.
    """
    robot, data = env.robots[0], env.sim.data
    base = np.asarray(data.xpos[env.robot_base_body_id])
    rotation = np.asarray(data.xmat[env.robot_base_body_id]).reshape(3, 3).T
    hand = robot.pose_in_base_from_name(env.gripper_body_name)
    table_rotation = np.asarray(data.xmat[env.worktable_body_id]).reshape(3, 3)
    goal = data.xpos[env.worktable_body_id] + table_rotation @ env.metrics.target_frame_local_origin_m
    values = {
        "robot0_joint_pos": data.qpos[robot._ref_joint_pos_indexes],
        "robot0_joint_vel": data.qvel[robot._ref_joint_vel_indexes],
        "robot0_eef_pos_robot_base": hand[:3, 3],
        "robot0_eef_quat_robot_base": T.mat2quat(hand[:3, :3]),
        "robot0_gripper_state": np.concatenate((
            data.qpos[robot._ref_gripper_joint_pos_indexes["right"]],
            data.qvel[robot._ref_gripper_joint_vel_indexes["right"]],
        )),
        "robot0_wrist_force": robot.ee_force["right"],
        "robot0_wrist_torque": robot.ee_torque["right"],
        "robot0_fingertip_pos_robot_base": np.concatenate([
            rotation @ (data.geom_xpos[env.sim.model.geom_name2id(name)] - base)
            for name in env.finger_pad_geom_names
        ]),
        "object_pos_robot_base": rotation @ (data.xpos[env.can_body_id] - base),
        "object_quat_robot_base": T.mat2quat(rotation @ np.asarray(data.xmat[env.can_body_id]).reshape(3, 3)),
        "goal_frame_pos_robot_base": rotation @ (goal - base),
        "goal_frame_quat_robot_base": T.mat2quat(rotation @ table_rotation),
        "goal_inner_half_extents_target": np.asarray(env.target_inner_xy_m) / 2,
        "goal_z_bounds_target": (0, env.arena.target_container_spec["wall_height_m"]),
        "goal_orientation_mask": np.zeros(3, dtype=bool),
    }
    return {key: np.asarray(values[key], dtype=POLICY_FIELD_CONTRACT[key]["dtype"]).copy()
            for key in ORACLE_STATE_KEYS}
