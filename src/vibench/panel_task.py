"""Panel-operation task wrapper on the existing C2 vibrating worktable.

The panel housing follows the C2 table support, while the knob, lever, and
button are one-DoF Newton articulations.  Their progress is derived from the
simulated joint state, never from a scripted visual counter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

import torch

from isaaclab.assets import Articulation, RigidObject, RigidObjectCollection
from isaaclab.scene import InteractiveScene
from isaaclab.sensors import ContactSensor, JointWrenchSensor
from isaaclab.utils.math import quat_apply, quat_from_euler_xyz, subtract_frame_transforms

from .config import CONTROL_KINDS, BenchmarkConfig
from .diagnostics import PenetrationSample, collision_shape_geometry, penetration_probe
from .panel import CONTROL_INDEX, control_panel_layout, padded_sequence_ids
from .shaker import solve_leg_transforms
from .supports import support_group_geometries, support_pose_velocity, write_support_groups
from .vibration import SpectralVibration
from .wrist_camera import NewtonGlWristCameraSensor, wrist_camera_frame_from_hand


@dataclass
class PanelEpisodeMetrics:
    success: bool = False
    sequence: tuple[str, ...] = ()
    completed_sequence: tuple[str, ...] = ()
    active_control: str = "none"
    controls_done: tuple[bool, bool, bool] = (False, False, False)
    knob_progress: float = 0.0
    lever_progress: float = 0.0
    button_progress: float = 0.0
    wrong_order: bool = False
    wrong_control_contact: bool = False
    wrong_control_name: str = "none"
    operation_timeout: bool = False
    move_timeout: bool = False
    contact_lost: bool = False
    max_wrist_force_n: float = 0.0
    max_wrist_torque_nm: float = 0.0
    max_left_finger_contact_n: float = 0.0
    max_right_finger_contact_n: float = 0.0
    max_knob_contact_n: float = 0.0
    max_lever_contact_n: float = 0.0
    max_button_contact_n: float = 0.0
    max_penetration_mm: float = 0.0
    max_penetration_pair: str = "none"
    max_penetration_shapes: tuple[str, str] = ("", "")
    max_penetration_t: float = 0.0
    penetration_frames_over_0p5mm: float = 0.0
    _control_contact_streak: dict[str, int] = field(default_factory=dict, repr=False)


class PanelBenchmarkTask:
    """Task API for the fixed control-panel manipulation benchmark."""

    def __init__(self, sim: Any, scene: InteractiveScene, cfg: BenchmarkConfig):
        self.sim = sim
        self.scene = scene
        self.cfg = cfg
        self.robot: Articulation = scene["robot"]
        self.platform: RigidObject = scene["platform"]
        self.shaker_legs: RigidObjectCollection = scene["shaker_legs"]
        self.worktable: RigidObject = scene["worktable"]
        self.table_legs: tuple[RigidObject, ...] = (
            scene["table_leg_fl"],
            scene["table_leg_fr"],
            scene["table_leg_rl"],
            scene["table_leg_rr"],
        )
        self.panel: RigidObject = scene["panel"]
        self.knob: Articulation = scene["knob"]
        self.lever: Articulation = scene["lever"]
        self.button: Articulation = scene["button"]
        try:
            self.wrist_wrench: JointWrenchSensor | None = scene["wrist_wrench"]
        except KeyError:
            self.wrist_wrench = None
        self.knob_contact_left: ContactSensor = scene["knob_contact_left"]
        self.knob_contact_right: ContactSensor = scene["knob_contact_right"]
        self.lever_contact_left: ContactSensor = scene["lever_contact_left"]
        self.lever_contact_right: ContactSensor = scene["lever_contact_right"]
        self.button_contact_left: ContactSensor = scene["button_contact_left"]
        self.button_contact_right: ContactSensor = scene["button_contact_right"]
        self.device = self.robot.device
        self.num_envs = scene.num_envs
        self.time_s = 0.0
        self.metrics = PanelEpisodeMetrics()
        self.panel_sequence = cfg.panel.resolved_sequence()
        self.vibration = SpectralVibration(cfg.vibration, self.num_envs, self.device)
        self.wrist_camera = NewtonGlWristCameraSensor()

        finger_shapes = [
            entry
            for entry in collision_shape_geometry("/Robot/panda_leftfinger")
            if "/collisions/" in str(entry["label"]).lower()
            and entry["mesh_extent_m"] is not None
            and entry["mesh_center_m"] is not None
        ]
        self.finger_downward_reach_m = max(
            float(entry["mesh_center_m"][2]) + 0.5 * float(entry["mesh_extent_m"][2])
            for entry in finger_shapes
        ) if finger_shapes else 0.054

        self.arm_joint_ids = self._joints(["panda_joint[1-7]"])
        self.finger_joint_ids = self._joints(["panda_finger_joint.*"])
        self.ee_body_id = self._body("panda_hand")
        self.left_finger_body_id = self._body("panda_leftfinger")
        self.right_finger_body_id = self._body("panda_rightfinger")
        self.knob_joint_id = self.knob.find_joints(["knob_joint"])[0][0]
        self.lever_joint_id = self.lever.find_joints(["lever_joint"])[0][0]
        self.button_joint_id = self.button.find_joints(["button_joint"])[0][0]
        self.knob_link_id = self.knob.find_bodies("knob_link")[0][0]
        self.lever_link_id = self.lever.find_bodies("lever_link")[0][0]
        self.button_link_id = self.button.find_bodies("button_link")[0][0]
        self.ee_jacobian_id = self.ee_body_id if not self.robot.is_fixed_base else self.ee_body_id - 1
        self.wrench_body_id = -1
        if self.wrist_wrench is not None and "panda_link7" in self.wrist_wrench.body_names:
            self.wrench_body_id = self.wrist_wrench.body_names.index("panda_link7")

        layout = control_panel_layout(cfg)
        self.layout = layout
        self._panel_local = self._repeat3(layout.board_center)
        self._knob_pivot_local = self._repeat3(layout.knob_pivot)
        self._lever_pivot_local = self._repeat3(layout.lever_pivot)
        self._button_pivot_local = self._repeat3(layout.button_pivot)
        depth = cfg.panel.console_depth_m
        height = cfg.panel.console_height_m
        front_top_z = -0.5 * height + cfg.panel.console_front_height_m
        shoulder_x = 0.5 * depth - cfg.panel.console_rear_flat_depth_m
        slope_dx = shoulder_x + 0.5 * depth
        slope_dz = 0.5 * height - front_top_z
        slope_length = math.hypot(slope_dx, slope_dz)
        self._surface_tangent_local = self._repeat3(
            (slope_dx / slope_length, 0.0, slope_dz / slope_length)
        )
        self._surface_normal_local = self._repeat3(
            (-slope_dz / slope_length, 0.0, slope_dx / slope_length)
        )
        self._robot_local = self._repeat3(cfg.resolved_robot_base)
        self._platform_local = self._repeat3(cfg.platform_center)
        self._worktable_local = self._repeat3(cfg.resolved_worktable_center)
        self._table_leg_local = [leg.data.default_root_pose.torch[:, :3].clone() for leg in self.table_legs]

        self._support_groups = support_group_geometries(cfg)
        self._vibration_q = torch.zeros((self.num_envs, 6), device=self.device)
        self._vibration_qd = torch.zeros_like(self._vibration_q)
        self._vibration_qdd = torch.zeros_like(self._vibration_q)
        self._arm_table_delta_z = torch.zeros((self.num_envs, 1), device=self.device)
        self._platform_quat = torch.zeros((self.num_envs, 4), device=self.device)
        self._platform_quat[:, 0] = 1.0
        self._shaker_leg_lengths = torch.zeros((self.num_envs, 6), device=self.device)

        self._control_state = torch.zeros((self.num_envs, len(CONTROL_KINDS)), device=self.device)
        self._control_state_dot = torch.zeros_like(self._control_state)
        self._control_peak_state = torch.zeros_like(self._control_state)
        self._control_completed = torch.zeros(
            (self.num_envs, len(CONTROL_KINDS)), device=self.device, dtype=torch.bool
        )
        self._active_control_index = -1
        self._active_control = ""
        self._knob_contact_left_n = torch.zeros(self.num_envs, device=self.device)
        self._knob_contact_right_n = torch.zeros(self.num_envs, device=self.device)
        self._lever_contact_left_n = torch.zeros(self.num_envs, device=self.device)
        self._lever_contact_right_n = torch.zeros(self.num_envs, device=self.device)
        self._button_contact_left_n = torch.zeros(self.num_envs, device=self.device)
        self._button_contact_right_n = torch.zeros(self.num_envs, device=self.device)
        self._wrong_contact_streak = {kind: 0 for kind in CONTROL_KINDS}

        self._current_penetration = PenetrationSample()
        self._penetration_frame_count = 0
        self._penetration_over_limit_count = 0
        history_n = int(round(4.0 / cfg.dt)) + 2
        self._history_t = torch.zeros(history_n, device=self.device)
        self._history_q = torch.zeros((history_n, 6), device=self.device)
        self._history_count = 0
        self._history_index = 0

    def _repeat3(self, xyz: tuple[float, float, float]) -> torch.Tensor:
        return torch.tensor(xyz, dtype=torch.float32, device=self.device).repeat(self.num_envs, 1)

    def _joints(self, expressions: list[str]) -> list[int]:
        ids, _ = self.robot.find_joints(expressions)
        if not ids:
            raise RuntimeError(f"No joints matched {expressions}; available={self.robot.joint_names}")
        return ids

    def _body(self, expression: str) -> int:
        ids, _ = self.robot.find_bodies(expression)
        if len(ids) != 1:
            raise RuntimeError(f"Expected one body for {expression}; available={self.robot.body_names}")
        return ids[0]

    def _support_state(
        self,
        local: torch.Tensor,
        motion: torch.Tensor,
        velocity_motion: torch.Tensor,
        rotation_anchor: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del rotation_anchor
        return support_pose_velocity(
            local,
            motion,
            velocity_motion,
            self.cfg.platform_center,
            self.scene.env_origins,
        )

    def _write_supports(self) -> None:
        self._platform_quat = quat_from_euler_xyz(
            self._vibration_q[:, 3], self._vibration_q[:, 4], self._vibration_q[:, 5]
        )
        write_support_groups(
            groups=self._support_groups,
            scene=self.scene,
            q_deck=self._vibration_q,
            qd_deck=self._vibration_qd,
        )
        panel_position, panel_quat, panel_velocity = self._support_state(
            self._panel_local, self._vibration_q, self._vibration_qd
        )
        self.panel.write_root_pose_to_sim_index(
            root_pose=torch.cat((panel_position, panel_quat), dim=1)
        )
        self.panel.write_root_velocity_to_sim_index(root_velocity=panel_velocity)

        deck_top = self.cfg.platform_center[2] + 0.5 * self.cfg.platform_size[2]
        robot_surface_local = self._repeat3((*self.cfg.robot_base[:2], deck_top))
        table_surface_local = self._repeat3((*self.cfg.worktable_center[:2], deck_top))
        robot_position, _, _ = support_pose_velocity(
            robot_surface_local,
            self._vibration_q,
            self._vibration_qd,
            self.cfg.platform_center,
            self.scene.env_origins,
        )
        table_position, _, _ = support_pose_velocity(
            table_surface_local,
            self._vibration_q,
            self._vibration_qd,
            self.cfg.platform_center,
            self.scene.env_origins,
        )
        self._arm_table_delta_z = (robot_position[:, 2] - table_position[:, 2]).unsqueeze(1)

        self._write_control_pose("knob")
        self._write_control_pose("lever")
        self._write_control_pose("button")

        platen_position, platen_quat, platen_velocity = self._support_state(
            self._platform_local, self._vibration_q, self._vibration_qd
        )
        platen_pose = torch.cat((platen_position - self.scene.env_origins, platen_quat), dim=1)
        legs = solve_leg_transforms(platen_pose, self.cfg.shaker)
        self._shaker_leg_lengths = legs.lengths_m
        shadow_pose, shadow_velocity = self._panel_shadow_state()
        leg_pose = torch.cat((legs.outer_pose_xyzw, legs.rod_pose_xyzw, shadow_pose.unsqueeze(1)), dim=1)
        leg_pose[..., :3] += self.scene.env_origins[:, None, :]
        self.shaker_legs.write_body_pose_to_sim_index(body_poses=leg_pose)
        linear_velocity = platen_velocity[:, None, :3].expand(-1, 12, -1)
        angular_velocity = platen_velocity[:, None, 3:].expand(-1, 12, -1)
        leg_velocity = torch.cat((linear_velocity, angular_velocity), dim=-1)
        self.shaker_legs.write_body_velocity_to_sim_index(
            body_velocities=torch.cat((leg_velocity, shadow_velocity.unsqueeze(1)), dim=1)
        )

    def _control_layout(self, kind: str):
        return self.layout.control(kind)

    def _write_control_pose(self, kind: str) -> None:
        local_by_kind = {
            "knob": self._knob_pivot_local,
            "lever": self._lever_pivot_local,
            "button": self._button_pivot_local,
        }
        asset_by_kind = {"knob": self.knob, "lever": self.lever, "button": self.button}
        if kind not in local_by_kind:
            raise ValueError(f"unknown panel control: {kind}")
        position, quat, velocity = self._support_state(
            local_by_kind[kind],
            self._vibration_q,
            self._vibration_qd,
            self._worktable_local,
        )
        asset = asset_by_kind[kind]
        asset.write_root_pose_to_sim_index(root_pose=torch.cat((position, quat), dim=1))
        asset.write_root_velocity_to_sim_index(root_velocity=velocity)

    def _panel_shadow_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Project a contact shadow onto the moving tabletop below the panel."""

        table_top_local = self._worktable_local.clone()
        table_top_local[:, 2] += 0.5 * self.cfg.worktable_size[2] + 0.001
        surface_position, surface_quat, surface_velocity = self._support_state(
            table_top_local,
            self._vibration_q,
            self._vibration_qd,
            self._worktable_local,
        )
        shadow_pose = torch.cat((surface_position - self.scene.env_origins, surface_quat), dim=1)
        panel_xy = self._panel_local[:, :2].clone()
        panel_xy -= self.scene.env_origins[:, :2]
        shadow_pose[:, :2] = panel_xy
        return shadow_pose, surface_velocity

    def _update_control_state_from_joints(self) -> None:
        """Resolve normalized progress from the three simulated joints."""

        knob_q = self.knob.data.joint_pos.torch[:, self.knob_joint_id]
        lever_q = self.lever.data.joint_pos.torch[:, self.lever_joint_id]
        button_q = self.button.data.joint_pos.torch[:, self.button_joint_id]
        knob_qd = self.knob.data.joint_vel.torch[:, self.knob_joint_id]
        lever_qd = self.lever.data.joint_vel.torch[:, self.lever_joint_id]
        button_qd = self.button.data.joint_vel.torch[:, self.button_joint_id]
        self._control_state = torch.stack(
            (
                knob_q / self.cfg.panel.knob_goal_rad,
                lever_q / self.cfg.panel.lever_goal_rad,
                -button_q / self.cfg.panel.button_travel_m,
            ),
            dim=1,
        ).clamp(0.0, 1.0)
        self._control_state_dot = torch.stack(
            (
                knob_qd / self.cfg.panel.knob_goal_rad,
                lever_qd / self.cfg.panel.lever_goal_rad,
                -button_qd / self.cfg.panel.button_travel_m,
            ),
            dim=1,
        )
        self._control_peak_state = torch.maximum(self._control_peak_state, self._control_state)

    def reset(self) -> dict[str, torch.Tensor]:
        self.scene.reset()
        joint_pos = self.robot.data.default_joint_pos.torch.clone()
        joint_vel = torch.zeros_like(joint_pos)
        self.robot.write_joint_position_to_sim_index(position=joint_pos)
        self.robot.write_joint_velocity_to_sim_index(velocity=joint_vel)
        self.robot.set_joint_position_target(joint_pos)

        self.time_s = 0.0
        self.vibration.reseed(self.cfg.vibration.seed)
        self._vibration_q.zero_()
        self._vibration_qd.zero_()
        self._vibration_qdd.zero_()
        self._control_state.zero_()
        self._control_state_dot.zero_()
        self._control_peak_state.zero_()
        self._control_completed.zero_()
        for control in (self.knob, self.lever, self.button):
            control_joint_pos = control.data.default_joint_pos.torch.clone()
            control_joint_vel = torch.zeros_like(control_joint_pos)
            control.write_joint_position_to_sim_index(position=control_joint_pos)
            control.write_joint_velocity_to_sim_index(velocity=control_joint_vel)
            control.set_joint_position_target(control_joint_pos)
        self._active_control_index = -1
        self._active_control = ""
        self._write_supports()
        self.scene.write_data_to_sim()
        self.sim.step()
        self.scene.update(self.cfg.dt)
        self._update_control_state_from_joints()

        self.metrics = PanelEpisodeMetrics()
        self._current_penetration = PenetrationSample()
        self._penetration_frame_count = 0
        self._penetration_over_limit_count = 0
        self._knob_contact_left_n.zero_()
        self._knob_contact_right_n.zero_()
        self._lever_contact_left_n.zero_()
        self._lever_contact_right_n.zero_()
        self._button_contact_left_n.zero_()
        self._button_contact_right_n.zero_()
        self._wrong_contact_streak = {kind: 0 for kind in CONTROL_KINDS}
        self._history_count = 0
        self._history_index = 0
        self._update_contact_forces()
        self._update_penetration_metrics()
        return self.observation()

    def set_active_control(self, kind: str) -> None:
        index = self.panel_sequence.index(kind) if kind in self.panel_sequence else -1
        if index < 0:
            self.metrics.wrong_order = True
            return
        previous_done = True
        if index > 0:
            previous_kind = self.panel_sequence[index - 1]
            previous_done = bool(self._control_completed[0, CONTROL_INDEX[previous_kind]].item())
        if index < self._active_control_index or not previous_done:
            self.metrics.wrong_order = True
        self._active_control_index = index
        self._active_control = kind

    def request_panel_progress(self, kind: str, progress: float) -> None:
        raise RuntimeError(
            "panel progress is read from physical joint state; direct scripted progress is disabled"
        )

    def mark_control_complete(self, kind: str) -> None:
        if self._active_control != kind:
            self.metrics.wrong_order = True
        index = CONTROL_INDEX[kind]
        self._control_completed[:, index] |= self._control_peak_state[:, index] >= 0.95

    def mark_operation_timeout(self) -> None:
        self.metrics.operation_timeout = True

    def mark_move_timeout(self) -> None:
        self.metrics.move_timeout = True

    def mark_contact_lost(self) -> None:
        self.metrics.contact_lost = True

    def _filtered_contact_force(self, sensor: ContactSensor) -> torch.Tensor:
        matrix = sensor.data.force_matrix_w
        if matrix is None:
            return torch.zeros(self.num_envs, device=self.device)
        values = matrix.torch
        return torch.linalg.vector_norm(values, dim=-1).flatten(1).amax(dim=1)

    def _update_contact_forces(self) -> None:
        self._knob_contact_left_n = self._filtered_contact_force(self.knob_contact_left)
        self._knob_contact_right_n = self._filtered_contact_force(self.knob_contact_right)
        self._lever_contact_left_n = self._filtered_contact_force(self.lever_contact_left)
        self._lever_contact_right_n = self._filtered_contact_force(self.lever_contact_right)
        self._button_contact_left_n = self._filtered_contact_force(self.button_contact_left)
        self._button_contact_right_n = self._filtered_contact_force(self.button_contact_right)

    def contact_force_n(self, kind: str) -> tuple[torch.Tensor, torch.Tensor]:
        if kind == "knob":
            return self._knob_contact_left_n, self._knob_contact_right_n
        if kind == "lever":
            return self._lever_contact_left_n, self._lever_contact_right_n
        if kind == "button":
            return self._button_contact_left_n, self._button_contact_right_n
        raise ValueError(f"unknown panel control: {kind}")

    def step(self, arm_target: torch.Tensor, finger_target: torch.Tensor) -> dict[str, torch.Tensor]:
        self._vibration_q, self._vibration_qd, self._vibration_qdd = self.vibration.sample(self.time_s)
        self._write_supports()
        self._update_contact_forces()
        self.robot.set_joint_position_target_index(target=arm_target, joint_ids=self.arm_joint_ids)
        self.robot.set_joint_position_target_index(target=finger_target, joint_ids=self.finger_joint_ids)
        self.scene.write_data_to_sim()
        self.sim.step()
        self.scene.update(self.cfg.dt)
        self._update_control_state_from_joints()
        self.time_s += self.cfg.dt
        self._update_penetration_metrics()

        self._history_t[self._history_index] = self.time_s
        self._history_q[self._history_index] = self._vibration_q[0]
        self._history_index = (self._history_index + 1) % len(self._history_t)
        self._history_count = min(self._history_count + 1, len(self._history_t))
        obs = self.observation()
        self._update_metrics(obs)
        return obs

    def observation(self) -> dict[str, torch.Tensor]:
        hand = self.robot.data.body_pose_w.torch[:, self.ee_body_id]
        root = self.robot.data.root_pose_w.torch
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root[:, :3], root[:, 3:7], hand[:, :3], hand[:, 3:7]
        )
        finger_center_w = 0.5 * (
            self.robot.data.body_pose_w.torch[:, self.left_finger_body_id, :3]
            + self.robot.data.body_pose_w.torch[:, self.right_finger_body_id, :3]
        )
        finger_center_b, _ = subtract_frame_transforms(
            root[:, :3], root[:, 3:7], finger_center_w, hand[:, 3:7]
        )

        panel_w = self.panel.data.root_pose_w.torch
        knob_w = self.knob.data.body_pose_w.torch[:, self.knob_link_id]
        lever_w = self.lever.data.body_pose_w.torch[:, self.lever_link_id]
        button_w = self.button.data.body_pose_w.torch[:, self.button_link_id]

        lever_axis_w = quat_apply(
            lever_w[:, 3:7],
            self._surface_normal_local * float(self.layout.lever.length_m),
        )
        lever_tip_w = torch.cat(
            (lever_w[:, :3] + lever_axis_w, lever_w[:, 3:7]), dim=1
        )
        lever_pivot_w = torch.cat((lever_w[:, :3], lever_w[:, 3:7]), dim=1)
        button_axis_w = quat_apply(
            button_w[:, 3:7],
            self._surface_normal_local * 0.034,
        )
        button_face_w = torch.cat(
            (button_w[:, :3] + button_axis_w, button_w[:, 3:7]), dim=1
        )

        def in_base(world_pose: torch.Tensor) -> torch.Tensor:
            pos, quat = subtract_frame_transforms(
                root[:, :3], root[:, 3:7], world_pose[:, :3], world_pose[:, 3:7]
            )
            return torch.cat((pos, quat), dim=1)

        wrist_camera_eye_w, wrist_camera_target_w, wrist_camera_up_w = wrist_camera_frame_from_hand(hand)
        if self.wrist_wrench is not None and self.wrench_body_id >= 0:
            force = self.wrist_wrench.data.force.torch[:, self.wrench_body_id]
            torque = self.wrist_wrench.data.torque.torch[:, self.wrench_body_id]
        else:
            force = torch.zeros((self.num_envs, 3), device=self.device)
            torque = torch.zeros_like(force)
        sequence_ids = padded_sequence_ids(self.panel_sequence)
        sequence_tensor = torch.tensor(sequence_ids, dtype=torch.int32, device=self.device).repeat(
            self.num_envs, 1
        )
        return {
            "joint_pos": self.robot.data.joint_pos.torch,
            "left_finger_pose_w": self.robot.data.body_pose_w.torch[:, self.left_finger_body_id],
            "right_finger_pose_w": self.robot.data.body_pose_w.torch[:, self.right_finger_body_id],
            "wrist_camera_eye_w": wrist_camera_eye_w,
            "wrist_camera_target_w": wrist_camera_target_w,
            "wrist_camera_up_w": wrist_camera_up_w,
            "finger_center_b": finger_center_b,
            "ee_pose_w": hand,
            "ee_pose_b": torch.cat((ee_pos_b, ee_quat_b), dim=1),
            "root_pose_w": root,
            "panel_pose_w": panel_w,
            "panel_pose_b": in_base(panel_w),
            "panel_surface_normal_b": quat_apply(
                in_base(panel_w)[:, 3:7], self._surface_normal_local
            ),
            "panel_surface_tangent_b": quat_apply(
                in_base(panel_w)[:, 3:7], self._surface_tangent_local
            ),
            "panel_surface_lateral_b": quat_apply(
                in_base(panel_w)[:, 3:7],
                torch.tensor((0.0, 1.0, 0.0), device=self.device).repeat(self.num_envs, 1),
            ),
            "knob_pose_w": knob_w,
            "knob_pose_b": in_base(knob_w),
            "lever_pose_w": lever_w,
            "lever_pose_b": in_base(lever_w),
            "lever_pivot_w": lever_pivot_w,
            "lever_pivot_b": in_base(lever_pivot_w),
            "lever_tip_w": lever_tip_w,
            "lever_tip_b": in_base(lever_tip_w),
            "button_pose_w": button_w,
            "button_pose_b": in_base(button_w),
            "button_face_w": button_face_w,
            "button_face_b": in_base(button_face_w),
            "panel_sequence_ids": sequence_tensor,
            "panel_sequence_length": torch.full(
                (self.num_envs, 1),
                len(self.panel_sequence),
                dtype=torch.int32,
                device=self.device,
            ),
            "panel_state": self._control_state,
            "panel_state_dot": self._control_state_dot,
            "active_control_index": torch.full(
                (self.num_envs, 1),
                self._active_control_index,
                dtype=torch.int32,
                device=self.device,
            ),
            "vibration_q": self._vibration_q,
            "vibration_qd": self._vibration_qd,
            "vibration_qdd": self._vibration_qdd,
            "mount_delta_z": self._arm_table_delta_z,
            "shaker_leg_lengths_m": self._shaker_leg_lengths,
            "wrist_force_b": force,
            "wrist_torque_b": torque,
            "knob_contact_left_n": self._knob_contact_left_n.unsqueeze(1),
            "knob_contact_right_n": self._knob_contact_right_n.unsqueeze(1),
            "lever_contact_left_n": self._lever_contact_left_n.unsqueeze(1),
            "lever_contact_right_n": self._lever_contact_right_n.unsqueeze(1),
            "button_contact_left_n": self._button_contact_left_n.unsqueeze(1),
            "button_contact_right_n": self._button_contact_right_n.unsqueeze(1),
            "penetration_mm": torch.full(
                (self.num_envs, 1),
                self._current_penetration.depth_mm,
                device=self.device,
            ),
            "grasped": torch.zeros(self.num_envs, device=self.device, dtype=torch.bool),
        }

    def wrist_camera_rgb(self, obs: dict[str, torch.Tensor]) -> Any:
        return self.wrist_camera.render(
            obs["wrist_camera_eye_w"],
            obs["wrist_camera_target_w"],
            obs["wrist_camera_up_w"],
        )

    def arm_jacobian(self) -> torch.Tensor:
        ids = [joint_id + self.robot.num_base_dofs for joint_id in self.arm_joint_ids]
        return self.robot.data.body_link_jacobian_w.torch[:, self.ee_jacobian_id, :, ids]

    def arm_position(self) -> torch.Tensor:
        return self.robot.data.joint_pos.torch[:, self.arm_joint_ids]

    def vibration_history(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._history_count < len(self._history_t):
            return self._history_t[: self._history_count], self._history_q[: self._history_count]
        order = torch.cat(
            (
                torch.arange(self._history_index, len(self._history_t), device=self.device),
                torch.arange(0, self._history_index, device=self.device),
            )
        )
        return self._history_t[order], self._history_q[order]

    def _update_metrics(self, obs: dict[str, torch.Tensor]) -> None:
        metrics = self.metrics
        metrics.sequence = tuple(self.panel_sequence)
        metrics.active_control = self._active_control if self._active_control else "none"
        done = [
            bool(self._control_completed[0, CONTROL_INDEX[kind]].item())
            for kind in CONTROL_KINDS
        ]
        metrics.controls_done = tuple(done)
        metrics.knob_progress = float(self._control_peak_state[0, CONTROL_INDEX["knob"]].item())
        metrics.lever_progress = float(self._control_peak_state[0, CONTROL_INDEX["lever"]].item())
        metrics.button_progress = float(self._control_peak_state[0, CONTROL_INDEX["button"]].item())
        metrics.completed_sequence = tuple(
            kind
            for kind in self.panel_sequence
            if bool(self._control_completed[0, CONTROL_INDEX[kind]].item())
        )
        metrics.success = (
            len(metrics.completed_sequence) == len(self.panel_sequence)
            and not metrics.wrong_order
            and not metrics.wrong_control_contact
        )

        wrist_force_n = float(torch.linalg.norm(obs["wrist_force_b"][0]).item())
        metrics.max_wrist_force_n = max(metrics.max_wrist_force_n, wrist_force_n)
        metrics.max_wrist_torque_nm = max(
            metrics.max_wrist_torque_nm,
            float(torch.linalg.norm(obs["wrist_torque_b"][0]).item()),
        )
        for kind, index in CONTROL_INDEX.items():
            left, right = self.contact_force_n(kind)
            left_n = float(left[0].item())
            right_n = float(right[0].item())
            metrics.max_left_finger_contact_n = max(metrics.max_left_finger_contact_n, left_n)
            metrics.max_right_finger_contact_n = max(metrics.max_right_finger_contact_n, right_n)
            contact_n = max(left_n, right_n)
            if kind == "knob":
                metrics.max_knob_contact_n = max(metrics.max_knob_contact_n, contact_n)
            elif kind == "lever":
                metrics.max_lever_contact_n = max(metrics.max_lever_contact_n, contact_n)
            else:
                metrics.max_button_contact_n = max(metrics.max_button_contact_n, contact_n)
            is_active = kind == self._active_control
            state = float(self._control_state[0, index].item())
            if not is_active and state <= 0.0 and contact_n > self.cfg.panel.contact_threshold_n:
                self._wrong_contact_streak[kind] += 1
            else:
                self._wrong_contact_streak[kind] = 0
            if self._wrong_contact_streak[kind] >= max(
                1, int(round(0.12 / self.cfg.dt))
            ):
                metrics.wrong_control_contact = True
                metrics.wrong_control_name = kind
        metrics.success = metrics.success and not metrics.wrong_control_contact

    def _update_penetration_metrics(self) -> None:
        self._current_penetration = penetration_probe()
        self._penetration_frame_count += 1
        if self._current_penetration.depth_m > 0.0005:
            self._penetration_over_limit_count += 1
        self.metrics.penetration_frames_over_0p5mm = (
            self._penetration_over_limit_count / self._penetration_frame_count
        )
        if self._current_penetration.depth_mm > self.metrics.max_penetration_mm:
            self.metrics.max_penetration_mm = self._current_penetration.depth_mm
            self.metrics.max_penetration_pair = self._current_penetration.pair
            self.metrics.max_penetration_shapes = (
                self._current_penetration.shape0,
                self._current_penetration.shape1,
            )
            self.metrics.max_penetration_t = self.time_s
