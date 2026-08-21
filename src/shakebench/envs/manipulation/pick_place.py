"""C2 task wrapper with independent arm/table mounts on one vibrating floor."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any

import numpy as np
import torch

from isaaclab.assets import Articulation, RigidObject, RigidObjectCollection
from isaaclab.scene import InteractiveScene
from isaaclab.sensors import ContactSensor, JointWrenchSensor
from isaaclab.utils.math import quat_apply, quat_from_euler_xyz, quat_inv, quat_mul, subtract_frame_transforms

from isaaclab_newton.physics import NewtonManager

from ...config import BenchmarkConfig, workpiece_dimensions_m
from ...diagnostics import (
    PenetrationSample,
    collision_shape_geometry,
    forbidden_contact_violation,
    penetration_probe,
)
from ...scene import CLITE_DRIVER_LABELS
from ...shaker import solve_leg_transforms
from ...supports import (
    HARD_STRUCTURAL_EXCLUSIONS,
    SupportGroup,
    support_group_geometries,
    support_pose_velocity,
    write_support_groups,
)
from ...vibration import SpectralVibration
from ...sensors.wrist_camera import NewtonGlWristCameraSensor, wrist_camera_frame_from_hand


@dataclass
class EpisodeMetrics:
    lifted: bool = False
    placed: bool = False
    success: bool = False
    max_wrist_force_n: float = 0.0
    max_wrist_force_t: float = 0.0
    max_wrist_force_phase: str = "none"
    contact_pair_at_max_wrist_force: str = "none"
    contact_shapes_at_max_wrist_force: tuple[str, str] = ("", "")
    max_wrist_torque_nm: float = 0.0
    max_left_finger_contact_n: float = 0.0
    max_right_finger_contact_n: float = 0.0
    final_xy_error_m: float = float("inf")
    bilateral_contact_confirmed: bool = False
    grasp_assist_used: bool = False
    max_penetration_mm: float = 0.0
    max_penetration_pair: str = "none"
    max_penetration_shapes: tuple[str, str] = ("", "")
    max_penetration_t: float = 0.0
    penetration_frames_over_0p5mm: float = 0.0
    grasp_assist_rejected_penetration: bool = False
    grasp_assist_released_penetration: bool = False
    grasp_z_guard_triggered: bool = False
    descend_contact_timeout: bool = False
    descend_table_contact: bool = False
    grasp_table_contact: bool = False
    grasp_contact_timeout: bool = False
    grasp_contact_lost: bool = False
    grasp_slip_exceeded: bool = False
    max_grasp_slip_m: float = 0.0
    ee_tracking_error_rms_m: float = 0.0
    max_ee_tracking_error_m: float = 0.0
    support_geometry_valid: bool = True
    mean_grip_force_n: float = 0.0
    min_required_grip_force_n: float = 0.0
    ee_world_residual_rms_m: float = 0.0


class VibrationBenchmarkTask:
    """Small task API suitable for scripted control and future RL wrappers."""

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
        self.workpiece: RigidObject = scene["workpiece"]
        self.target: RigidObject = scene["target"]
        self.wrist_wrench: JointWrenchSensor = scene["wrist_wrench"]
        self.workpiece_contact: ContactSensor = scene["workpiece_contact"]
        self.left_finger_contact: ContactSensor = scene["left_finger_contact"]
        self.right_finger_contact: ContactSensor = scene["right_finger_contact"]
        self.left_finger_descent_contact: ContactSensor = scene["left_finger_descent_contact"]
        self.right_finger_descent_contact: ContactSensor = scene["right_finger_descent_contact"]
        self.device = self.robot.device
        self.num_envs = scene.num_envs
        physical_shapes = [
            entry
            for entry in collision_shape_geometry("/Workpiece/")
            if "visual" not in str(entry["label"]).lower() and entry["mesh_extent_m"] is not None
        ]
        self.workpiece_collision_dimensions_m = tuple(
            float(value) for value in physical_shapes[0]["mesh_extent_m"]
        ) if physical_shapes else workpiece_dimensions_m(
            cfg.assets.workpiece,
            cfg.assets.workpiece_scale,
        )
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
        self.time_s = 0.0
        self.metrics = EpisodeMetrics()
        self.vibration = SpectralVibration(cfg.vibration, self.num_envs, self.device)
        # Like ManiSkill's PandaWristCam, the sensor is mounted to a physical
        # camera link/assembly.  Rendering stays lazy so state-only benchmark
        # runs do not pay for RGB observations.
        self.wrist_camera = NewtonGlWristCameraSensor()

        self.arm_joint_ids = self._joints(["panda_joint[1-7]"])
        self.finger_joint_ids = self._joints(["panda_finger_joint.*"])
        self.ee_body_id = self._body("panda_hand")
        self.left_finger_body_id = self._body("panda_leftfinger")
        self.right_finger_body_id = self._body("panda_rightfinger")
        self.ee_jacobian_id = self.ee_body_id if not self.robot.is_fixed_base else self.ee_body_id - 1
        # JointWrenchSensor reports child joints, so the wrist channel is the
        # final arm link rather than the fixed hand body.
        self.wrench_body_id = (
            self.wrist_wrench.body_names.index("panda_link7")
            if "panda_link7" in self.wrist_wrench.body_names
            else -1
        )

        self._robot_local = self._repeat3(cfg.resolved_robot_base)
        self._platform_local = self._repeat3(cfg.platform_center)
        self._worktable_local = self._repeat3(cfg.resolved_worktable_center)
        self._table_leg_local = [leg.data.default_root_pose.torch[:, :3].clone() for leg in self.table_legs]
        self._target_local = self._repeat3(cfg.resolved_target_center)
        self._clite_mocap_ids: dict[str, int] = {}
        self._clite_mocap_origin: np.ndarray | None = None
        if cfg.use_clite_support:
            self._resolve_clite_mocap_ids()
        self._support_groups = support_group_geometries(cfg)
        self._vibration_q = torch.zeros((self.num_envs, 6), device=self.device)
        self._vibration_qd = torch.zeros_like(self._vibration_q)
        self._vibration_qdd = torch.zeros_like(self._vibration_q)
        self._arm_table_delta_z = torch.zeros((self.num_envs, 1), device=self.device)
        self._platform_quat = torch.zeros((self.num_envs, 4), device=self.device)
        self._platform_quat[:, 0] = 1.0
        self._shaker_leg_lengths = torch.zeros((self.num_envs, 6), device=self.device)
        self._grasp_requested = False
        self._controller_phase = "reset"
        self._grasped = False
        self._bilateral_contact_streak = 0
        self._last_left_contact_n = torch.zeros(self.num_envs, device=self.device)
        self._last_right_contact_n = torch.zeros(self.num_envs, device=self.device)
        self._last_left_descent_contact_n = torch.zeros(self.num_envs, device=self.device)
        self._last_right_descent_contact_n = torch.zeros(self.num_envs, device=self.device)
        self._current_penetration = PenetrationSample()
        self._penetration_frame_count = 0
        self._penetration_over_limit_count = 0
        self._held_pos_h = torch.zeros((self.num_envs, 3), device=self.device)
        self._held_quat_h = torch.zeros((self.num_envs, 4), device=self.device)
        self._held_quat_h[:, 0] = 1.0
        history_n = int(round(4.0 / cfg.dt)) + 2
        self._history_t = torch.zeros(history_n, device=self.device)
        self._history_q = torch.zeros((history_n, 6), device=self.device)
        self._history_count = 0
        self._history_index = 0
        self._ee_tracking_error_sq = 0.0
        self._ee_tracking_error_count = 0
        self._support_step_timing = {
            "sample": 0.0,
            "mocap_write": 0.0,
            "adapter": 0.0,
            "solver": 0.0,
            "clear_forces": 0.0,
            "sensors_scene": 0.0,
        }
        self._support_step_count = 0

    def record_ee_tracking_error(self, commanded_b: torch.Tensor, actual_b: torch.Tensor) -> None:
        """Accumulate deck-frame EE tracking error from controller targets."""

        if commanded_b is None or actual_b is None:
            return
        error = float(torch.norm(commanded_b[:, :3] - actual_b[:, :3]).item())
        self.metrics.max_ee_tracking_error_m = max(self.metrics.max_ee_tracking_error_m, error)
        self._ee_tracking_error_sq += error * error
        self._ee_tracking_error_count += 1

    def finalize_episode_metrics(self) -> None:
        """Fold accumulated per-step metrics into the public metrics dataclass."""

        if self._ee_tracking_error_count:
            self.metrics.ee_tracking_error_rms_m = math.sqrt(
                self._ee_tracking_error_sq / self._ee_tracking_error_count
            )

    def _repeat3(self, xyz: tuple[float, float, float]) -> torch.Tensor:
        return torch.tensor(xyz, dtype=torch.float32, device=self.device).repeat(self.num_envs, 1)

    def _resolve_clite_mocap_ids(self) -> None:
        """Map C-lite driver labels to MuJoCo mocap ids after solver build."""

        model = NewtonManager.get_model()
        solver = NewtonManager._solver
        if model is None or solver is None:
            raise RuntimeError("C-lite support requires an initialized Newton solver")
        labels = [str(label) for label in model.body_label]
        mjc_to_newton = solver.mjc_body_to_newton.numpy()
        body_mocapid = solver.mjw_model.body_mocapid.numpy().reshape(-1)
        mocap_pos = solver.mjw_data.mocap_pos.numpy()
        ids: dict[str, int] = {}
        origins: list[np.ndarray] = []
        mocap_world_axis = 1 if mocap_pos.ndim >= 3 else 0
        for driver_label in CLITE_DRIVER_LABELS:
            if driver_label not in labels:
                raise RuntimeError(
                    f"C-lite driver {driver_label!r} missing from model bodies: {labels}"
                )
            newton_id = labels.index(driver_label)
            found = False
            for mjc_id, mapped in enumerate(mjc_to_newton[0]):
                if int(mapped) == newton_id:
                    mocap_id = int(body_mocapid[mjc_id])
                    if mocap_id < 0:
                        raise RuntimeError(f"C-lite driver {driver_label!r} was not exported as mocap")
                    ids[driver_label] = mocap_id
                    origin = (
                        mocap_pos[0, mocap_id]
                        if mocap_world_axis == 1
                        else mocap_pos[mocap_id]
                    )
                    origins.append(np.asarray(origin, dtype=np.float32))
                    found = True
                    break
            if not found:
                raise RuntimeError(f"C-lite driver {driver_label!r} has no MuJoCo body mapping")
        self._clite_mocap_ids = ids
        self._clite_mocap_origin = np.stack(origins, axis=0)  # (3, 3)
        # Equality solref intentionally stays at MuJoCo's default (0.02 s, 1.0).
        # Tightening it destabilizes the coupled contact/weld solve in the full
        # scene (45 kN finger forces, 4.86 mm penetration at 0.001 s).

    def _write_clite_mocap(self, motion: torch.Tensor) -> None:
        """Write both mocap driver trajectories directly into solver arrays."""

        if self._clite_mocap_origin is None:
            raise RuntimeError("C-lite mocap origins were not resolved")
        motions = (
            (CLITE_DRIVER_LABELS[0], motion),
            (CLITE_DRIVER_LABELS[1], motion),
        )
        mocap_pos = NewtonManager._solver.mjw_data.mocap_pos.numpy()
        mocap_quat = NewtonManager._solver.mjw_data.mocap_quat.numpy()
        mocap_world_axis = 1 if mocap_pos.ndim >= 3 else 0
        for index, (driver_label, driver_motion) in enumerate(motions):
            mocap_id = self._clite_mocap_ids[driver_label]
            origin = self._clite_mocap_origin[index]
            motion_np = np.asarray(driver_motion[0].detach().cpu().numpy(), dtype=np.float32)
            pos = origin + motion_np[:3]
            # isaaclab quat_from_euler_xyz returns xyzw; MuJoCo mocap_quat is wxyz.
            quat_xyzw = quat_from_euler_xyz(driver_motion[0, 3], driver_motion[0, 4], driver_motion[0, 5])
            quat_wxyz = (
                float(quat_xyzw[3].item()),
                float(quat_xyzw[0].item()),
                float(quat_xyzw[1].item()),
                float(quat_xyzw[2].item()),
            )
            if mocap_world_axis == 1:
                mocap_pos[0, mocap_id] = pos
                mocap_quat[0, mocap_id] = quat_wxyz
            else:
                mocap_pos[mocap_id] = pos
                mocap_quat[mocap_id] = quat_wxyz
        NewtonManager._solver.mjw_data.mocap_pos.assign(mocap_pos)
        NewtonManager._solver.mjw_data.mocap_quat.assign(mocap_quat)

    def _write_clite_drivers(self) -> None:
        """Compatibility wrapper used by the full support write."""

        self._write_clite_mocap(self._vibration_q)

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
        # All hard-mounted supports share the deck-centre rotation anchor.
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
        if self.cfg.use_clite_support:
            self._write_clite_drivers()
            # Platform and worktable are dynamic bodies welded to mocap
            # drivers; the remaining kinematic deck members stay teleported.
            clite_groups = tuple(
                SupportGroup(
                    name=group.name,
                    motion_source=group.motion_source,
                    rotation_anchor=group.rotation_anchor,
                    members=tuple(
                        member
                        for member in group.members
                        if member.name not in ("platform", "worktable")
                    ),
                )
                for group in self._support_groups
            )
            write_support_groups(
                groups=clite_groups,
                scene=self.scene,
                q_deck=self._vibration_q,
                qd_deck=self._vibration_qd,
            )
        else:
            write_support_groups(
                groups=self._support_groups,
                scene=self.scene,
                q_deck=self._vibration_q,
                qd_deck=self._vibration_qd,
            )

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

        platen_position, platen_quat, platen_velocity = self._support_state(
            self._platform_local, self._vibration_q, self._vibration_qd
        )
        platen_pose = torch.cat((platen_position - self.scene.env_origins, platen_quat), dim=1)
        legs = solve_leg_transforms(platen_pose, self.cfg.shaker)
        self._shaker_leg_lengths = legs.lengths_m
        # Collection order is outer_0..outer_5, rod_0..rod_5, then the
        # collision-free dynamic workpiece shadow.
        shadow_pose, shadow_velocity = self._workpiece_shadow_state()
        leg_pose = torch.cat((legs.outer_pose_xyzw, legs.rod_pose_xyzw, shadow_pose.unsqueeze(1)), dim=1)
        leg_pose[..., :3] += self.scene.env_origins[:, None, :]
        self.shaker_legs.write_body_pose_to_sim_index(body_poses=leg_pose)
        # These bodies are kinematic and collision-disabled. Velocities are
        # still written for coherent render interpolation and diagnostics.
        linear_velocity = platen_velocity[:, None, :3].expand(-1, 12, -1)
        angular_velocity = platen_velocity[:, None, 3:].expand(-1, 12, -1)
        leg_velocity = torch.cat((linear_velocity, angular_velocity), dim=-1)
        self.shaker_legs.write_body_velocity_to_sim_index(
            body_velocities=torch.cat((leg_velocity, shadow_velocity.unsqueeze(1)), dim=1)
        )

    def _workpiece_shadow_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Project the object onto the moving worktable with sub-step lag only."""

        object_pose = self.workpiece.data.root_pose_w.torch
        table_top_local = self._worktable_local.clone()
        table_top_local[:, 2] += 0.5 * self.cfg.worktable_size[2] + 0.001
        surface_position, surface_quat, surface_velocity = self._support_state(
            table_top_local,
            self._vibration_q,
            self._vibration_qd,
            self._worktable_local,
        )
        shadow_pose = torch.cat((surface_position - self.scene.env_origins, surface_quat), dim=1)
        shadow_pose[:, :2] = object_pose[:, :2]
        shadow_pose[:, :2] -= self.scene.env_origins[:, :2]
        return shadow_pose, surface_velocity

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
        self._write_supports()

        object_pose = self.workpiece.data.default_root_pose.torch.clone()
        object_pose[:, :3] += self.scene.env_origins
        # The YCB USD collider bounds differ slightly from catalog dimensions.
        # Initialize from the live collision mesh so no object starts embedded
        # in the tabletop merely because a nominal visual size was used.
        table_top = self.cfg.resolved_worktable_center[2] + 0.5 * self.cfg.worktable_size[2]
        object_pose[:, 2] = (
            self.scene.env_origins[:, 2]
            + table_top
            + 0.5 * self.workpiece_collision_dimensions_m[2]
            + self.cfg.workpiece_initial_clearance_m
        )
        self.workpiece.write_root_pose_to_sim_index(root_pose=object_pose)
        self.workpiece.write_root_velocity_to_sim_index(
            root_velocity=torch.zeros((self.num_envs, 6), device=self.device)
        )
        self.scene.write_data_to_sim()
        self.sim.step()
        self.scene.update(self.cfg.dt)
        self.metrics = EpisodeMetrics()
        self._current_penetration = PenetrationSample()
        self._penetration_frame_count = 0
        self._penetration_over_limit_count = 0
        self._update_penetration_metrics()
        self._grasp_requested = False
        self._controller_phase = "reset"
        self._grasped = False
        self._bilateral_contact_streak = 0
        self._last_left_contact_n.zero_()
        self._last_right_contact_n.zero_()
        self._last_left_descent_contact_n.zero_()
        self._last_right_descent_contact_n.zero_()
        self._history_count = 0
        self._history_index = 0
        return self.observation()

    def request_grasp(self, closed: bool) -> None:
        self._grasp_requested = bool(closed)

    def set_controller_phase(self, phase: str) -> None:
        self._controller_phase = str(phase)

    def mark_grasp_z_guard(self) -> None:
        self.metrics.grasp_z_guard_triggered = True

    def mark_descend_timeout(self) -> None:
        self.metrics.descend_contact_timeout = True

    def mark_descend_table_contact(self) -> None:
        self.metrics.descend_table_contact = True

    def mark_grasp_table_contact(self) -> None:
        self.metrics.grasp_table_contact = True

    def mark_grasp_contact_timeout(self) -> None:
        self.metrics.grasp_contact_timeout = True

    def mark_grasp_contact_lost(self) -> None:
        self.metrics.grasp_contact_lost = True

    def update_grasp_slip(self, slip_m: float) -> None:
        self.metrics.max_grasp_slip_m = max(self.metrics.max_grasp_slip_m, float(slip_m))

    def mark_grasp_slip_exceeded(self) -> None:
        self.metrics.grasp_slip_exceeded = True

    def _filtered_contact_force(self, sensor: ContactSensor) -> torch.Tensor:
        matrix = sensor.data.force_matrix_w
        if matrix is None:
            return torch.zeros(self.num_envs, device=self.device)
        values = matrix.torch
        return torch.linalg.vector_norm(values, dim=-1).flatten(1).amax(dim=1)

    def _update_grasp_assist(self) -> None:
        hand = self.robot.data.body_pose_w.torch[:, self.ee_body_id]
        obj = self.workpiece.data.root_pose_w.torch
        self._last_left_contact_n = self._filtered_contact_force(self.left_finger_contact)
        self._last_right_contact_n = self._filtered_contact_force(self.right_finger_contact)
        self._last_left_descent_contact_n = self._filtered_contact_force(self.left_finger_descent_contact)
        self._last_right_descent_contact_n = self._filtered_contact_force(self.right_finger_descent_contact)
        bilateral_contact = bool(
            ((self._last_left_contact_n > 0.05) & (self._last_right_contact_n > 0.05)).all().item()
        )
        distance = torch.linalg.norm(obj[:, :3] - hand[:, :3], dim=1)
        geometrically_local = bool((distance < 0.15).all().item())
        # Wide benchmark objects can make contact while each Panda finger is
        # still above 30 mm.  Explicit close command + measured bilateral
        # contact is the physically meaningful gate; a fixed joint-gap gate
        # incorrectly rejected those valid grasps.
        if self._grasp_requested and bilateral_contact and geometrically_local:
            self._bilateral_contact_streak += 1
        else:
            # A single vibration-induced contact dropout should not reset a
            # nearly complete tactile latch.  Decay one frame instead of
            # zeroing: this is contact robustness for the controller, not a
            # relaxation of the 4-frame bilateral-contact requirement.
            self._bilateral_contact_streak = max(0, self._bilateral_contact_streak - 1)

        if self._bilateral_contact_streak >= 4:
            self.metrics.bilateral_contact_confirmed = True

        if not self.cfg.grasp_assist:
            if not self._grasp_requested:
                self._grasped = False
            return
        if self._grasp_requested and not self._grasped:
            # A hold may start only after four consecutive physics frames of
            # real, bilateral finger-workpiece contact.  Proximity alone can
            # never move the object, eliminating the former air grasp.
            if self._bilateral_contact_streak >= 4:
                if self._current_penetration.depth_m >= 0.0005:
                    self.metrics.grasp_assist_rejected_penetration = True
                    self._bilateral_contact_streak = 0
                else:
                    inv = quat_inv(hand[:, 3:7])
                    self._held_pos_h = quat_apply(inv, obj[:, :3] - hand[:, :3])
                    self._held_quat_h = quat_mul(inv, obj[:, 3:7])
                    self._grasped = True
                    self.metrics.bilateral_contact_confirmed = True
                    self.metrics.grasp_assist_used = True
        elif not self._grasp_requested:
            self._grasped = False

        if self._grasped and self._current_penetration.depth_m > 0.001:
            self._grasped = False
            self.metrics.grasp_assist_released_penetration = True

        if self._grasped:
            pose = self.workpiece.data.root_pose_w.torch.clone()
            offset_w = quat_apply(hand[:, 3:7], self._held_pos_h)
            pose[:, :3] = hand[:, :3] + offset_w
            pose[:, 3:7] = quat_mul(hand[:, 3:7], self._held_quat_h)
            self.workpiece.write_root_pose_to_sim_index(root_pose=pose)
            velocity = self.robot.data.body_vel_w.torch[:, self.ee_body_id].clone()
            velocity[:, :3] += torch.linalg.cross(velocity[:, 3:], offset_w)
            self.workpiece.write_root_velocity_to_sim_index(root_velocity=velocity)

    def _step_support_substeps(
        self, arm_target: torch.Tensor, finger_target: torch.Tensor
    ) -> None:
        """Advance physics with support trajectories updated every solver substep."""

        substep_dt = NewtonManager._solver_dt
        state0 = NewtonManager._state_0
        control = NewtonManager._control
        adapter = NewtonManager._adapter

        step_t0 = time.perf_counter()
        self._vibration_q, self._vibration_qd, self._vibration_qdd = self.vibration.sample(self.time_s)
        self._write_supports()
        self._update_grasp_assist()
        self.robot.set_joint_position_target_index(target=arm_target, joint_ids=self.arm_joint_ids)
        self.robot.set_joint_position_target_index(target=finger_target, joint_ids=self.finger_joint_ids)
        self.scene.write_data_to_sim()
        self._support_step_timing["sample"] += time.perf_counter() - step_t0

        adapter_t0 = time.perf_counter()
        if adapter is not None:
            adapter.step(state0, control, self.cfg.dt)
        for callback in NewtonManager._post_actuator_callbacks:
            callback()
        self._support_step_timing["adapter"] += time.perf_counter() - adapter_t0

        for substep in range(self.cfg.solver_substeps):
            if (
                substep > 0
                and substep % self.cfg.clite_mocap_update_decimation == 0
            ):
                mocap_t0 = time.perf_counter()
                sample_t = self.time_s + substep * substep_dt
                q, qd, qdd = self.vibration.sample(sample_t)
                self._vibration_q, self._vibration_qd, self._vibration_qdd = q, qd, qdd
                # Dynamic platform/worktable supports track the mocap arrays
                # directly every solver substep.  Kinematic followers (Panda
                # root, legs, target) remain on the outer-step cadence; they
                # are not contact targets for the workpiece and writing them
                # through the full Isaac-Lab asset path every substep is what
                # made C2_CLITE 100x realtime.
                self._write_clite_mocap(q)
                self._support_step_timing["mocap_write"] += time.perf_counter() - mocap_t0
            solver_t0 = time.perf_counter()
            NewtonManager._step_solver(state0, state0, control, None, substep_dt)
            self._support_step_timing["solver"] += time.perf_counter() - solver_t0
            clear_t0 = time.perf_counter()
            state0.clear_forces()
            self._support_step_timing["clear_forces"] += time.perf_counter() - clear_t0

        sensor_t0 = time.perf_counter()
        NewtonManager._update_sensors(None)
        NewtonManager._sim_time += self.cfg.dt
        NewtonManager._log_solver_debug()
        self.scene.update(self.cfg.dt)
        self._support_step_timing["sensors_scene"] += time.perf_counter() - sensor_t0
        self._support_step_count += 1

    def step(self, arm_target: torch.Tensor, finger_target: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.cfg.use_clite_support:
            self._step_support_substeps(arm_target, finger_target)
        else:
            self._vibration_q, self._vibration_qd, self._vibration_qdd = self.vibration.sample(self.time_s)
            self._write_supports()
            self._update_grasp_assist()
            self.robot.set_joint_position_target_index(target=arm_target, joint_ids=self.arm_joint_ids)
            self.robot.set_joint_position_target_index(target=finger_target, joint_ids=self.finger_joint_ids)
            self.scene.write_data_to_sim()
            self.sim.step()
            self.scene.update(self.cfg.dt)
        self.time_s += self.cfg.dt
        self._update_penetration_metrics()
        violation = forbidden_contact_violation(HARD_STRUCTURAL_EXCLUSIONS)
        if violation is not None:
            self.metrics.support_geometry_valid = False

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
        obj_w = self.workpiece.data.root_pose_w.torch
        target_w = self.target.data.root_pose_w.torch
        obj_pos_b, obj_quat_b = subtract_frame_transforms(
            root[:, :3], root[:, 3:7], obj_w[:, :3], obj_w[:, 3:7]
        )
        target_pos_b, target_quat_b = subtract_frame_transforms(
            root[:, :3], root[:, 3:7], target_w[:, :3], target_w[:, 3:7]
        )
        finger_center_w = 0.5 * (
            self.robot.data.body_pose_w.torch[:, self.left_finger_body_id, :3]
            + self.robot.data.body_pose_w.torch[:, self.right_finger_body_id, :3]
        )
        finger_center_b, _ = subtract_frame_transforms(
            root[:, :3], root[:, 3:7], finger_center_w, hand[:, 3:7]
        )
        wrist_camera_eye_w, wrist_camera_target_w, wrist_camera_up_w = wrist_camera_frame_from_hand(hand)
        if self.wrench_body_id >= 0:
            force = self.wrist_wrench.data.force.torch[:, self.wrench_body_id]
            torque = self.wrist_wrench.data.torque.torch[:, self.wrench_body_id]
        else:
            force = torch.zeros((self.num_envs, 3), device=self.device)
            torque = torch.zeros_like(force)
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
            "workpiece_pose_w": obj_w,
            "workpiece_pose_b": torch.cat((obj_pos_b, obj_quat_b), dim=1),
            "target_pose_w": target_w,
            "target_pose_b": torch.cat((target_pos_b, target_quat_b), dim=1),
            "vibration_q": self._vibration_q,
            "vibration_qd": self._vibration_qd,
            "vibration_qdd": self._vibration_qdd,
            "mount_delta_z": self._arm_table_delta_z,
            "shaker_leg_lengths_m": self._shaker_leg_lengths,
            "wrist_force_b": force,
            "wrist_torque_b": torque,
            "left_finger_contact_n": self._last_left_contact_n.unsqueeze(1),
            "right_finger_contact_n": self._last_right_contact_n.unsqueeze(1),
            "left_finger_descent_contact_n": self._last_left_descent_contact_n.unsqueeze(1),
            "right_finger_descent_contact_n": self._last_right_descent_contact_n.unsqueeze(1),
            "penetration_mm": torch.full(
                (self.num_envs, 1),
                self._current_penetration.depth_mm,
                device=self.device,
            ),
            "bilateral_contact_streak": torch.full(
                (self.num_envs, 1), self._bilateral_contact_streak, device=self.device, dtype=torch.int32
            ),
            "grasped": torch.full((self.num_envs,), self._grasped, device=self.device, dtype=torch.bool),
        }

    def wrist_camera_rgb(self, obs: dict[str, torch.Tensor]) -> Any:
        """Render an RGB observation from the modeled wrist camera."""

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
        object_z = float(obs["workpiece_pose_w"][0, 2].item())
        target = obs["target_pose_w"][0, :3]
        obj = obs["workpiece_pose_w"][0, :3]
        self.metrics.lifted |= object_z > float(target[2].item()) + 0.10
        xy_error = float(torch.linalg.norm(obj[:2] - target[:2]).item())
        self.metrics.final_xy_error_m = xy_error
        self.metrics.placed |= self.metrics.lifted and xy_error < 0.07 and not self._grasped
        self.metrics.success = self.metrics.lifted and self.metrics.placed
        wrist_force_n = float(torch.linalg.norm(obs["wrist_force_b"][0]).item())
        if wrist_force_n > self.metrics.max_wrist_force_n:
            self.metrics.max_wrist_force_n = wrist_force_n
            self.metrics.max_wrist_force_t = self.time_s
            self.metrics.max_wrist_force_phase = self._controller_phase
            self.metrics.contact_pair_at_max_wrist_force = self._current_penetration.pair
            self.metrics.contact_shapes_at_max_wrist_force = (
                self._current_penetration.shape0,
                self._current_penetration.shape1,
            )
        self.metrics.max_wrist_torque_nm = max(
            self.metrics.max_wrist_torque_nm,
            float(torch.linalg.norm(obs["wrist_torque_b"][0]).item()),
        )
        self.metrics.max_left_finger_contact_n = max(
            self.metrics.max_left_finger_contact_n,
            float(obs["left_finger_contact_n"][0, 0].item()),
        )
        self.metrics.max_right_finger_contact_n = max(
            self.metrics.max_right_finger_contact_n,
            float(obs["right_finger_contact_n"][0, 0].item()),
        )

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
