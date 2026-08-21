"""Deterministic differential-IK pick-and-place baseline."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.utils.math import quat_apply, quat_from_euler_xyz, quat_mul

from ..config import YCB_ASSETS, workpiece_dimensions_m
from ..envs.manipulation.pick_place import VibrationBenchmarkTask


GRIPPER_USABLE_OPENING_M = 0.075


def rate_limit_translation(
    current: torch.Tensor,
    target: torch.Tensor,
    max_step_m: float,
) -> torch.Tensor:
    """Move a batched Cartesian command toward its target without a pose jump."""

    if max_step_m <= 0.0:
        raise ValueError("max_step_m must be positive")
    delta = target - current
    distance = torch.linalg.vector_norm(delta, dim=1, keepdim=True)
    scale = torch.clamp(max_step_m / torch.clamp(distance, min=1.0e-12), max=1.0)
    return current + scale * delta


def rate_limit_joint_target(
    current: torch.Tensor,
    target: torch.Tensor,
    max_step: float,
) -> torch.Tensor:
    """Rate-limit independent joint targets, including both Panda fingers."""

    if max_step <= 0.0:
        raise ValueError("max_step must be positive")
    return current + torch.clamp(target - current, min=-max_step, max=max_step)


def latch_finger_contact_targets(
    current: torch.Tensor,
    commanded: torch.Tensor,
    desired: torch.Tensor,
    contact_mask: torch.Tensor,
    latched_mask: torch.Tensor,
    hold_targets: torch.Tensor | None,
    preload_m: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Freeze each finger independently when it first contacts the object."""

    if current.shape != commanded.shape or current.shape != desired.shape:
        raise ValueError("finger position tensors must have identical shapes")
    if contact_mask.shape != current.shape or latched_mask.shape != current.shape:
        raise ValueError("finger contact masks must match the position tensors")
    if preload_m < 0.0:
        raise ValueError("finger preload must be non-negative")
    newly_latched = contact_mask & ~latched_mask
    updated_latched = latched_mask | contact_mask
    if not bool(updated_latched.any().item()):
        return desired, updated_latched, hold_targets
    if hold_targets is None:
        hold_targets = desired.clone()
    reference = torch.minimum(current, commanded)
    preload = torch.clamp(reference - preload_m, min=0.0)
    hold_targets = torch.where(newly_latched, preload, hold_targets)
    return torch.where(updated_latched, hold_targets, desired), updated_latched, hold_targets


def collision_safe_descend_clearance(
    nominal_clearance_m: float,
    finger_downward_reach_m: float,
    object_height_m: torch.Tensor,
    table_clearance_m: float,
) -> torch.Tensor:
    """Clearance above object top that keeps the finger tip above its support."""

    required = finger_downward_reach_m + table_clearance_m - object_height_m
    return torch.clamp(required, min=nominal_clearance_m)


def grasp_feasibility(workpiece: str, scale: float) -> tuple[bool, float]:
    """Return top-down feasibility and the smaller horizontal extent."""

    dimensions = workpiece_dimensions_m(workpiece, scale)
    minimum = min(dimensions[:2])
    return minimum <= GRIPPER_USABLE_OPENING_M, minimum


def grasp_feasibility_table(scales: tuple[float, ...] = (0.55, 0.75, 1.0)) -> list[dict[str, object]]:
    """Machine/console-friendly table for all benchmark YCB assets."""

    rows: list[dict[str, object]] = []
    for name in YCB_ASSETS:
        for scale in scales:
            feasible, minimum = grasp_feasibility(name, scale)
            rows.append(
                {
                    "workpiece": name,
                    "scale": scale,
                    "min_horizontal_m": minimum,
                    "feasible": feasible,
                }
            )
    return rows


def projected_half_height(quaternion_wxyz: torch.Tensor, dimensions_m: tuple[float, float, float]) -> torch.Tensor:
    """World/base Z half-extent of an oriented axis-aligned workpiece."""

    basis = torch.eye(3, device=quaternion_wxyz.device, dtype=quaternion_wxyz.dtype)
    axes = []
    for index in range(3):
        axes.append(quat_apply(quaternion_wxyz, basis[index].repeat(len(quaternion_wxyz), 1)))
    rotation = torch.stack(axes, dim=2)
    dimensions = torch.tensor(dimensions_m, device=quaternion_wxyz.device, dtype=quaternion_wxyz.dtype)
    return 0.5 * torch.sum(torch.abs(rotation[:, 2, :]) * dimensions, dim=1)


def short_axis_yaw(quaternion_wxyz: torch.Tensor, dimensions_m: tuple[float, float, float]) -> torch.Tensor:
    """Yaw of the workpiece's shorter horizontal principal axis."""

    short_index = 0 if dimensions_m[0] <= dimensions_m[1] else 1
    basis = torch.zeros((len(quaternion_wxyz), 3), device=quaternion_wxyz.device, dtype=quaternion_wxyz.dtype)
    basis[:, short_index] = 1.0
    axis_b = quat_apply(quaternion_wxyz, basis)
    return torch.atan2(axis_b[:, 1], axis_b[:, 0])


@dataclass(frozen=True)
class Phase:
    name: str
    duration_s: float
    position_b: tuple[float, float, float]
    finger_m: float
    grasp: bool


class ScriptedPickPlaceController:
    """Reference policy, not part of the benchmark score definition."""

    def __init__(self, task: VibrationBenchmarkTask):
        self.task = task
        dimensions = getattr(
            task,
            "workpiece_collision_dimensions_m",
            workpiece_dimensions_m(task.cfg.assets.workpiece, task.cfg.assets.workpiece_scale),
        )
        minimum = min(dimensions[:2])
        feasible = minimum <= GRIPPER_USABLE_OPENING_M
        if not feasible:
            raise ValueError(
                f"{task.cfg.assets.workpiece} at scale={task.cfg.assets.workpiece_scale:.3f} "
                f"has minimum horizontal extent {minimum:.4f} m > "
                f"{GRIPPER_USABLE_OPENING_M:.3f} m usable Panda opening"
            )
        self.workpiece_dimensions_m = dimensions
        self.ik = DifferentialIKController(
            DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
            num_envs=task.num_envs,
            device=task.device,
        )
        # Coordinates are relative to the vibrating robot base.
        self.phases = (
            Phase("settle", 1.2, (0.55, -0.13, 0.66), 0.04, False),
            Phase("approach", 1.8, (0.55, -0.13, 0.55), 0.04, False),
            Phase("descend", 1.5, (0.55, -0.13, 0.42), 0.04, False),
            # The 0.012 m target gives an approximately 24 mm nominal gap.
            # Closure stops at measured bilateral contact, so the fingers do
            # not continue toward a deeply penetrating target.
            Phase("grasp", task.cfg.grasp_timeout_s, (0.55, -0.13, 0.42), 0.012, True),
            Phase("lift", 1.8, (0.55, -0.13, 0.66), 0.012, True),
            Phase("transfer", 2.2, (0.55, 0.17, 0.66), 0.012, True),
            Phase("place", 1.8, (0.55, 0.17, 0.42), 0.012, True),
            Phase("release", 1.0, (0.55, 0.17, 0.42), 0.04, False),
            Phase("retreat", 1.5, (0.55, 0.17, 0.61), 0.04, False),
        )
        self.phase_index = 0
        self.phase_time = 0.0
        self.orientation_b: torch.Tensor | None = None
        self.settle_pose_b: torch.Tensor | None = None
        self.settle_joint_position: torch.Tensor | None = None
        self.grasp_xy_b: torch.Tensor | None = None
        self.grasp_hand_position_b: torch.Tensor | None = None
        self.hand_minus_object_b: torch.Tensor | None = None
        self.contact_finger_target: torch.Tensor | None = None
        self.finger_contact_latched = torch.zeros(
            (task.num_envs, len(task.finger_joint_ids)),
            device=task.device,
            dtype=torch.bool,
        )
        self.commanded_position_b: torch.Tensor | None = None
        self.commanded_finger_position: torch.Tensor | None = None
        self.failure_reason: str | None = None
        self.descend_target_reached = False
        self.contact_loss_time_s = 0.0
        self.grasp_ready_time_s = 0.0
        self.ik.reset()

    @property
    def name(self) -> str:
        return self.phases[min(self.phase_index, len(self.phases) - 1)].name

    @property
    def finished(self) -> bool:
        return self.phase_index >= len(self.phases)

    def command(self, obs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        phase = self.phases[min(self.phase_index, len(self.phases) - 1)]
        if phase.name in ("lift", "transfer", "place") and self.contact_finger_target is not None:
            left_now = obs["left_finger_contact_n"][:, 0] > self.task.cfg.descend_contact_threshold_n
            right_now = obs["right_finger_contact_n"][:, 0] > self.task.cfg.descend_contact_threshold_n
            # A one-frame loss on either sensor is common during vibration and
            # is not itself proof that the object was dropped.  Declare contact
            # loss only when both fingers are clear for a sustained interval.
            if bool((left_now | right_now).all().item()):
                self.contact_loss_time_s = 0.0
            else:
                self.contact_loss_time_s += self.task.cfg.dt
            slip_exceeded = False
            if self.hand_minus_object_b is not None:
                hand_minus_object_b = obs["ee_pose_b"][:, :3] - obs["workpiece_pose_b"][:, :3]
                slip = torch.linalg.vector_norm(hand_minus_object_b - self.hand_minus_object_b, dim=1)
                self.task.update_grasp_slip(float(slip.max().item()))
                slip_exceeded = bool((slip > self.task.cfg.grasp_slip_tolerance_m).any().item())
            if slip_exceeded:
                self.task.mark_grasp_slip_exceeded()
                self.failure_reason = "grasp_slip_exceeded"
                self.phase_index = len(self.phases)
            elif self.contact_loss_time_s >= self.task.cfg.grasp_contact_loss_timeout_s:
                self.task.mark_grasp_contact_lost()
                self.failure_reason = "grasp_contact_lost"
                self.phase_index = len(self.phases)
        if phase.name in ("descend", "grasp"):
            workpiece_contact = (
                (obs["left_finger_contact_n"][:, 0] > self.task.cfg.descend_contact_threshold_n)
                | (obs["right_finger_contact_n"][:, 0] > self.task.cfg.descend_contact_threshold_n)
            )
            any_descent_contact = (
                (obs["left_finger_descent_contact_n"][:, 0] > self.task.cfg.descend_contact_threshold_n)
                | (obs["right_finger_descent_contact_n"][:, 0] > self.task.cfg.descend_contact_threshold_n)
            )
            table_only_contact = any_descent_contact & ~workpiece_contact
            if bool(table_only_contact.any().item()):
                if phase.name == "descend":
                    self.task.mark_descend_table_contact()
                    self.failure_reason = "descend_table_contact"
                else:
                    self.task.mark_grasp_table_contact()
                    self.failure_reason = "grasp_table_contact"
                self.phase_index = len(self.phases)
            elif phase.name == "descend" and bool(workpiece_contact.any().item()):
                self.phase_index += 1
                self.phase_time = 0.0
                phase = self.phases[self.phase_index]

        object_quat_b = obs["workpiece_pose_b"][:, 3:7]
        yaw = short_axis_yaw(object_quat_b, self.workpiece_dimensions_m) - 0.5 * math.pi
        yaw_quat = quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)
        top_down = torch.tensor((1.0, 0.0, 0.0, 0.0), device=self.task.device).repeat(
            self.task.num_envs, 1
        )
        self.orientation_b = quat_mul(yaw_quat, top_down)
        position = torch.tensor(phase.position_b, device=self.task.device).repeat(self.task.num_envs, 1)
        if phase.name == "settle":
            # "Settle" must genuinely hold the reset pose.  Driving directly
            # toward the first task waypoint swept the hand through the table
            # before approach began and dominated the penetration metric.
            if self.settle_pose_b is None:
                self.settle_pose_b = obs["ee_pose_b"].clone()
                self.settle_joint_position = self.task.arm_position().clone()
            position = self.settle_pose_b[:, :3]
            self.orientation_b = self.settle_pose_b[:, 3:7]
        elif phase.name in ("approach", "descend", "grasp"):
            # C2 makes the table move relative to the arm, and a free YCB
            # object can additionally slide.  Track the measured object pose
            # rather than aiming at its initial coordinates.
            object_b = obs["workpiece_pose_b"][:, :3]
            hand_to_finger_center_b = obs["finger_center_b"] - obs["ee_pose_b"][:, :3]
            object_top_z_b = object_b[:, 2] + projected_half_height(
                object_quat_b,
                self.workpiece_dimensions_m,
            )
            object_height_b = 2.0 * projected_half_height(
                object_quat_b,
                self.workpiece_dimensions_m,
            )
            desired_finger_center_b = object_b.clone()
            if phase.name == "approach":
                clearance_b = torch.full_like(object_top_z_b, self.task.cfg.approach_clearance_m)
            else:
                clearance_b = collision_safe_descend_clearance(
                    self.task.cfg.descend_clearance_m,
                    self.task.finger_downward_reach_m,
                    object_height_b,
                    self.task.cfg.finger_table_clearance_m,
                )
            desired_finger_center_b[:, 2] = object_top_z_b + clearance_b
            position = desired_finger_center_b - hand_to_finger_center_b
            self.grasp_xy_b = position[:, :2].clone()
            if phase.name in ("descend", "grasp"):
                finger_z = torch.minimum(
                    obs["left_finger_pose_w"][:, 2],
                    obs["right_finger_pose_w"][:, 2],
                )
                guard_z = obs["workpiece_pose_w"][:, 2] + self.task.cfg.grasp_z_guard_margin_m
                if bool((finger_z < guard_z).any().item()):
                    self.task.mark_grasp_z_guard()
                    self.failure_reason = "grasp_z_guard_triggered"
                    self.phase_index = len(self.phases)
            if phase.name == "grasp" and bool(obs["grasped"].all().item()):
                if self.grasp_hand_position_b is None:
                    self.grasp_hand_position_b = obs["ee_pose_b"][:, :3].clone()
                    self.hand_minus_object_b = (
                        obs["ee_pose_b"][:, :3] - obs["workpiece_pose_b"][:, :3]
                    ).clone()
                position = self.grasp_hand_position_b
        elif phase.name == "lift" and self.grasp_xy_b is not None:
            if self.grasp_hand_position_b is not None:
                position[:, :2] = self.grasp_hand_position_b[:, :2]
            else:
                position[:, :2] = self.grasp_xy_b
        elif phase.name in ("transfer", "place", "release", "retreat"):
            object_to_hand = self.hand_minus_object_b
            if object_to_hand is None:
                object_to_hand = torch.zeros_like(obs["target_pose_b"][:, :3])
            position[:, :2] = obs["target_pose_b"][:, :2] + object_to_hand[:, :2]
            if phase.name == "place" or phase.name == "release":
                position[:, 2] = obs["target_pose_b"][:, 2] + object_to_hand[:, 2]
        desired_position_b = position.clone()
        if phase.name == "settle":
            self.commanded_position_b = position.clone()
        else:
            if self.commanded_position_b is None:
                self.commanded_position_b = obs["ee_pose_b"][:, :3].clone()
            if phase.name in ("descend", "grasp"):
                # Horizontal tracking follows the sliding/rebounding object
                # at the normal arm speed; only the vertical channel stays
                # deliberately slow to keep descent contact gentle.
                xy = rate_limit_translation(
                    self.commanded_position_b[:, :2],
                    position[:, :2],
                    self.task.cfg.arm_linear_speed_m_s * self.task.cfg.dt,
                )
                z = rate_limit_translation(
                    self.commanded_position_b[:, 2:3],
                    position[:, 2:3],
                    self.task.cfg.descend_linear_speed_m_s * self.task.cfg.dt,
                )
                self.commanded_position_b = torch.cat((xy, z), dim=1)
            else:
                if phase.name == "lift" and self.phase_time < self.task.cfg.lift_takeoff_duration_s:
                    linear_speed = self.task.cfg.lift_takeoff_speed_m_s
                elif phase.name in ("place", "release"):
                    linear_speed = self.task.cfg.place_linear_speed_m_s
                else:
                    linear_speed = self.task.cfg.arm_linear_speed_m_s
                self.commanded_position_b = rate_limit_translation(
                    self.commanded_position_b,
                    position,
                    linear_speed * self.task.cfg.dt,
                )
            position = self.commanded_position_b
        self.descend_target_reached = bool(
            phase.name == "descend"
            and (
                torch.linalg.vector_norm(desired_position_b - position, dim=1)
                <= self.task.cfg.descend_position_tolerance_m
            ).all().item()
        )
        command = torch.cat((position, self.orientation_b), dim=1)
        self.ik.set_command(command)
        if phase.name == "settle":
            # A floating C2-driven base makes Cartesian hold inject needless
            # joint corrections.  Preserve the reset joint posture exactly.
            arm = self.settle_joint_position
        else:
            arm = self.ik.compute(
                obs["ee_pose_b"][:, :3],
                obs["ee_pose_b"][:, 3:7],
                self.task.arm_jacobian(),
                self.task.arm_position(),
            )
        desired_fingers = torch.full(
            (self.task.num_envs, len(self.task.finger_joint_ids)),
            phase.finger_m,
            device=self.task.device,
        )
        if phase.grasp:
            contact_mask = torch.stack(
                (
                    obs["left_finger_contact_n"][:, 0] > self.task.cfg.descend_contact_threshold_n,
                    obs["right_finger_contact_n"][:, 0] > self.task.cfg.descend_contact_threshold_n,
                ),
                dim=1,
            )
            current = obs["joint_pos"][:, self.task.finger_joint_ids]
            commanded = self.commanded_finger_position
            if commanded is None:
                commanded = current
            desired_fingers, self.finger_contact_latched, self.contact_finger_target = (
                latch_finger_contact_targets(
                    current,
                    commanded,
                    desired_fingers,
                    contact_mask,
                    self.finger_contact_latched,
                    self.contact_finger_target,
                    self.task.cfg.gripper_contact_preload_m,
                )
            )
            if self.contact_finger_target is not None:
                # If centering/rebound opens one side after its first touch,
                # ratchet only that finger inward until measured contact
                # returns.  This is a physical tactile servo, not an object
                # attachment or pose override.
                recover = self.finger_contact_latched & ~contact_mask
                recovered_target = torch.clamp(
                    self.contact_finger_target
                    - self.task.cfg.gripper_contact_recovery_speed_m_s * self.task.cfg.dt,
                    min=phase.finger_m,
                )
                self.contact_finger_target = torch.where(
                    recover,
                    recovered_target,
                    self.contact_finger_target,
                )
                desired_fingers = torch.where(
                    self.finger_contact_latched,
                    self.contact_finger_target,
                    desired_fingers,
                )
            if (
                phase.name == "grasp"
                and bool(self.finger_contact_latched.all().item())
                and self.task.metrics.bilateral_contact_confirmed
            ):
                self.grasp_ready_time_s += self.task.cfg.dt
            elif phase.name == "grasp":
                self.grasp_ready_time_s = 0.0
        else:
            self.contact_finger_target = None
            self.finger_contact_latched.zero_()
        if self.commanded_finger_position is None:
            self.commanded_finger_position = obs["joint_pos"][:, self.task.finger_joint_ids].clone()
        finger_speed = (
            self.task.cfg.gripper_closing_speed_m_s
            if phase.grasp
            else self.task.cfg.gripper_opening_speed_m_s
        )
        self.commanded_finger_position = rate_limit_joint_target(
            self.commanded_finger_position,
            desired_fingers,
            finger_speed * self.task.cfg.dt,
        )
        fingers = self.commanded_finger_position
        self.task.set_controller_phase(phase.name)
        self.task.request_grasp(phase.grasp)
        self.phase_time += self.task.cfg.dt
        if phase.name == "descend":
            if self.descend_target_reached:
                self.phase_time = 0.0
                self.phase_index += 1
            elif self.phase_time >= self.task.cfg.descend_timeout_s:
                self.task.mark_descend_timeout()
                self.failure_reason = "descend_contact_timeout"
                self.phase_index = len(self.phases)
        elif phase.name == "grasp":
            grasp_ready = self.grasp_ready_time_s >= self.task.cfg.grasp_settle_s
            if grasp_ready:
                # The timeout is a failure deadline, not a mandatory dwell.
                # Once real bilateral contact has been latched and allowed to
                # settle, start transport while the object is still centered
                # instead of squeezing it against the moving table for the
                # entire timeout window.
                self.grasp_hand_position_b = obs["ee_pose_b"][:, :3].clone()
                self.hand_minus_object_b = (
                    obs["ee_pose_b"][:, :3] - obs["workpiece_pose_b"][:, :3]
                ).clone()
                self.phase_time = 0.0
                self.phase_index += 1
            elif self.phase_time >= phase.duration_s:
                self.task.mark_grasp_contact_timeout()
                self.failure_reason = "grasp_contact_timeout"
                self.phase_index = len(self.phases)
        elif self.phase_time >= phase.duration_s:
            self.phase_time = 0.0
            self.phase_index += 1
        return arm, fingers
