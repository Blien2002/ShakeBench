"""Deterministic differential-IK baseline for the panel-operation task.

The reference policy is intentionally simple and transparent: it moves the
fingertip center to the live (table-relative) interaction point of the next
instructed control, performs a contact-gated operation, and retreats between
controls.  Control states are advanced by :class:`PanelBenchmarkTask` only
while the corresponding finger contact is measured, and the required order is
enforced by the task.
"""

from __future__ import annotations

import math

import torch

from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.utils.math import quat_apply, quat_inv

from ..config import CONTROL_KINDS
from .scripted import latch_finger_contact_targets, rate_limit_joint_target, rate_limit_translation
from ..models.objects.panel import CONTROL_INDEX
from ..envs.manipulation.panel_operation import PanelBenchmarkTask


OPEN_FINGER_M = 0.040
KNOB_CLOSE_FINGER_M = 0.010
MOVE_TOLERANCE_M = 0.012
RETREAT_POSE_B = (0.30, 0.0, 0.62)
SETTLE_TIME_S = 0.60
PANEL_JOINT_SPEED_RAD_S = 1.20
REACH_STANDOFF_M = 0.025
KNOB_SWEEP_M = 0.060
LEVER_SWEEP_M = 0.055
LEVER_HAND_RADIUS_M = 0.110


class ScriptedPanelController:
    """Reference policy, not part of the benchmark score definition."""

    def __init__(self, task: PanelBenchmarkTask):
        self.task = task
        self.sequence = task.panel_sequence
        self.ik = DifferentialIKController(
            DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
            num_envs=task.num_envs,
            device=task.device,
        )
        self.mode = "settle"
        self.control_index = 0
        self.mode_time = 0.0
        self.hold_time = 0.0
        self.settle_pose_b: torch.Tensor | None = None
        self.settle_joint_position: torch.Tensor | None = None
        self.hand_to_finger_center_b: torch.Tensor | None = None
        self.commanded_position_b: torch.Tensor | None = None
        self.commanded_arm_position: torch.Tensor | None = None
        self.commanded_finger_position: torch.Tensor | None = None
        self.contact_finger_target: torch.Tensor | None = None
        self.finger_contact_latched = torch.zeros(
            (task.num_envs, len(task.finger_joint_ids)),
            device=task.device,
            dtype=torch.bool,
        )
        self.operation_contact_ok = False
        self.knob_bilateral_streak = 0
        self.contact_loss_time_s = 0.0
        self.failure_reason: str | None = None
        self._finished = False
        self.ik.reset()

    @property
    def name(self) -> str:
        if self.control_index < len(self.sequence):
            return f"{self.mode}_{self.sequence[self.control_index]}"
        return self.mode

    @property
    def finished(self) -> bool:
        return self._finished

    def _active_kind(self) -> str | None:
        if self.control_index < len(self.sequence):
            return self.sequence[self.control_index]
        return None

    def _fail(self, reason: str) -> None:
        self.failure_reason = reason
        self._finished = True

    def _interaction_point_b(self, obs: dict[str, torch.Tensor], kind: str, progress: float) -> torch.Tensor:
        normal_b = obs["panel_surface_normal_b"]
        tangent_b = obs["panel_surface_tangent_b"]
        lateral_b = obs["panel_surface_lateral_b"]
        outward = REACH_STANDOFF_M * normal_b
        if kind == "knob":
            # One open finger sweeps the pointer nose laterally.  Offset the
            # gripper center so the +Y finger, rather than the palm, meets the
            # convex knob proxy.
            nose = obs["knob_pose_b"][:, :3] + 0.025 * tangent_b + 0.034 * normal_b
            return nose + outward - OPEN_FINGER_M * lateral_b
        if kind == "lever":
            # The gripper body centre stays beyond the tip so the palm clears
            # the panel; the long finger pads still close around the grip.
            pivot = obs["lever_pivot_b"][:, :3]
            return pivot + LEVER_HAND_RADIUS_M * normal_b
        if kind == "button":
            face = obs["button_face_b"][:, :3]
            return face + outward - OPEN_FINGER_M * lateral_b
        raise ValueError(f"unknown panel control: {kind}")

    def _orientation_b(self, obs: dict[str, torch.Tensor], kind: str) -> torch.Tensor:
        # Keep a stable top-down grasp orientation for every control.  The
        # knob's visual rotation is authored by the task state; forcing the
        # wrist to roll with it makes the DLS solution jump between elbow
        # branches and sweep the forearm through the tabletop.
        return obs["panel_pose_b"][:, 3:7]

    def _desired_finger_center_b(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        kind = self._active_kind()
        if self.mode == "settle" or kind is None:
            if self.settle_pose_b is None:
                self.settle_pose_b = obs["ee_pose_b"].clone()
                self.settle_joint_position = self.task.arm_position().clone()
            return obs["finger_center_b"]
        if self.mode == "retreat":
            # Retreat target is a hand pose, not a fingertip target.
            return None
        if self.mode == "pre":
            index = CONTROL_INDEX[kind]
            progress = float(self.task._control_state[0, index].item())
            point = self._interaction_point_b(obs, kind, progress)
            panel_quat_b = obs["panel_pose_b"][:, 3:7]
            offset = quat_apply(panel_quat_b, torch.tensor(
                (-0.22, 0.0, 0.16), device=self.task.device
            ).repeat(self.task.num_envs, 1))
            return point + offset
        if self.mode == "approach":
            index = CONTROL_INDEX[kind]
            progress = float(self.task._control_state[0, index].item())
            point = self._interaction_point_b(obs, kind, progress)
            panel_quat_b = obs["panel_pose_b"][:, 3:7]
            offset = quat_apply(panel_quat_b, torch.tensor(
                (-0.06, 0.0, 0.16), device=self.task.device
            ).repeat(self.task.num_envs, 1))
            return point + offset
        if self.mode in ("move", "operate", "hold"):
            index = CONTROL_INDEX[kind]
            progress = float(self.task._control_state[0, index].item())
            point = self._interaction_point_b(obs, kind, progress)
            if self.mode == "operate":
                duration_s = 3.0 if kind == "lever" else 1.25
                phase = min(1.0, self.mode_time / duration_s)
                normal_b = obs["panel_surface_normal_b"]
                tangent_b = obs["panel_surface_tangent_b"]
                lateral_b = obs["panel_surface_lateral_b"]
                if kind == "knob":
                    point = point + phase * KNOB_SWEEP_M * lateral_b - 0.010 * normal_b
                elif kind == "lever":
                    # Follow the shaft on its revolute arc instead of chasing
                    # the current tip with a Cartesian line.  This keeps the
                    # closed fingers centred on the grip throughout the 30°
                    # motion and generates torque about the real joint.
                    angle = phase * float(self.task.cfg.panel.lever_goal_rad)
                    pivot = obs["lever_pivot_b"][:, :3]
                    point = pivot + LEVER_HAND_RADIUS_M * (
                        math.cos(angle) * normal_b + math.sin(angle) * tangent_b
                    )
                elif kind == "button":
                    # Follow the live button face instead of driving a fixed
                    # 40 mm Cartesian waypoint past it.  The prismatic joint
                    # only has 4.6 mm of travel; a waypoint 15 mm beyond the
                    # face keeps loading the stopped link and produces deep
                    # finger<->button penetration and kN-class contact peaks.
                    face = obs["button_face_b"][:, :3]
                    press_depth = phase * (
                        float(self.task.cfg.panel.button_travel_m) + 0.0008
                    )
                    # ``normal_b`` points outward; pressing travels against it.
                    point = face - press_depth * normal_b - OPEN_FINGER_M * lateral_b
            return point
        return obs["finger_center_b"]

    def command(self, obs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        kind = self._active_kind()
        panel_quat_b = obs["panel_pose_b"][:, 3:7]
        desired_finger_center_b = self._desired_finger_center_b(obs)
        if self.hand_to_finger_center_b is None:
            self.hand_to_finger_center_b = (
                obs["finger_center_b"] - obs["ee_pose_b"][:, :3]
            ).clone()
        hand_to_finger_center_b = self.hand_to_finger_center_b

        if self.mode == "settle":
            if self.settle_pose_b is None:
                self.settle_pose_b = obs["ee_pose_b"].clone()
                self.settle_joint_position = self.task.arm_position().clone()
            position = self.settle_pose_b[:, :3]
            orientation_b = self.settle_pose_b[:, 3:7]
        elif self.mode == "retreat":
            position = torch.tensor(
                RETREAT_POSE_B, device=self.task.device, dtype=torch.float32
            ).repeat(self.task.num_envs, 1)
            orientation_b = (
                self.settle_pose_b[:, 3:7]
                if self.settle_pose_b is not None
                else panel_quat_b
            )
        elif kind is not None and desired_finger_center_b is not None:
            position = desired_finger_center_b - hand_to_finger_center_b
            # Hold the settle hand orientation throughout the panel task.  The
            # settle posture keeps the fingertips below the hand, and changing
            # toward the panel-front frame made DLS swap elbow branches.
            orientation_b = (
                self.settle_pose_b[:, 3:7]
                if self.settle_pose_b is not None
                else panel_quat_b
            )
        else:
            position = obs["ee_pose_b"][:, :3]
            orientation_b = panel_quat_b

        desired_position_b = position.clone()
        if self.mode == "settle":
            self.commanded_position_b = position.clone()
        else:
            if self.commanded_position_b is None:
                self.commanded_position_b = obs["ee_pose_b"][:, :3].clone()
            speed = self.task.cfg.arm_linear_speed_m_s
            if self.mode in ("move", "operate"):
                speed = min(speed, 0.05)
            self.commanded_position_b = rate_limit_translation(
                self.commanded_position_b,
                position,
                speed * self.task.cfg.dt,
            )
            position = self.commanded_position_b

        position_error = torch.linalg.vector_norm(desired_position_b - position, dim=1)
        command = torch.cat((position, orientation_b), dim=1)
        self.ik.set_command(command)
        if self.mode == "settle" and self.settle_joint_position is not None:
            arm = self.settle_joint_position
        else:
            arm = self.ik.compute(
                obs["ee_pose_b"][:, :3],
                obs["ee_pose_b"][:, 3:7],
                self.task.arm_jacobian(),
                self.task.arm_position(),
            )
        # Differential IK can propose unwrapped joint angles far outside the
        # Panda limits for panel-front poses.  Clamp to joint limits and then
        # rate-limit in joint space so a pose-mode IK branch switch can never
        # command a table-sweeping joint jump in one physics step.
        arm_limits = self.task.robot.data.joint_pos_limits[:, self.task.arm_joint_ids, :]
        arm = torch.clamp(arm, min=arm_limits[:, :, 0], max=arm_limits[:, :, 1])
        if self.commanded_arm_position is None:
            self.commanded_arm_position = self.task.arm_position().clone()
        self.commanded_arm_position = rate_limit_joint_target(
            self.commanded_arm_position,
            arm,
            PANEL_JOINT_SPEED_RAD_S * self.task.cfg.dt,
        )
        arm = self.commanded_arm_position

        # Knob and button are operated with one open finger.  The toggle is
        # narrow enough that the gripper closes around its rounded grip before
        # the Cartesian sweep, producing a genuine contact torque.
        desired_fingers = torch.full(
            (self.task.num_envs, len(self.task.finger_joint_ids)),
            OPEN_FINGER_M,
            device=self.task.device,
        )
        if kind == "lever" and self.mode in ("move", "operate", "hold"):
            # The lever proxy is 22 mm across.  A 20 mm inner opening gives
            # both pads a real, shallow grasp instead of leaving the 4 mm air
            # gap produced by the former 26 mm target.
            desired_fingers.fill_(KNOB_CLOSE_FINGER_M)
        if self.commanded_finger_position is None:
            self.commanded_finger_position = obs["joint_pos"][:, self.task.finger_joint_ids].clone()
        self.commanded_finger_position = rate_limit_joint_target(
            self.commanded_finger_position,
            desired_fingers,
            self.task.cfg.gripper_opening_speed_m_s * self.task.cfg.dt,
        )
        fingers = self.commanded_finger_position

        self._advance_modes(obs, desired_position_b, position, position_error)

        if self.mode == "settle" and self.mode_time >= SETTLE_TIME_S:
            self.mode_time = 0.0
            self.mode = "pre"
            self.commanded_position_b = None
            if self.control_index < len(self.sequence):
                self.task.set_active_control(self.sequence[self.control_index])
        return arm, fingers

    def _advance_modes(
        self,
        obs: dict[str, torch.Tensor],
        desired_position_b: torch.Tensor,
        position: torch.Tensor,
        position_error: torch.Tensor,
    ) -> None:
        self.mode_time += self.task.cfg.dt
        kind = self._active_kind()
        threshold = self.task.cfg.panel.contact_threshold_n
        # Phase transitions use the *measured* hand pose.  The rate-limited
        # Cartesian command can reach the target while the real arm still
        # lags; switching phases on commanded pose made the fingers stop short
        # of the controls.
        actual_position_error = torch.linalg.vector_norm(
            desired_position_b - obs["ee_pose_b"][:, :3], dim=1
        )
        reached = bool((actual_position_error <= MOVE_TOLERANCE_M).all().item())

        if self.mode == "pre" and kind is not None:
            if reached:
                self.mode = "approach"
                self.mode_time = 0.0
                self.commanded_position_b = None
            elif self.mode_time >= self.task.cfg.panel.move_timeout_s:
                self.task.mark_move_timeout()
                self._fail(f"move_timeout_{kind}")
            return

        if self.mode == "approach" and kind is not None:
            if reached:
                self.mode = "move"
                self.mode_time = 0.0
                self.commanded_position_b = None
            elif self.mode_time >= self.task.cfg.panel.move_timeout_s:
                self.task.mark_move_timeout()
                self._fail(f"move_timeout_{kind}")
            return

        if self.mode == "move" and kind is not None:
            progress = float(self.task._control_state[0, CONTROL_INDEX[kind]].item())
            left_n, right_n = self.task.contact_force_n(kind)
            made_contact = bool(
                ((left_n >= threshold) | (right_n >= threshold)).all().item()
            )
            if progress >= 0.95:
                # Contact may begin before the gripper-center waypoint is
                # reached.  Joint completion is the authoritative condition;
                # retreat immediately instead of continuing to load the stop.
                self.task.mark_control_complete(kind)
                self.mode = "retreat"
                self.mode_time = 0.0
                self.commanded_position_b = None
            elif reached or (kind == "button" and made_contact):
                self.mode = "operate"
                self.mode_time = 0.0
                self.hold_time = 0.0
                self.operation_contact_ok = False
                self.contact_loss_time_s = 0.0
                self.knob_bilateral_streak = 0
            elif self.mode_time >= self.task.cfg.panel.move_timeout_s:
                self.task.mark_move_timeout()
                self._fail(f"move_timeout_{kind}")
            return

        if self.mode == "operate" and kind is not None:
            index = CONTROL_INDEX[kind]
            progress = float(self.task._control_state[0, index].item())
            left_n, right_n = self.task.contact_force_n(kind)
            contact = bool(
                ((left_n >= threshold) | (right_n >= threshold)).all().item()
            )
            if contact:
                self.operation_contact_ok = True
                self.contact_loss_time_s = 0.0
            elif self.operation_contact_ok:
                self.contact_loss_time_s += self.task.cfg.dt
                if self.contact_loss_time_s >= self.task.cfg.panel.contact_loss_timeout_s:
                    self.task.mark_contact_lost()
                    self._fail(f"contact_lost_{kind}")
                    return

            if progress >= 0.95:
                self.hold_time += self.task.cfg.dt
                if self.hold_time >= self.task.cfg.panel.operation_hold_s:
                    self.task.mark_control_complete(kind)
                    self.mode = "retreat"
                    self.mode_time = 0.0
                    self.commanded_position_b = None
            elif self.mode_time >= self.task.cfg.panel.operation_timeout_s:
                self.task.mark_operation_timeout()
                self._fail(f"operation_timeout_{kind}")
            return

        if self.mode == "retreat":
            if reached:
                if self.control_index + 1 >= len(self.sequence):
                    self.control_index += 1
                    self._finished = True
                else:
                    self.control_index += 1
                    self.mode = "pre"
                    self.mode_time = 0.0
                    self.commanded_position_b = None
                    self.task.set_active_control(self.sequence[self.control_index])
            elif self.mode_time >= self.task.cfg.panel.move_timeout_s:
                self.task.mark_move_timeout()
                self._fail("retreat_timeout")
            return
