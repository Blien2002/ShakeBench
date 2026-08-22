"""Reactive and frozen-replay policies for the disposable robosuite spike."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.spatial.transform import Rotation

from metrics import (
    CONTACT_THRESHOLD_N,
    SLIP_TOLERANCE_M,
    ContactSnapshot,
    eef_position_b,
    object_position_b,
)


@dataclass(frozen=True)
class ReactiveConfig:
    control_freq_hz: int = 20
    arm_speed_m_s: float = 0.15
    descend_speed_m_s: float = 0.06
    gripper_closing_speed_m_s: float = 0.010
    gripper_opening_speed_m_s: float = 0.040
    gripper_contact_preload_m: float = 0.0009
    grip_mode: Literal["position_latch", "force_limited_close"] = "position_latch"
    descend_timeout_s: float = 2.0
    grasp_timeout_s: float = 6.0
    grasp_settle_s: float = 0.30
    contact_loss_timeout_s: float = 0.20
    transfer_hold_s: float = 4.0
    move_action_gain: float = 4.0

    @property
    def dt(self) -> float:
        return 1.0 / self.control_freq_hz


class ReactiveScriptedPolicy:
    """Object-feedback controller with no deck pose, phase, or t0 access."""

    requires_privileged = ("object",)
    phases = ("settle", "approach", "descend", "grasp", "lift", "transfer_hold", "place", "release")

    def __init__(self, env, config: ReactiveConfig | None = None):
        self.env = env
        self.config = config or ReactiveConfig(control_freq_hz=int(env.control_freq))
        if env.action_dim != 8:
            raise ValueError("reactive policy requires SpikeDirectPandaGripper (8D action)")
        if self.config.grip_mode == "force_limited_close" and env.gripper_force_limit_n is None:
            raise ValueError("force_limited_close requires a finite gripper actuator force limit")
        self.phase_index = 0
        self.phase_time_s = 0.0
        self.failure_reason: str | None = None
        self.finished = False
        self.phase_history: list[dict] = [{"phase": self.phase, "time_s": 0.0}]
        self.policy_command_count = 0
        self.hold_started = False
        self.hold_completed = False
        self.hold_time_s = 0.0
        self.contact_loss_time_s = 0.0
        self.bilateral_time_s = 0.0
        self.hand_minus_object_b_at_grasp: np.ndarray | None = None
        self.latched_finger_target_m: float | None = None
        self.initial_hand_b = eef_position_b(env).copy()
        self.initial_object_b = object_position_b(env).copy()
        self.initial_eef_rotation_b = self._eef_rotation_b()
        self.grasp_rotation_b = self._aligned_grasp_rotation_b()
        self.grasp_hand_b: np.ndarray | None = None
        self.lift_target_b: np.ndarray | None = None
        self.hold_target_b: np.ndarray | None = None
        self.place_target_b: np.ndarray | None = None
        self.finger_target_m = self._finger_half_opening_m()

    @property
    def phase(self) -> str:
        if self.finished:
            return "finished"
        return self.phases[min(self.phase_index, len(self.phases) - 1)]

    def _advance(self) -> None:
        self.phase_index += 1
        self.phase_time_s = 0.0
        if self.phase_index >= len(self.phases):
            self.finished = True
            self.phase_history.append({"phase": "finished", "time_s": float(self.env.sim.data.time)})
        else:
            self.phase_history.append({"phase": self.phase, "time_s": float(self.env.sim.data.time)})

    def _fail(self, reason: str) -> None:
        self.failure_reason = reason
        self.finished = True
        self.phase_history.append(
            {"phase": "failed", "reason": reason, "time_s": float(self.env.sim.data.time)}
        )

    def _finger_half_opening_m(self) -> float:
        joints = self.env.robots[0].gripper["right"].joints
        left = float(self.env.sim.data.get_joint_qpos(joints[0]))
        right = float(self.env.sim.data.get_joint_qpos(joints[1]))
        return 0.5 * (left - right)

    def _eef_rotation_b(self) -> np.ndarray:
        site_ids = self.env.robots[0].eef_site_id
        site_id = int(site_ids.get("right", next(iter(site_ids.values()))))
        base_id = self.env.sim.model.body_name2id("robot0_base")
        rotation_w_b = np.asarray(self.env.sim.data.body_xmat[base_id]).reshape(3, 3)
        rotation_w_e = np.asarray(self.env.sim.data.site_xmat[site_id]).reshape(3, 3)
        return rotation_w_b.T @ rotation_w_e

    def _aligned_grasp_rotation_b(self) -> np.ndarray:
        base_id = self.env.sim.model.body_name2id("robot0_base")
        cube_id = self.env.sim.model.body_name2id("cube_main")
        rotation_w_b = np.asarray(self.env.sim.data.body_xmat[base_id]).reshape(3, 3)
        rotation_w_o = np.asarray(self.env.sim.data.body_xmat[cube_id]).reshape(3, 3)
        rotation_b_o = rotation_w_b.T @ rotation_w_o
        short_axis = int(np.argmin(np.asarray(self.env.cube.size[:2])))
        object_axis = rotation_b_o[:, short_axis]
        jaw_axis = self.initial_eef_rotation_b[:, 0]
        object_yaw = float(np.arctan2(object_axis[1], object_axis[0]))
        jaw_yaw = float(np.arctan2(jaw_axis[1], jaw_axis[0]))
        delta = (object_yaw - jaw_yaw + np.pi) % (2.0 * np.pi) - np.pi
        if delta > 0.5 * np.pi:
            delta -= np.pi
        elif delta < -0.5 * np.pi:
            delta += np.pi
        return Rotation.from_rotvec(np.array((0.0, 0.0, delta))).as_matrix() @ self.initial_eef_rotation_b

    def _orientation_action(self, target_rotation_b: np.ndarray) -> np.ndarray:
        current = self._eef_rotation_b()
        error = target_rotation_b @ current.T
        rotvec = Rotation.from_matrix(error).as_rotvec()
        return np.clip(2.0 * rotvec / 0.5, -1.0, 1.0)

    @staticmethod
    def _finger_action(target_m: float) -> np.ndarray:
        target_m = float(np.clip(target_m, 0.0, 0.04))
        return np.array((target_m / 0.02 - 1.0, 1.0 - target_m / 0.02), dtype=np.float64)

    def _move_action(self, current_b: np.ndarray, target_b: np.ndarray, speed_m_s: float) -> np.ndarray:
        delta = target_b - current_b
        distance = float(np.linalg.norm(delta))
        max_step = speed_m_s * self.config.dt
        if distance > max_step:
            delta *= max_step / distance
        # BASIC OSC_POSE maps normalized translation actions to +/- 0.05 m.
        # Its achieved-frame goal is compliant and realizes about one quarter
        # of a small requested step here, so this gain restores the authored
        # Cartesian rate while the feedback loop still clips at the target.
        return np.clip(self.config.move_action_gain * delta / 0.05, -1.0, 1.0)

    def _finger_guard_triggered(self) -> bool:
        object_z = float(self.env.sim.data.body_xpos[self.env.sim.model.body_name2id("cube_main")][2])
        body_ids = [
            self.env.sim.model.body_name2id("gripper0_right_leftfinger"),
            self.env.sim.model.body_name2id("gripper0_right_rightfinger"),
        ]
        finger_z = min(float(self.env.sim.data.body_xpos[body_id][2]) for body_id in body_ids)
        return finger_z < object_z + 0.002

    def _monitor_grasp(self, contacts: ContactSnapshot, hand_b: np.ndarray, object_b: np.ndarray) -> None:
        if self.hand_minus_object_b_at_grasp is None:
            return
        slip = float(np.linalg.norm((hand_b - object_b) - self.hand_minus_object_b_at_grasp))
        if slip > SLIP_TOLERANCE_M:
            self._fail("grasp_slip_exceeded")
            return
        if contacts.bilateral:
            self.contact_loss_time_s = 0.0
        else:
            self.contact_loss_time_s += self.config.dt
            if self.contact_loss_time_s >= self.config.contact_loss_timeout_s:
                self._fail("grasp_contact_lost")

    def command(self, contacts: ContactSnapshot) -> np.ndarray:
        policy_step_index = self.policy_command_count
        self.policy_command_count += 1
        action = np.zeros(self.env.action_dim, dtype=np.float64)
        if self.finished:
            if self.latched_finger_target_m is not None:
                action[-2:] = self._finger_action(self.latched_finger_target_m)
            return action

        hand_b = eef_position_b(self.env)
        object_b = object_position_b(self.env)
        phase = self.phase

        if phase in ("lift", "transfer_hold", "place"):
            self._monitor_grasp(contacts, hand_b, object_b)
            if self.finished:
                return self.command(contacts)
        if phase in ("descend", "grasp"):
            if contacts.finger_table_n > CONTACT_THRESHOLD_N and not contacts.any_cube:
                self._fail("descend_table_contact")
                return action
            if self._finger_guard_triggered():
                self._fail("grasp_z_guard_triggered")
                return action

        target_b = hand_b.copy()
        arm_speed = self.config.arm_speed_m_s
        if phase == "settle":
            target_b = self.initial_hand_b
            self.finger_target_m = min(0.04, self.finger_target_m + self.config.gripper_opening_speed_m_s * self.config.dt)
            if self.phase_time_s >= 0.75:
                self._advance()
        elif phase == "approach":
            target_b = object_b + np.array((0.0, 0.0, 0.10))
            approach_error = float(np.linalg.norm(target_b - hand_b))
            if approach_error < 0.04:
                # Begin the authored 10 mm/s closure only after Cartesian and
                # yaw alignment, leaving the full two-second descend window
                # for contact detection rather than actuator travel.
                self.finger_target_m = max(
                    0.030,
                    self.finger_target_m - self.config.gripper_closing_speed_m_s * self.config.dt,
                )
            else:
                self.finger_target_m = min(
                    0.04,
                    self.finger_target_m + self.config.gripper_opening_speed_m_s * self.config.dt,
                )
            if approach_error < 0.004:
                self._advance()
            elif self.phase_time_s >= 5.0:
                self._fail("descend_contact_timeout")
        elif phase == "descend":
            # Pad centres need only enter the object's vertical span; retaining
            # 12 mm centre clearance keeps the main finger bodies off the table.
            target_b = object_b + np.array((0.0, 0.0, 0.012))
            arm_speed = self.config.descend_speed_m_s
            self.finger_target_m = max(
                0.0,
                self.finger_target_m - self.config.gripper_closing_speed_m_s * self.config.dt,
            )
            if contacts.any_cube:
                self.grasp_hand_b = hand_b.copy()
                self._advance()
            elif self.phase_time_s >= self.config.descend_timeout_s:
                self._fail("descend_contact_timeout")
        elif phase == "grasp":
            target_b = self.grasp_hand_b.copy() if self.grasp_hand_b is not None else hand_b.copy()
            if self.latched_finger_target_m is None:
                self.finger_target_m = max(
                    0.0,
                    self.finger_target_m - self.config.gripper_closing_speed_m_s * self.config.dt,
                )
                if contacts.bilateral:
                    current = self._finger_half_opening_m()
                    if self.config.grip_mode == "force_limited_close":
                        # Full-close position command deliberately saturates at
                        # the actuator forcerange configured by the environment.
                        # Unlike a frozen opening, force then remains available
                        # when relative lateral motion unloads either pad.
                        self.latched_finger_target_m = 0.0
                        switch_event = self.env.activate_hold_force_limit(
                            policy_step_index=policy_step_index,
                            trigger="bilateral_latch",
                        )
                        self.phase_history.append(switch_event)
                    else:
                        self.latched_finger_target_m = max(
                            0.0,
                            current - self.config.gripper_contact_preload_m,
                        )
                    self.finger_target_m = self.latched_finger_target_m
                    self.hand_minus_object_b_at_grasp = (hand_b - object_b).copy()
            else:
                self.finger_target_m = self.latched_finger_target_m
                # Bilateral acquisition already froze a preloaded physical
                # target. Allow that target to settle for 0.3 s; transport's
                # independent 0.2 s contact-loss monitor remains authoritative.
                self.bilateral_time_s += self.config.dt
                if self.bilateral_time_s >= self.config.grasp_settle_s:
                    self.grasp_hand_b = hand_b.copy()
                    self.lift_target_b = hand_b + np.array((0.0, 0.0, 0.12))
                    self._advance()
            if self.phase_time_s >= self.config.grasp_timeout_s and not self.finished:
                self._fail("grasp_contact_timeout")
        elif phase == "lift":
            target_b = self.lift_target_b.copy()
            self.finger_target_m = self.latched_finger_target_m
            if np.linalg.norm(target_b - hand_b) < 0.006:
                self.hold_target_b = target_b + np.array((0.0, 0.15, 0.0))
                self._advance()
            elif self.phase_time_s >= 3.0:
                self._fail("grasp_contact_lost")
        elif phase == "transfer_hold":
            target_b = self.hold_target_b.copy()
            self.finger_target_m = self.latched_finger_target_m
            if np.linalg.norm(target_b - hand_b) < 0.010:
                self.hold_started = True
                self.hold_time_s += self.config.dt
                if self.hold_time_s >= self.config.transfer_hold_s:
                    desired_object_b = self.initial_object_b + np.array((0.0, 0.15, 0.0))
                    self.place_target_b = desired_object_b + self.hand_minus_object_b_at_grasp
                    self.hold_completed = True
                    self._advance()
            if self.phase_time_s >= 8.0 and not self.finished:
                self._fail("grasp_contact_lost")
        elif phase == "place":
            target_b = self.place_target_b.copy()
            arm_speed = self.config.descend_speed_m_s
            self.finger_target_m = self.latched_finger_target_m
            if np.linalg.norm(target_b - hand_b) < 0.006:
                self._advance()
            elif self.phase_time_s >= 3.0:
                self._fail("grasp_contact_lost")
        elif phase == "release":
            target_b = self.place_target_b.copy()
            self.finger_target_m = min(
                0.04,
                self.finger_target_m + self.config.gripper_opening_speed_m_s * self.config.dt,
            )
            if self.phase_time_s >= 0.75:
                self._advance()

        if not self.finished:
            action[:3] = self._move_action(hand_b, target_b, arm_speed)
            target_rotation = self.initial_eef_rotation_b if phase == "settle" else self.grasp_rotation_b
            action[3:6] = self._orientation_action(target_rotation)
            action[-2:] = self._finger_action(self.finger_target_m)
            self.phase_time_s += self.config.dt
        return action


class FrozenReplayPolicy:
    """Open-loop policy that returns a previously recorded 20 Hz action tape."""

    requires_privileged: tuple[str, ...] = ()

    def __init__(self, actions: list[list[float]]):
        self.actions = np.asarray(actions, dtype=np.float64)
        self.index = 0

    def command(self) -> np.ndarray:
        if self.index >= len(self.actions):
            return self.actions[-1].copy()
        action = self.actions[self.index].copy()
        self.index += 1
        return action
