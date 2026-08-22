"""ShakeBench-compatible contact, slip, and episode metrics for the spike."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING

import mujoco
import numpy as np

if TYPE_CHECKING:
    from env_shakedeck import ShakeDeckLift
    from policies import ReactiveScriptedPolicy


SLIP_TOLERANCE_M = 0.010
CONTACT_THRESHOLD_N = 0.05


def point_in_body_frame(env: ShakeDeckLift, body_id: int, point_w: np.ndarray) -> np.ndarray:
    rotation_w = np.asarray(env.sim.data.body_xmat[body_id]).reshape(3, 3)
    origin_w = np.asarray(env.sim.data.body_xpos[body_id])
    return rotation_w.T @ (point_w - origin_w)


def eef_position_b(env: ShakeDeckLift) -> np.ndarray:
    site_ids = env.robots[0].eef_site_id
    site_id = int(site_ids.get("right", next(iter(site_ids.values()))))
    base_id = env.sim.model.body_name2id("robot0_base")
    return point_in_body_frame(env, base_id, np.asarray(env.sim.data.site_xpos[site_id]))


def object_position_b(env: ShakeDeckLift) -> np.ndarray:
    base_id = env.sim.model.body_name2id("robot0_base")
    cube_id = env.sim.model.body_name2id("cube_main")
    return point_in_body_frame(env, base_id, np.asarray(env.sim.data.body_xpos[cube_id]))


def table_position_b(env: ShakeDeckLift) -> np.ndarray:
    """Return the table support origin in the robot-base frame."""

    base_id = env.sim.model.body_name2id("robot0_base")
    table_support_id = (
        env.table_deck_body_id if env.decoupled_table else env.deck_body_id
    )
    return point_in_body_frame(
        env,
        base_id,
        np.asarray(env.sim.data.body_xpos[table_support_id]),
    )


def object_position_t(env: ShakeDeckLift) -> np.ndarray:
    """Return the cube position in the physical table-support frame."""

    table_support_id = (
        env.table_deck_body_id if env.decoupled_table else env.deck_body_id
    )
    cube_id = env.sim.model.body_name2id("cube_main")
    return point_in_body_frame(
        env,
        table_support_id,
        np.asarray(env.sim.data.body_xpos[cube_id]),
    )


@dataclass(frozen=True)
class ContactSnapshot:
    left_cube_n: float
    right_cube_n: float
    finger_table_n: float
    max_penetration_m: float

    @property
    def bilateral(self) -> bool:
        return self.left_cube_n > CONTACT_THRESHOLD_N and self.right_cube_n > CONTACT_THRESHOLD_N

    @property
    def any_cube(self) -> bool:
        return self.left_cube_n > CONTACT_THRESHOLD_N or self.right_cube_n > CONTACT_THRESHOLD_N


def contact_snapshot(env: ShakeDeckLift) -> ContactSnapshot:
    gripper = env.robots[0].gripper["right"]
    left = {env.sim.model.geom_name2id(name) for name in gripper.important_geoms["left_fingerpad"]}
    right = {env.sim.model.geom_name2id(name) for name in gripper.important_geoms["right_fingerpad"]}
    all_fingers = {
        env.sim.model.geom_name2id(name)
        for key in ("left_finger", "right_finger")
        for name in gripper.important_geoms[key]
    }
    cube = env.sim.model.geom_name2id("cube_g0")
    table = env.sim.model.geom_name2id("table_collision")
    left_force = 0.0
    right_force = 0.0
    table_force = 0.0
    max_penetration = 0.0
    wrench = np.zeros(6, dtype=np.float64)
    for index, contact in enumerate(env.sim.data.contact[: env.sim.data.ncon]):
        geom1, geom2 = int(contact.geom1), int(contact.geom2)
        max_penetration = max(max_penetration, max(0.0, -float(contact.dist)))
        mujoco.mj_contactForce(env.sim.model._model, env.sim.data._data, index, wrench)
        normal_n = abs(float(wrench[0]))
        pair = {geom1, geom2}
        if cube in pair:
            finger = next((geom for geom in pair if geom != cube), None)
            if finger in left:
                left_force += normal_n
            if finger in right:
                right_force += normal_n
        if table in pair and any(geom in all_fingers for geom in pair):
            table_force += normal_n
    return ContactSnapshot(left_force, right_force, table_force, max_penetration)


@dataclass
class EpisodeMetrics:
    initial_object_b: np.ndarray
    initial_object_t: np.ndarray
    initial_table_b: np.ndarray
    max_grasp_slip_m: float = 0.0
    max_penetration_m: float = 0.0
    ee_wobble_base_m: float = 0.0
    obj_slip_on_table_m: float = 0.0
    base_frame_table_motion_m: float = 0.0
    max_object_lift_m: float = 0.0
    _hold_reference_b: np.ndarray | None = None
    _left_force_n: list[float] = field(default_factory=list)
    _right_force_n: list[float] = field(default_factory=list)

    @classmethod
    def start(cls, env: ShakeDeckLift) -> "EpisodeMetrics":
        return cls(
            initial_object_b=object_position_b(env).copy(),
            initial_object_t=object_position_t(env).copy(),
            initial_table_b=table_position_b(env).copy(),
        )

    def update(
        self,
        env: ShakeDeckLift,
        policy: ReactiveScriptedPolicy,
        contacts: ContactSnapshot,
    ) -> None:
        hand_b = eef_position_b(env)
        object_b = object_position_b(env)
        object_t = object_position_t(env)
        table_b = table_position_b(env)
        self.max_penetration_m = max(self.max_penetration_m, contacts.max_penetration_m)
        self.max_object_lift_m = max(
            self.max_object_lift_m,
            float(object_b[2] - self.initial_object_b[2]),
        )
        if policy.hand_minus_object_b_at_grasp is None:
            self.obj_slip_on_table_m = max(
                self.obj_slip_on_table_m,
                float(np.linalg.norm(object_t - self.initial_object_t)),
            )
        else:
            # Release intentionally removes both contacts, so force diagnostics
            # cover only the interval in which the latch is meant to hold.
            if policy.phase in ("grasp", "lift", "transfer_hold", "place"):
                self._left_force_n.append(contacts.left_cube_n)
                self._right_force_n.append(contacts.right_cube_n)
            slip = float(
                np.linalg.norm(
                    (hand_b - object_b) - policy.hand_minus_object_b_at_grasp
                )
            )
            self.max_grasp_slip_m = max(self.max_grasp_slip_m, slip)
        if policy.hold_started:
            if self._hold_reference_b is None:
                self._hold_reference_b = hand_b.copy()
            self.ee_wobble_base_m = max(
                self.ee_wobble_base_m,
                float(np.linalg.norm(hand_b - self._hold_reference_b)),
            )
        self.base_frame_table_motion_m = max(
            self.base_frame_table_motion_m,
            float(np.linalg.norm(table_b - self.initial_table_b)),
        )

    def result(self, policy: ReactiveScriptedPolicy, warning_count: int) -> dict:
        slip_exceeded = self.max_grasp_slip_m > SLIP_TOLERANCE_M
        success = bool(
            policy.finished
            and policy.failure_reason is None
            and policy.hold_completed
            and self.max_object_lift_m >= 0.05
            and not slip_exceeded
        )
        failure_reason = policy.failure_reason
        if failure_reason is None and slip_exceeded:
            failure_reason = "grasp_slip_exceeded"
        force_samples = len(self._left_force_n)
        both_zero = sum(
            left == 0.0 and right == 0.0
            for left, right in zip(self._left_force_n, self._right_force_n, strict=True)
        )
        left_below = sum(value <= CONTACT_THRESHOLD_N for value in self._left_force_n)
        right_below = sum(value <= CONTACT_THRESHOLD_N for value in self._right_force_n)
        unilateral_below = sum(
            (left <= CONTACT_THRESHOLD_N) != (right <= CONTACT_THRESHOLD_N)
            for left, right in zip(self._left_force_n, self._right_force_n, strict=True)
        )
        bilateral_below = sum(
            left <= CONTACT_THRESHOLD_N and right <= CONTACT_THRESHOLD_N
            for left, right in zip(self._left_force_n, self._right_force_n, strict=True)
        )

        def force_summary(values: list[float]) -> dict:
            if not values:
                return {"mean": None, "min": None, "max": None}
            array = np.asarray(values, dtype=np.float64)
            return {
                "mean": float(np.mean(array)),
                "min": float(np.min(array)),
                "max": float(np.max(array)),
            }

        return {
            "success": success,
            "failure_reason": None if success else failure_reason or "episode_timeout",
            "max_grasp_slip_m": self.max_grasp_slip_m,
            "grasp_slip_exceeded": slip_exceeded,
            "max_penetration_m": self.max_penetration_m,
            "ee_wobble_base_m": self.ee_wobble_base_m,
            "obj_slip_on_table_m": self.obj_slip_on_table_m,
            "base_frame_table_motion_m": self.base_frame_table_motion_m,
            "max_object_lift_m": self.max_object_lift_m,
            "transfer_hold_s": policy.hold_time_s,
            "post_latch_finger_force_n": {
                "sample_count": force_samples,
                "left": force_summary(self._left_force_n),
                "right": force_summary(self._right_force_n),
                "both_zero_fraction": both_zero / force_samples if force_samples else None,
                "contact_threshold_n": CONTACT_THRESHOLD_N,
                "left_below_threshold_fraction": (
                    left_below / force_samples if force_samples else None
                ),
                "right_below_threshold_fraction": (
                    right_below / force_samples if force_samples else None
                ),
                "exactly_one_below_threshold_fraction": (
                    unilateral_below / force_samples if force_samples else None
                ),
                "both_below_threshold_fraction": (
                    bilateral_below / force_samples if force_samples else None
                ),
            },
            "phase_history": policy.phase_history,
            "mujoco_warning_count": warning_count,
        }
