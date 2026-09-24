"""Privileged, closed-loop planar pushing oracle for the static Push-T task."""

import numpy as np

from robosuite.utils.control_utils import orientation_error
from shakebench.environments.push_t import COVERAGE_THRESHOLD, PRE_STANDOFF_M
from shakebench.models.objects.push_t import OUTLINE

# The two non-overlapping collision boxes have uniform density.
PRESSURE_CENTER = np.array([0.0, (0.0045 * 0.06 - 0.0036 * 0.015) / 0.0081])
# Closed Panda fingertip with the closing axis along the push: one finger face, 17.8 mm wide (collision mesh).
PUSHER_HALF_WIDTH = 0.0089
# Delta-mode OSC (kp 150, critical damping) tracks a per-step delta d at about d / OSC_LAG_S metres per second.
OSC_LAG_S = 2.0 / np.sqrt(150.0)
# The T has been engaged once it has moved this far along the push in the table frame.
ENGAGE_MOTION_M = 0.0015
# Stop decisions use an exponentially smoothed error so table-excitation jitter does not end a push early.
SCORE_SMOOTHING = 0.5


class PushTOracle:
    """Choose short pushes from privileged T pose and verify after withdrawal."""

    def __init__(self, env):
        self.env = env
        self.abort_requested = False
        self.verified = False
        self.phase = "idle"
        self.trace = []
        self.site = env.robots[0].eef_site_id["right"]
        self.controller = env.robots[0].part_controllers["right"]
        self.worktable = env.sim.model.body_name2id(env.worktable_body_name)
        self.pads = [env.sim.model.geom_name2id(f"gripper0_right_finger{index}_pad_collision") for index in (1, 2)]
        self.pusher_geoms = {
            env.sim.model.geom_name2id(name)
            for name in env.sim.model.geom_names
            if name.startswith("gripper0_right_") and name.endswith("_collision")
        }
        wrist_x = env.sim.data.site_xmat[self.site].reshape(3, 3)[:, 0].copy()
        wrist_x[2] = 0
        wrist_x /= np.linalg.norm(wrist_x)
        down = np.array([0.0, 0.0, -1.0])
        self.orientation = np.column_stack((wrist_x, np.cross(down, wrist_x), down))
        # The closed finger pair is symmetric under a half turn, so every push can keep the tool yaw within
        # +-90 deg of the start pose; choosing relative to the current pose instead lets the wrist wind up.
        self.home_closing = wrist_x.copy()
        self._actions = self._run()

    def action(self):
        return next(self._actions, self._command(self._eef(), 1))

    def _eef(self):
        return self.env.sim.data.site_xpos[self.site].copy()

    def _table(self):
        data = self.env.sim.data._data
        rotation = data.xmat[self.worktable].reshape(3, 3).copy()
        top = data.xpos[self.worktable] + rotation @ [0, 0, self.env.arena.table_half_size[2]]
        return top, rotation

    def _world(self, point):
        top, rotation = self._table()
        return top + rotation @ point

    def _pose(self):
        data = self.env.sim.data._data
        top, table_rotation = self._table()
        position = table_rotation.T @ (data.xpos[self.env.tee_body_id] - top)
        rotation = table_rotation.T @ data.xmat[self.env.tee_body_id].reshape(3, 3)
        return position[:2], np.arctan2(rotation[1, 0], rotation[0, 0])

    def _pusher(self):
        data = self.env.sim.data._data
        return np.mean(data.geom_xpos[self.pads], axis=0).copy()

    def _contact(self):
        data = self.env.sim.data._data
        return any(
            (c.geom1 in self.env.tee_geom_ids and c.geom2 in self.pusher_geoms)
            or (c.geom2 in self.env.tee_geom_ids and c.geom1 in self.pusher_geoms)
            for c in data.contact[: data.ncon]
        )

    def _command(self, target, grip=1, limit=0.008, rotation_limit=0.08):
        """Convert a world-space position error to the 5 cm normalized OSC delta."""
        delta = np.clip(np.asarray(target) - self._eef(), -limit, limit)
        rotation = orientation_error(self.orientation, self.env.sim.data.site_xmat[self.site].reshape(3, 3))
        if self.controller.input_ref_frame == "base":
            delta = self.controller.origin_ori.T @ delta
            rotation = self.controller.origin_ori.T @ rotation
        return np.r_[
            np.clip(delta / self.controller.output_max[:3], -1, 1),
            np.clip(rotation / 0.5, -rotation_limit, rotation_limit),
            grip,
        ]

    def _orient(self):
        target = self._eef()
        for _ in range(150):
            error = orientation_error(self.orientation, self.env.sim.data.site_xmat[self.site].reshape(3, 3))
            if np.linalg.norm(error) < 0.03:
                return True
            if np.linalg.norm(self._eef() - target) > 0.02:
                return False
            yield self._command(target, rotation_limit=0.5)
        return False

    def _move(self, target, *, speed=0.14, tolerance=0.003, timeout=120, rotation_limit=0.08):
        """Move the end effector toward a world point, aborting on stalls."""
        previous = self._eef()
        stalled = 0
        for _ in range(timeout):
            current = self._eef()
            delta = target - current
            distance = np.linalg.norm(delta)
            if distance < tolerance:
                return True
            step = min(distance, speed * OSC_LAG_S)
            yield self._command(
                current + delta * step / distance, limit=max(0.008, step), rotation_limit=rotation_limit
            )
            stalled = stalled + 1 if np.linalg.norm(self._eef() - previous) < 2e-5 else 0
            if stalled >= 25:
                return False
            previous = self._eef()
        return False

    def _score(self, position, yaw):
        target = np.asarray(self.env.target_xy_m)
        target_yaw = self.env.target_yaw_rad
        actual = position + OUTLINE @ self._rot(yaw).T
        desired = target + OUTLINE @ self._rot(target_yaw).T
        return float(np.linalg.norm(actual - desired, axis=1).mean())

    @staticmethod
    def _rot(yaw):
        return np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])

    def _plan(self):
        position, yaw = self._pose()
        rotation = self._rot(yaw)
        baseline = self._score(position, yaw)
        current = self._pusher()
        top, table_rotation = self._table()
        current_xy = (table_rotation.T @ (current - top))[:2]
        best = None
        best_value = 0.0002
        lookahead = 0.008 if baseline < 0.012 else 0.020
        # ponytail: The fixed pressure-center model omits stick/slip; replace only if rollout errors demand it.
        c = 0.055
        for start, end in zip(OUTLINE, np.roll(OUTLINE, -1, axis=0)):
            tangent = end - start
            length = np.linalg.norm(tangent)
            outward = np.array([tangent[1], -tangent[0]]) / length
            direction = -(rotation @ outward)
            margin = min(0.012, length / 2)
            count = max(1, int((length - 2 * margin) / 0.012) + 1)
            fractions = [0.5] if count == 1 else np.linspace(margin / length, 1 - margin / length, count)
            for fraction in fractions:
                point = start + fraction * tangent
                contact = position + rotation @ point
                pre = contact - direction * PRE_STANDOFF_M
                push_distance = (
                    0.008
                    if baseline < 0.012
                    else float(
                        np.clip(
                            1.7 * np.dot(np.asarray(self.env.target_xy_m) - position, direction),
                            0.02,
                            0.12,
                        )
                    )
                )
                finish = contact + direction * push_distance
                safe = pre - direction * 0.040
                if not all(-0.33 <= xy[0] <= 0.11 and abs(xy[1]) <= 0.26 for xy in (pre, finish)) or not (
                    -0.35 <= safe[0] <= 0.14 and abs(safe[1]) <= 0.29
                ):
                    continue
                # Line contact: the flat finger face carries the resultant at the point of the contact
                # segment nearest to the line through the pressure centre along the push direction.
                unit = tangent / length
                along = np.clip(
                    np.dot(PRESSURE_CENTER - start, unit),
                    max(0.0, fraction * length - PUSHER_HALF_WIDTH),
                    min(length, fraction * length + PUSHER_HALF_WIDTH),
                )
                arm = rotation @ (start + along * unit - PRESSURE_CENTER)
                torque = np.linalg.det(np.stack((arm, direction)))
                lengths = (0.002, 0.004, 0.008) if baseline < 0.012 else (lookahead,)
                predicted, used = min(
                    (
                        self._score(
                            position + direction * (length_ / (1 + (torque / c) ** 2)),
                            yaw + length_ / (1 + (torque / c) ** 2) * torque / c**2,
                        ),
                        length_,
                    )
                    for length_ in lengths
                )
                if baseline < 0.012:
                    push_distance = 2 * used
                improvement = baseline - predicted
                travel = np.linalg.norm(pre - current_xy)
                value = improvement - 0.0015 * travel - (0.0005 if self.trace else 0)
                if value > best_value:
                    best_value = value
                    best = (pre, direction, push_distance, predicted, torque / c**2)
        return best

    def _run(self):
        env = self.env
        for attempt in range(45):
            if env.get_metrics()["task_rule_violation"]:
                self.abort_requested = True
                return
            if env.get_metrics()["coverage"]["final"] >= COVERAGE_THRESHOLD:
                self.phase = "verify"
                for _ in range(12):
                    yield self._command(self._eef())
                metrics = env.get_metrics()
                if metrics["success"]["passed"] and metrics["coverage"]["final"] >= COVERAGE_THRESHOLD:
                    self.verified = True
                    self.phase = "done"
                    return
            plan = self._plan()
            if plan is None:
                self.abort_requested = True
                return
            pre, direction, distance, predicted, yaw_rate = plan
            top, table_rotation = self._table()
            direction_world = table_rotation @ np.r_[direction, 0.0]
            down = -table_rotation[:, 2]
            reference = self.home_closing
            if abs(np.dot(direction_world, reference)) < 0.2:
                reference = self.env.sim.data.site_xmat[self.site].reshape(3, 3)[:, 0]
            closing = direction_world if np.dot(direction_world, reference) >= 0 else -direction_world
            self.orientation = np.column_stack((closing, np.cross(down, closing), down))
            above = self._world(np.r_[pre - direction * 0.040, 0.045])
            low = self._world(np.r_[pre, 0.016])
            self.phase = "transit"
            if not (yield from self._move(above, speed=0.16, timeout=220, rotation_limit=0.5)):
                self.abort_requested = True
                return
            self.phase = "orient"
            if not (yield from self._orient()):
                self.abort_requested = True
                return
            self.phase = "descend"
            if not (yield from self._move(low, speed=0.06, timeout=100)):
                self.abort_requested = True
                return
            self.phase = "approach"
            pusher_offset = self._pusher() - self._eef()
            reference = low + pusher_offset
            approach_position, _ = self._pose()
            engaged = False
            for _ in range(50):
                position, _ = self._pose()
                if self._contact() or np.dot(position - approach_position, direction) > ENGAGE_MOTION_M:
                    engaged = True
                    break
                reference += direction_world * 0.0005
                yield self._command(reference - pusher_offset, limit=0.005)
            if engaged:
                self.phase = "push"
                before_position, before_yaw = self._pose()
                baseline = self._score(before_position, before_yaw)
                start_pusher = self._pusher()
                start_eef = self._eef()
                best_score, stale, score = baseline, 0, baseline
                for index in range(int(distance / 0.001) + 120):
                    if np.dot(self._pusher() - start_pusher, direction_world) >= distance:
                        break
                    # Hold the pusher on the planned world line: lateral and height errors are corrected,
                    # progress along the line is limited to `lead` ahead of the current pusher.
                    lead = 0.001 if score < 0.02 else 0.0025
                    progress = np.dot(self._eef() - start_eef, direction_world)
                    yield self._command(start_eef + direction_world * (progress + lead), limit=0.0025)
                    position, yaw = self._pose()
                    score = SCORE_SMOOTHING * score + (1 - SCORE_SMOOTHING) * self._score(position, yaw)
                    if score < best_score - 0.0001:
                        best_score, stale = score, 0
                    else:
                        stale += 1
                    moved = np.dot(position - before_position, direction)
                    turned = np.arctan2(np.sin(yaw - before_yaw), np.cos(yaw - before_yaw))
                    if (
                        score > best_score + (0.0003 if best_score < 0.012 else 0.0015)
                        or stale >= 12
                        or abs(turned - yaw_rate * moved) > 0.05 + 0.5 * abs(yaw_rate * moved)
                    ):
                        break
                after_position, after_yaw = self._pose()
                self.trace.append(
                    {
                        "attempt": attempt,
                        "predicted_error_m": predicted,
                        "actual_error_m": self._score(after_position, after_yaw),
                        "translation_m": float(np.linalg.norm(after_position - before_position)),
                        "rotation_rad": float(
                            np.arctan2(np.sin(after_yaw - before_yaw), np.cos(after_yaw - before_yaw))
                        ),
                    }
                )
            self.phase = "retreat"
            if not (yield from self._move(self._eef() - direction_world * 0.004, speed=0.02, timeout=30)):
                self.abort_requested = True
                return
            retreat = self._eef() - direction_world * 0.021
            if not (yield from self._move(retreat, speed=0.16, timeout=45)):
                self.abort_requested = True
                return
            high = self._eef().copy()
            high[2] = above[2]
            if not (yield from self._move(high, speed=0.10, timeout=60)):
                self.abort_requested = True
                return
            self.phase = "settle"
            for _ in range(4):
                yield self._command(self._eef())
        self.abort_requested = True
