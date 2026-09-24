"""Privileged, closed-loop planar pushing oracle for the static Push-T task."""

import numpy as np

from robosuite.utils.control_utils import orientation_error
from shakebench.environments.push_t import PRE_STANDOFF_M, REACH_RADIUS_M
from shakebench.models.objects.push_t import CONTOUR_SAMPLE_SPACING_M, DECAL_RECTANGLES, OUTLINE

# The two non-overlapping collision boxes have uniform density.
PRESSURE_CENTER = np.array([0.0, (0.0045 * 0.06 - 0.0036 * 0.015) / 0.0081])
# Closed Panda fingertip with the closing axis along the push: one finger face, 17.8 mm wide (collision mesh).
PUSHER_HALF_WIDTH = 0.0089
# Delta-mode OSC (kp 150, critical damping) tracks a per-step delta d at about d / OSC_LAG_S metres per second.
OSC_LAG_S = 2.0 / np.sqrt(150.0)
# Pusher-object friction 0.5 -> 26.6 deg cone; stop predicting a push once the face has turned 20 deg.
FRICTION_CONE_RAD = np.radians(20.0)
FAR_LENGTHS = (0.02, 0.04, 0.06, 0.09, 0.12)
NEAR_LENGTHS = (0.002, 0.004, 0.008)
# Top-down reach at push height, measured as horizontal distance from the robot base: 0.766 m already
# stalls 4 mm above the push height, 0.78 m tracks with ~2 mm error at 25 mm.
# Push directions within the 26.6 deg friction cone: the T follows a sticking pusher off the face normal.
SKEWS_RAD = (np.radians(-15.0), 0.0, np.radians(15.0))
# Rotate first when far off in yaw: greedy vertex error prefers translation and can strand the T where the
# faces needed to turn it are out of reach.
ROTATE_FIRST_RAD = np.radians(30.0)
ROTATE_FIRST_PULL = 0.3
# Translation pushes may not re-open the yaw error past this (otherwise the modes ping-pong).
TRANSLATE_MAX_YAW_RAD = np.radians(25.0)
# Predicted T outline must stay this far inside the tabletop edges.
TABLE_HALF_XY_M = np.array([0.325, 0.30])
TABLE_MARGIN_M = 0.010
# Motion budget (control steps are 50 ms; delta-mode OSC moves ~6.1 x the commanded delta per second).
TRANSIT_SITE_Z_M = 0.045  # fingertip ~36 mm; a 0.22 m/s transit sags up to ~8 mm (38 mm swept the T on state 6)
PUSH_SITE_Z_M = 0.016  # fingertip ~7 mm above the table
TRANSIT_SPEED_MPS = 0.22
DESCEND_SPEED_MPS = 0.12
# Descend as close as the finger footprint allows: first standoff (pusher centre to contact) whose footprint
# clears the T outline by FOOTPRINT_CLEARANCE_M; 20 mm lands on the crossbar next to the stem otherwise.
DESCEND_STANDOFFS_M = (0.020, 0.026, 0.032)
FOOTPRINT_CLEARANCE_M = 0.003
# Closed fingertip footprint: 29.6 mm along the push (two fingers in line) x 17.8 mm across.
FOOTPRINT_HALF_ALONG_M, FOOTPRINT_HALF_ACROSS_M = 0.0148, PUSHER_HALF_WIDTH
# Push lead by live error: ~6 mm/s in contact at 1.5 mm (1 mm lost to 0.3 N friction at ~0.6 N/mm).
FAR_LEAD_M, MID_LEAD_M, NEAR_LEAD_M = 0.0035, 0.0020, 0.0015
MID_ERROR_M, NEAR_ERROR_M = 0.020, 0.010
# Low-torque far-field pushes (predicted |yaw rate| below this) run at a longer lead (~20 mm/s).
FAST_LEAD_M = 0.004
FAST_MAX_YAW_RATE = 2.0  # rad per metre of object translation
# Stop a push once the whole T sits inside the decal with this much of its 8 mm margin to spare.
INSIDE_SPARE_M = 0.002
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
        self.base_world = env.sim.data.xpos[env.sim.model.body_name2id(env.robot_base_body_name)].copy()
        self.excluded = []
        self.rotating = False
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

    def _score(self, position, yaw, rotation_only=False):
        target = np.asarray(self.env.target_xy_m)
        if rotation_only:
            # goal orientation placed at the T's current pressure centre, plus a weak pull toward the goal
            com = position + self._rot(yaw) @ PRESSURE_CENTER
            goal_com = target + self._rot(self.env.target_yaw_rad) @ PRESSURE_CENTER
            target = com - self._rot(self.env.target_yaw_rad) @ PRESSURE_CENTER
            actual = position + OUTLINE @ self._rot(yaw).T
            desired = target + OUTLINE @ self._rot(self.env.target_yaw_rad).T
            return float(np.linalg.norm(actual - desired, axis=1).mean()) + ROTATE_FIRST_PULL * float(
                np.linalg.norm(com - goal_com)
            )
        target_yaw = self.env.target_yaw_rad
        actual = position + OUTLINE @ self._rot(yaw).T
        desired = target + OUTLINE @ self._rot(target_yaw).T
        return float(np.linalg.norm(actual - desired, axis=1).mean())

    @staticmethod
    def _rot(yaw):
        return np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])

    def _rollout(self, position, yaw, contact_body, direction, length, c=0.055, step=0.004):
        com = position + self._rot(yaw) @ PRESSURE_CENTER
        start_yaw, travelled = yaw, 0.0
        while travelled < length - 1e-9:
            ds = min(step, length - travelled)
            arm = self._rot(yaw) @ (contact_body - PRESSURE_CENTER)
            torque = arm[0] * direction[1] - arm[1] * direction[0]
            translation = ds / (1 + (torque / c) ** 2)
            com = com + direction * translation
            yaw = yaw + translation * torque / c**2
            travelled += ds
            if abs(yaw - start_yaw) > FRICTION_CONE_RAD:
                break
        return com - self._rot(yaw) @ PRESSURE_CENTER, yaw

    def _give_up_on(self, pre, above):
        """Exclude an unreachable pre-contact point, back up to transit height, and allow a replan."""
        self.excluded.append(np.asarray(pre, dtype=float).copy())
        self.phase = "recover"
        lifted = self._eef().copy()
        lifted[2] = max(lifted[2], above[2])
        yield from self._move(lifted, speed=0.08, timeout=40)
        if len(self.excluded) > 6:
            self.abort_requested = True
            return False
        return True

    def _inside_decal(self, position, yaw, spare=0.0):
        """T outline (sampled every 2 mm, flat) inside the decal rectangles shrunk by `spare`."""
        goal_rotation = self._rot(self.env.target_yaw_rad)
        corners = position + OUTLINE @ self._rot(yaw).T
        points = []
        for start, end in zip(corners, np.roll(corners, -1, axis=0)):
            count = max(1, int(np.ceil(np.linalg.norm(end - start) / CONTOUR_SAMPLE_SPACING_M)))
            points.append(start + np.linspace(0, 1, count, endpoint=False)[:, None] * (end - start))
        local = (np.vstack(points) - np.asarray(self.env.target_xy_m)) @ goal_rotation
        inside = np.zeros(len(local), dtype=bool)
        for center, size in DECAL_RECTANGLES:
            inside |= np.all(np.abs(local - center) <= np.asarray(size) - spare, axis=1)
        return bool(inside.all())

    def _footprint_clearance(self, centre, direction, position, yaw):
        """Smallest distance from the fingertip footprint boundary to the T outline (negative if overlapping)."""
        across = np.array([-direction[1], direction[0]])
        a, b = FOOTPRINT_HALF_ALONG_M, FOOTPRINT_HALF_ACROSS_M
        corners = [centre + direction * sa * a + across * sb * b for sa, sb in ((1, 1), (1, -1), (-1, -1), (-1, 1))]
        samples = np.vstack([np.linspace(corners[i], corners[(i + 1) % 4], 8, endpoint=False) for i in range(4)])
        polygon = position + OUTLINE @ self._rot(yaw).T
        starts, ends = polygon, np.roll(polygon, -1, axis=0)
        edges = ends - starts
        rel = samples[:, None, :] - starts[None, :, :]
        t = np.clip((rel * edges).sum(-1) / (edges**2).sum(-1), 0, 1)
        distance = np.linalg.norm(rel - t[..., None] * edges, axis=-1).min(axis=1)
        # even-odd rule: samples inside the T count as overlapping
        y0, y1 = starts[None, :, 1], ends[None, :, 1]
        crosses = ((y0 > samples[:, None, 1]) != (y1 > samples[:, None, 1])) & (
            samples[:, None, 0]
            < starts[None, :, 0]
            + (samples[:, None, 1] - y0) * edges[None, :, 0] / np.where(y1 - y0 == 0, 1e-12, y1 - y0)
        )
        inside = crosses.sum(axis=1) % 2 == 1
        return float(np.where(inside, -distance, distance).min())

    def _plan(self):
        position, yaw = self._pose()
        rotation = self._rot(yaw)
        baseline = self._score(position, yaw)
        yaw_error = np.arctan2(np.sin(self.env.target_yaw_rad - yaw), np.cos(self.env.target_yaw_rad - yaw))
        rotate_first = abs(yaw_error) > ROTATE_FIRST_RAD
        self.rotating = rotate_first
        rotation_baseline = self._score(position, yaw, rotation_only=True)
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
            normal = -(rotation @ outward)
            margin = min(0.012, length / 2)
            count = max(1, int((length - 2 * margin) / 0.012) + 1)
            fractions = [0.5] if count == 1 else np.linspace(margin / length, 1 - margin / length, count)
            for fraction, skew in ((f, k) for f in fractions for k in SKEWS_RAD):
                direction = np.array([[np.cos(skew), -np.sin(skew)], [np.sin(skew), np.cos(skew)]]) @ normal
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
                base_xy = (table_rotation.T @ (self.base_world - top))[:2]
                if max(np.linalg.norm(xy - base_xy) for xy in (pre, finish, safe)) > REACH_RADIUS_M:
                    continue
                if any(np.linalg.norm(pre - bad) < 0.01 for bad in self.excluded):
                    continue
                if not all(-0.33 <= xy[0] <= 0.11 and abs(xy[1]) <= 0.26 for xy in (pre, finish)) or not (
                    -0.35 <= safe[0] <= 0.14 and abs(safe[1]) <= 0.29
                ):
                    continue
                # Line contact: the flat finger face carries the resultant at the point of the contact
                # segment nearest to the line through the pressure centre along the push direction.
                unit = tangent / length
                d_body = rotation.T @ direction
                w = PRESSURE_CENTER - start
                denom = unit[0] * d_body[1] - unit[1] * d_body[0]
                along = np.clip(
                    (w[0] * d_body[1] - w[1] * d_body[0]) / denom if abs(denom) > 1e-6 else np.dot(w, unit),
                    max(0.0, fraction * length - PUSHER_HALF_WIDTH),
                    min(length, fraction * length + PUSHER_HALF_WIDTH),
                )
                effective = start + along * unit
                arm = rotation @ (effective - PRESSURE_CENTER)
                torque = np.linalg.det(np.stack((arm, direction)))
                lengths = NEAR_LENGTHS if baseline < 0.012 else FAR_LENGTHS
                predicted, used = min(
                    (
                        self._score(
                            *self._rollout(position, yaw, effective, direction, length_, c), rotation_only=rotate_first
                        ),
                        length_,
                    )
                    for length_ in lengths
                )
                push_distance = 2 * used if baseline < 0.012 else min(0.12, 1.5 * used)
                finish = contact + direction * push_distance
                if not (-0.33 <= finish[0] <= 0.11 and abs(finish[1]) <= 0.26):
                    continue
                end_position, end_yaw = self._rollout(position, yaw, effective, direction, used, c)
                outline = end_position + OUTLINE @ self._rot(end_yaw).T
                if np.any(np.abs(outline) > TABLE_HALF_XY_M - TABLE_MARGIN_M):
                    continue
                end_yaw_error = np.arctan2(
                    np.sin(self.env.target_yaw_rad - end_yaw), np.cos(self.env.target_yaw_rad - end_yaw)
                )
                if not rotate_first and abs(end_yaw_error) > max(TRANSLATE_MAX_YAW_RAD, abs(yaw_error)):
                    continue
                standoff = next(
                    (
                        d
                        for d in DESCEND_STANDOFFS_M
                        if self._footprint_clearance(contact - direction * d, direction, position, yaw)
                        >= FOOTPRINT_CLEARANCE_M
                    ),
                    None,
                )
                if standoff is None:
                    continue
                improvement = (rotation_baseline if rotate_first else baseline) - predicted
                if skew != 0.0:
                    improvement -= 0.0003
                travel = np.linalg.norm(pre - current_xy)
                value = improvement - 0.0015 * travel - (0.0005 if self.trace else 0)
                if value > best_value:
                    best_value = value
                    best = (pre, direction, push_distance, predicted, torque / c**2, standoff)
        return best

    def _run(self):
        env = self.env
        for attempt in range(45):
            if env.get_metrics()["task_rule_violation"]:
                self.abort_requested = True
                return
            if env.get_metrics()["inside_target"]:
                self.phase = "verify"
                for _ in range(12):
                    yield self._command(self._eef())
                metrics = env.get_metrics()
                if metrics["success"]["passed"] and metrics["inside_target"]:
                    self.verified = True
                    self.phase = "done"
                    return
            plan = self._plan()
            if plan is None:
                self.abort_requested = True
                return
            pre, direction, distance, predicted, yaw_rate, standoff = plan
            top, table_rotation = self._table()
            direction_world = table_rotation @ np.r_[direction, 0.0]
            down = -table_rotation[:, 2]
            reference = self.home_closing
            if abs(np.dot(direction_world, reference)) < 0.2:
                reference = self.env.sim.data.site_xmat[self.site].reshape(3, 3)[:, 0]
            closing = direction_world if np.dot(direction_world, reference) >= 0 else -direction_world
            self.orientation = np.column_stack((closing, np.cross(down, closing), down))
            above = self._world(np.r_[pre - direction * 0.040, TRANSIT_SITE_Z_M])
            low = self._world(np.r_[pre + direction * (PRE_STANDOFF_M - standoff), PUSH_SITE_Z_M])
            self.phase = "transit"
            if not (yield from self._move(above, speed=TRANSIT_SPEED_MPS, timeout=220, rotation_limit=0.5)):
                if not (yield from self._give_up_on(pre, above)):
                    return
                continue
            self.phase = "orient"
            if not (yield from self._orient()):
                self.abort_requested = True
                return
            self.phase = "descend"
            if not (yield from self._move(low, speed=DESCEND_SPEED_MPS, timeout=100)):
                if not (yield from self._give_up_on(pre, above)):
                    return
                continue
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
                baseline = self._score(before_position, before_yaw, rotation_only=self.rotating)
                start_pusher = self._pusher()
                start_eef = self._eef()
                best_score, stale, score = baseline, 0, baseline
                for index in range(int(distance / 0.001) + 120):
                    if np.dot(self._pusher() - start_pusher, direction_world) >= distance:
                        break
                    # Hold the pusher on the planned world line: lateral and height errors are corrected,
                    # progress along the line is limited to `lead` ahead of the current pusher.
                    lead = NEAR_LEAD_M if score < NEAR_ERROR_M else MID_LEAD_M if score < MID_ERROR_M else FAR_LEAD_M
                    if lead == FAR_LEAD_M and abs(yaw_rate) < FAST_MAX_YAW_RATE:
                        lead = FAST_LEAD_M
                    progress = np.dot(self._eef() - start_eef, direction_world)
                    yield self._command(start_eef + direction_world * (progress + lead), limit=max(0.0025, lead))
                    position, yaw = self._pose()
                    score = SCORE_SMOOTHING * score + (1 - SCORE_SMOOTHING) * self._score(
                        position, yaw, rotation_only=self.rotating
                    )
                    if score < best_score - 0.0001:
                        best_score, stale = score, 0
                    else:
                        stale += 1
                    moved = np.dot(position - before_position, direction)
                    turned = np.arctan2(np.sin(yaw - before_yaw), np.cos(yaw - before_yaw))
                    if (
                        (not self.rotating and self._inside_decal(position, yaw, INSIDE_SPARE_M))
                        or score > best_score + (0.0003 if best_score < 0.012 else 0.0015)
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
            if not (yield from self._move(self._eef() - direction_world * 0.004, speed=0.03, timeout=30)):
                self.abort_requested = True
                return
            # Back off and rise to transit height in one move (the T does not coast measurably, so no settle).
            clear = self._eef() - direction_world * 0.010
            clear[2] = above[2]
            if not (yield from self._move(clear, speed=0.16, timeout=60)):
                self.abort_requested = True
                return
        self.abort_requested = True
