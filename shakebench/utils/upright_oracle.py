"""Privileged, closed-loop oracle that stands the fallen Upright object on the static worktable.

The plans target the official fingertip contact (condim 4, 0.005 m torsional friction). A closed Panda
pinch then transmits at most about 0.1-0.2 N*m about its closing axis, which splits the current objects:

* ``wine_bottle`` (1.1 kg): no pinch can hold its gravity torque (>1 N*m), so the neck is pinched from
  above and lifted straight up. The bottle turns inside the pinch while its base slides under it on the
  low-friction table; once past its tipping point the bottle is released onto its base. (A pivot about the
  base rim does not work here: with table friction 0.25, millimetre tracking errors of the stiff OSC already
  push the base away.)
* ``mug``, ``boxed_drink`` and ``power_drill``: the pinch holds them rigidly, so they are lifted, turned about the
  closing axis (plus a yaw that keeps the final side approach inside the Panda workspace) and set down upright.
  The drill is held only by its grip, pinched front to back at the grip's top where the lever to the centre of
  mass (in the motor) is shortest; see ``RIGID_GRASPS``.

Every plan is checked against the arm's joint limits with inverse kinematics on scratch data before it
runs. ``verified`` is set only after the gripper has released and withdrawn and the environment's
released-and-still success has latched.
"""

import mujoco
import numpy as np

from robosuite.utils.control_utils import orientation_error
from shakebench.environments.upright import UPRIGHT_OBJECTS

# Delta-mode OSC (kp 150, critical damping) tracks a per-step delta d at about d / OSC_LAG_S metres per second.
OSC_LAG_S = 2.0 / np.sqrt(150.0)
OPEN, CLOSE = -1.0, 1.0
# Grip-site frame (z = approach): pad centres sit 3.6 mm behind the site, the palm 31 mm behind it.
PAD_CENTRE_BEHIND_SITE_M = 0.0036
PRE_GRASP_M = 0.05
TRANSIT_SPEED_MPS = 0.30
DESCEND_SPEED_MPS = 0.10
CARRY_SPEED_MPS = 0.12
RETREAT_SPEED_MPS = 0.15
RETREAT_TIMEOUT_STEPS = 60
# The finger targets ramp from open to closed in 10 steps; the pinch has its full force only after that.
CLOSE_STEPS = 10
OPEN_STEPS = 10
OPEN_CLEARANCE_M = 0.008  # retreat once the fingers are this much wider than the pinch
SETTLE_STEPS = 10  # at most, above the grasp before descending
SETTLE_SPEED_MPS = 0.02
SETTLE_TOLERANCE_M = 0.003
GRIP_SETTLE_STEPS = 3  # extra steps allowed for both pad contacts to appear after closing
FINGERS_ON_AIR_M = 0.005  # finger opening below which the pinch holds nothing (narrowest pinch: 25 mm neck)
GRIP_DRIFT_MAX_M = 0.015  # the recorded pinch point may drift this far from the pad centre

# Wine bottle, object frame: base disc at z = -0.128 m (r 33.9 mm), neck r ~12.5-13 mm for z in [0.072, 0.128].
# While the bottle turns inside the pinch, the rim of the neck top sweeps a radius sqrt(d^2 + 13^2) mm around
# the pinch (d: pinch to neck top), and the palm sits only 27.4 mm behind the pad centre. Pinching at
# z = 0.114 (d = 14 mm, sweep 19 mm) leaves room for ~5 mm of pinch slip along the neck and keeps the finger
# bodies on the neck.
BOTTLE_GRASP_Z_M = 0.114
BOTTLE_BASE_Z_M = -0.128
BOTTLE_BASE_RADIUS_M = 0.0339
# Stand-up lift: the base slides under the pinch (table mu 0.25) while the bottle turns inside it.
# Hand off to straightening at 80 deg, before OSC lag and the final fast rotation lift the base off the table.
LIFT_RATE_M = 0.004
LIFT_SLOW_RATE_M = 0.002  # above LIFT_SLOW_ELEVATION_RAD, so the base keeps sliding instead of lifting off
LIFT_SLOW_ELEVATION_RAD = np.radians(40.0)
LIFT_OVERSHOOT_M = 0.03  # allowance for the OSC sag under the bottle's weight
STAND_ELEVATION_RAD = np.radians(80.0)
STAND_MIN_ELEVATION_RAD = np.radians(78.0)
BASE_LIFTOFF_M = 0.003
# Straighten before letting go: hold the pinch above the base centre until the bottle is this close to vertical
# and this slow (a fast final fall can swing it ~10 deg past vertical).
STRAIGHT_TOLERANCE_RAD = np.radians(2.0)
STRAIGHT_SPIN_MAX_RAD_S = 0.3
STRAIGHT_STEPS = 20
LIFT_TIMEOUT_STEPS = 260
LIFT_STALL_STEPS = 40
# The standing bottle's base centre (under the final pinch) must end inside this table-frame box: on_table needs
# |x| < 0.28 and |y| < 0.25, and pointing-down reach at the 0.23 m final pinch height ends near table x = 0.0.
BOTTLE_STAND_BOX_T = ((-0.23, -0.04), (-0.18, 0.18))
# The standing pinch is already near the top of the pointing-down workspace. The open fingers straddle the neck
# along the closing axis, so the gripper may leave diagonally: up and back toward the robot at once.
BOTTLE_RISE_M = 0.05
BOTTLE_BACK_OFF_M = 0.05

# Rigidly held objects, object frame: pinch point and how far above it (world up) the grip site sits.
# Mug: the body axis runs through (0, 0.0144); the site 5 mm above it keeps the palm 2.6 mm above the
# widest body wall (33.4 mm) while the pads still reach the wall's equator, and keeps the fingertips ~7 mm
# above the handle, which hangs 28 deg below horizontal on one side. z = 0.012 puts the pinch 20 mm from the
# centre of mass (0.06 N*m) while the upright palm still clears the table by ~17 mm.
# Boxed drink: pads on the 33.6 mm wide faces, next to the wider top section.
RIGID_GRASPS = {
    "mug": {"point": (0.0, 0.0144, 0.012), "site_rise_m": 0.005, "early_close_m": 0.0},
    "boxed_drink": {"point": (0.0, 0.0, 0.020), "site_rise_m": 0.0, "early_close_m": 0.012},
    "power_drill": {
        "point": (0.007, -0.027, 0.020),
        "site_rise_m": 0.005,
        "early_close_m": 0.012,
        "slide_out_m": 0.055,
    },
}
# Power drill (0.895 kg, object z from battery to motor, x front-back, y side to side): its centre of mass sits in
# the motor, at z = 0.041 above the whole grip. The upright turn is about the pinch's closing axis (x), where the
# pinch passes only its ~0.16 N*m torsional friction (plus the box contacts' spread), so the pinch sits at the top
# of the grip, 6.5 mm below the motor: the lying drill then loads it with 0.18 N*m instead of the 0.47 N*m of a
# grip-centre pinch, which lets the drill swing battery-first into the palm. The table cannot take that torque
# instead: every grip point lies between the battery-edge pivot and the centre of mass. The motor overhangs the
# grip on the palm side, so the open fingers slide out horizontally (slide_out_m) before rising.
# early_close_m: start closing this far above the grasp point. The fingers need ~7 steps to travel from fully
# open to the 34 mm box, longer than the last centimetre of the descent takes; the box's flat faces do not care
# where the pads land. Round pinches (neck, mug body) close only at the grasp point, on their equator.
# Transfer: lift straight off the table, then lift, turn and carry in one motion to just above the placement.
# The turn is limited to TRANSFER_RATE_RAD per step (0.9 rad/s); a parabolic height bump, the smallest in
# TRANSFER_BUMPS_M that keeps every object vertex and the gripper hull TABLE_CLEARANCE_M above the table,
# makes room for the turn.
TRANSFER_LIFT_M = 0.03
TRANSFER_RATE_RAD = 0.045
TRANSFER_BUMPS_M = (0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12)
TABLE_CLEARANCE_M = 0.012
TRANSFER_SAMPLES = 21
# Grip-site frame corners of the Panda hand and fingertips (fingers at most 50 mm off the closing centre).
GRIPPER_HULL = np.array(
    [[x, y, z] for x in (-0.1, 0.1) for y in (-0.0316, 0.0316) for z in (-0.123, -0.031)]
    + [[x, y, 0.0092] for x in (-0.05, 0.05) for y in (-0.0105, 0.0105)]
)
PLACE_ABOVE_M = 0.015
PLACE_SPEED_MPS = 0.04
# Touch down slowly: a 4 cm/s contact can leave the light box rocking on its base for seconds.
PLACE_SLOW_GAP_M = 0.008
PLACE_SLOW_SPEED_MPS = 0.015
PLACE_GAP_M = 0.002  # release once the upright bottom face is this close to the table
LEVEL_STEP_RAD = 0.02  # per-step correction of the grip target toward the measured object axis
LEVEL_STEPS = 40
RETREAT_M = 0.05
RISE_M = 0.04
# Final side approaches, measured from the table x axis (horizontal), and the minimum joint margin to accept.
APPROACH_YAWS_RAD = np.radians(np.arange(-180.0, 180.0, 15.0))
IK_MARGIN_MIN_RAD = 0.08
APPROACH_TURN_WEIGHT = 0.6
BRANCH_JUMP_RAD = 0.5  # largest joint change between consecutive densified poses
VERIFY_STEPS = 60


def _unit(vector):
    vector = np.asarray(vector, dtype=float)
    return vector / np.linalg.norm(vector)


def _frame(closing, approach):
    """Grip-site rotation with x along the closing axis and z along the approach."""
    approach = _unit(approach)
    closing = _unit(closing - approach * np.dot(closing, approach))
    return np.column_stack((closing, np.cross(approach, closing), approach))


def _rotation(axis, angle):
    """Rodrigues rotation matrix about a unit axis."""
    x, y, z = _unit(axis)
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * skew @ skew


def _rotvec(rotation):
    """World-frame rotation vector (axis * angle, angle in [0, pi]) of a rotation matrix.

    robosuite's ``orientation_error`` is only sin(angle) * axis, which saturates at 90 degrees.
    """
    angle = float(np.arccos(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)))
    if angle < 1e-9:
        return np.zeros(3)
    if angle > np.pi - 1e-4:
        axis = _unit(np.sqrt(np.clip((np.diag(rotation) + 1.0) / 2.0, 0.0, None)))
        # Recover the relative signs from the symmetric part.
        pivot = int(np.argmax(np.abs(axis)))
        axis = _unit(rotation[:, pivot] + np.eye(3)[pivot])
        return axis * angle
    return (
        angle
        / (2.0 * np.sin(angle))
        * np.array([rotation[2, 1] - rotation[1, 2], rotation[0, 2] - rotation[2, 0], rotation[1, 0] - rotation[0, 1]])
    )


def _turn_angle(start, end):
    return float(np.linalg.norm(_rotvec(end @ start.T)))


def _slerp(start, end, fraction):
    """Rotation matrix a fraction of the way from start to end along the shortest arc."""
    delta = _rotvec(end @ start.T)
    angle = np.linalg.norm(delta)
    if angle < 1e-9:
        return end.copy()
    return _rotation(delta / angle, fraction * angle) @ start


class _ArmKinematics:
    """Damped least-squares inverse kinematics of the grip site on scratch MuJoCo data."""

    def __init__(self, env):
        robot = env.robots[0]
        self.model = env.sim.model._model
        self.data = mujoco.MjData(self.model)
        self.data.qpos[:] = env.sim.data._data.qpos
        self.site = robot.eef_site_id["right"]
        self.qpos = np.array(robot._ref_joint_pos_indexes)
        self.dofs = np.array(robot._ref_joint_vel_indexes)
        joints = [self.model.joint(name).id for name in robot.robot_model.joints]
        self.lower, self.upper = self.model.jnt_range[joints].T
        self.rest = env.sim.data._data.qpos[self.qpos].copy()
        data = env.sim.data._data
        self.start = (data.site_xpos[self.site].copy(), data.site_xmat[self.site].reshape(3, 3).copy())

    def solve(self, position, rotation, start=None, iterations=150):
        """Return (joints, converged) for one site pose, warm-started from ``start``."""
        q = (self.rest if start is None else start).copy()
        jacp, jacr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        for _ in range(iterations):
            self.data.qpos[self.qpos] = q
            mujoco.mj_kinematics(self.model, self.data)
            mujoco.mj_comPos(self.model, self.data)
            error = np.r_[
                position - self.data.site_xpos[self.site],
                orientation_error(rotation, self.data.site_xmat[self.site].reshape(3, 3)),
            ]
            if np.linalg.norm(error[:3]) < 3e-4 and np.linalg.norm(error[3:]) < 3e-3:
                return q, True
            mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.site)
            jacobian = np.r_[jacp[:, self.dofs], jacr[:, self.dofs]]
            step = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + 1e-4 * np.eye(6), error)
            # The OSC nullspace pulls toward the start joints as well.
            step += (np.eye(7) - np.linalg.pinv(jacobian) @ jacobian) @ (0.05 * (self.rest - q))
            q = np.clip(q + np.clip(step, -0.2, 0.2), self.lower + 1e-3, self.upper - 1e-3)
        return q, False

    def path_margin(self, poses):
        """Smallest joint-limit margin while tracking poses from the current site pose; -inf if infeasible.

        The path is densified (3 cm / 0.15 rad) and solved by continuation from the current joints, so a
        solution on another arm branch, which the OSC cannot reach continuously, counts as a failure.
        """
        q, margin = self.rest.copy(), np.inf
        position, rotation = self.start
        for target_position, target_rotation in poses:
            distance = np.linalg.norm(target_position - position)
            angle = _turn_angle(rotation, target_rotation)
            count = max(1, int(np.ceil(max(distance / 0.03, angle / 0.15))))
            for index in range(1, count + 1):
                fraction = index / count
                solution, converged = self.solve(
                    position + fraction * (target_position - position), _slerp(rotation, target_rotation, fraction), q
                )
                if not converged or np.max(np.abs(solution - q)) > BRANCH_JUMP_RAD:
                    return -np.inf
                q = solution
                margin = min(margin, float(np.min(np.minimum(q - self.lower, self.upper - q))))
            position, rotation = target_position, target_rotation
        return margin


class UprightOracle:
    """Stand the selected fallen object up from privileged object state."""

    def __init__(self, env):
        if env.task_state["task"]["task_type"] != "upright":
            raise ValueError("UprightOracle requires an upright task state")
        if env.physics_profile.status != "official_immutable":
            raise ValueError("UprightOracle requires the official physics profile")
        self.env = env
        self.object_id = env.task_state["task"]["object_id"]
        self.spec = UPRIGHT_OBJECTS[self.object_id]
        self.abort_requested = False
        self.verified = False
        self.failure_reason = None
        self.phase = "idle"
        self.trace = []
        self.site = env.robots[0].eef_site_id["right"]
        self.controller = env.robots[0].part_controllers["right"]
        self.worktable = env.sim.model.body_name2id(env.worktable_body_name)
        self.pads = {env.sim.model.geom_name2id(f"gripper0_right_finger{index}_pad_collision") for index in (1, 2)}
        self.base = env.sim.data.xpos[env.sim.model.body_name2id(env.robots[0].robot_model.root_body)].copy()
        self.orientation = env.sim.data.site_xmat[self.site].reshape(3, 3).copy()
        self.grip = OPEN
        self.steps = 0
        self.control_dt = 1.0 / env.control_freq
        self._grip_local = np.zeros(3)
        self._grip_width = 0.0
        self._closed_at = 0
        self._points = self._object_points()
        self._hold_target = self._eef()
        self._actions = self._run()

    def action(self):
        """Next 7-D OSC action; holds the last target once the plan has finished or aborted."""
        self.steps += 1
        if self.phase not in ("done", "aborted"):
            action = next(self._actions, None)
            if action is not None:
                return action
            if self.phase not in ("done", "aborted"):
                self._stop("plan_exhausted")
        return self._command(self._hold_target)

    # --- state -------------------------------------------------------------------------------------------

    def _eef(self):
        return self.env.sim.data.site_xpos[self.site].copy()

    def _eef_rotation(self):
        return self.env.sim.data.site_xmat[self.site].reshape(3, 3).copy()

    def _table(self):
        data = self.env.sim.data._data
        rotation = data.xmat[self.worktable].reshape(3, 3).copy()
        return data.xpos[self.worktable] + rotation @ [0, 0, self.env.arena.table_half_size[2]], rotation

    def _object(self):
        data = self.env.sim.data._data
        body = self.env.object_body_id
        return data.xpos[body].copy(), data.xmat[body].reshape(3, 3).copy()

    def _touched(self):
        """Gripper geoms in contact with the object."""
        data = self.env.sim.data._data
        touched = set()
        for contact in data.contact[: data.ncon]:
            if contact.dist > 0.001:
                continue
            if contact.geom1 in self.env.object_geom_ids and contact.geom2 in self.env.gripper_geom_ids:
                touched.add(contact.geom2)
            elif contact.geom2 in self.env.object_geom_ids and contact.geom1 in self.env.gripper_geom_ids:
                touched.add(contact.geom1)
        return touched

    def _pinched(self):
        return self.pads <= self._touched()

    def _object_points(self):
        """Points on the object's collision geometry, in the object body frame.

        Mesh geoms contribute their vertices; box, capsule, cylinder and sphere geoms (the drill's primitives)
        contribute their corners or points sampled around their end caps. Only when the object has no collision
        geoms of these types does the body's visual mesh stand in.
        """
        model, data = self.env.sim.model._model, self.env.sim.data._data
        body = self.env.object_body_id
        rotation, origin = data.xmat[body].reshape(3, 3), data.xpos[body]
        types = mujoco.mjtGeom
        sampled = (
            types.mjGEOM_MESH,
            types.mjGEOM_BOX,
            types.mjGEOM_CAPSULE,
            types.mjGEOM_CYLINDER,
            types.mjGEOM_SPHERE,
        )
        geoms = [geom for geom in sorted(self.env.object_geom_ids) if model.geom_type[geom] in sampled]
        if not geoms:
            geoms = [
                geom
                for geom in range(model.ngeom)
                if model.geom_bodyid[geom] == body and model.geom_type[geom] == types.mjGEOM_MESH
            ]
        ring = np.array([[np.cos(a), np.sin(a), 0.0] for a in np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)])
        points = []
        for geom in geoms:
            kind, size = model.geom_type[geom], model.geom_size[geom]
            if kind == types.mjGEOM_MESH:
                mesh = model.geom_dataid[geom]
                start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
                local = model.mesh_vert[start : start + count]
            elif kind == types.mjGEOM_BOX:
                local = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * size
            elif kind == types.mjGEOM_SPHERE:
                local = np.vstack((ring, ring[:, [2, 0, 1]], ring[:, [1, 2, 0]])) * size[0]
            else:  # capsule or cylinder along the geom z axis, half-length size[1]
                caps = [ring * size[0] + [0.0, 0.0, end * size[1]] for end in (-1.0, 1.0)]
                if kind == types.mjGEOM_CAPSULE:
                    caps += [[0.0, 0.0, end * (size[1] + size[0])] for end in (-1.0, 1.0)]
                local = np.vstack(caps)
            world = local @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
            points.append((world - origin) @ rotation)
        return np.vstack(points)

    def _finger_opening(self):
        robot = self.env.robots[0]
        fingers = self.env.sim.data._data.qpos[robot._ref_gripper_joint_pos_indexes["right"]]
        return float(fingers[0] - fingers[1])

    def _close(self):
        if self.grip != CLOSE:
            self.grip = CLOSE
            self._closed_at = self.steps

    def _pad_centre(self):
        return self._eef() - PAD_CENTRE_BEHIND_SITE_M * self._eef_rotation()[:, 2]

    def _holding(self):
        """Geometric grasp check: fingers still on the object and the pinch point still between the pads.

        The stiff official pad contacts are single points that drop out of the contact list for a step or two
        while an object turns inside the pinch, so contact presence alone is not a usable signal. The pinch
        point is recorded in the object frame when the grasp closes; it lies on the in-hand turning axis.
        """
        if self._finger_opening() < FINGERS_ON_AIR_M:
            return False
        centre, rotation = self._object()
        return bool(np.linalg.norm(centre + rotation @ self._grip_local - self._pad_centre()) < GRIP_DRIFT_MAX_M)

    def _up_cosine(self):
        _, rotation = self._object()
        return float(np.dot(self._table()[1][:, 2], rotation[:, 2]))

    # --- control -----------------------------------------------------------------------------------------

    def _command(self, target, limit=0.01, rotation_limit=0.3):
        """Normalized OSC delta toward a world position and ``self.orientation``."""
        self._hold_target = np.asarray(target, dtype=float).copy()
        delta = np.clip(self._hold_target - self._eef(), -limit, limit)
        rotation = np.clip(_rotvec(self.orientation @ self._eef_rotation().T), -0.5, 0.5)
        if self.controller.input_ref_frame == "base":
            delta = self.controller.origin_ori.T @ delta
            rotation = self.controller.origin_ori.T @ rotation
        return np.r_[
            np.clip(delta / self.controller.output_max[:3], -1, 1),
            np.clip(rotation / self.controller.output_max[3:], -rotation_limit, rotation_limit),
            self.grip,
        ]

    def _orientation_error(self):
        return _turn_angle(self._eef_rotation(), self.orientation)

    def _move(self, target, *, speed, tolerance=0.003, timeout=200, rotation_limit=0.3, close_within=0.0):
        """Move the site toward a world point at a bounded speed; False on a stall or timeout.

        With ``close_within`` the fingers start closing once the site is that close to the target.
        """
        target = np.asarray(target, dtype=float)
        previous, stalled = self._eef(), 0
        for _ in range(timeout):
            current = self._eef()
            delta = target - current
            distance = np.linalg.norm(delta)
            if distance < tolerance and self._orientation_error() < 0.05:
                return True
            if distance < close_within:
                self._close()
            step = min(distance, speed * OSC_LAG_S)
            waypoint = current + (delta * step / distance if distance > 1e-9 else 0.0)
            yield self._command(waypoint, limit=max(0.004, step), rotation_limit=rotation_limit)
            stalled = stalled + 1 if np.linalg.norm(self._eef() - previous) < 5e-5 else 0
            if stalled >= 25:
                return False
            previous = self._eef()
        return False

    def _hold(self, target, steps):
        for _ in range(steps):
            yield self._command(target)

    def _level(self):
        """Tilt the grip target so the held object's measured axis points up (undo in-hand rotation)."""
        up = self._table()[1][:, 2]
        axis = self._object()[1][:, 2]
        correction = np.cross(axis, up)
        norm = np.linalg.norm(correction)
        if norm > 1e-6:
            angle = min(np.arctan2(norm, np.dot(axis, up)), LEVEL_STEP_RAD)
            self.orientation = _rotation(correction / norm, angle) @ self.orientation

    def _stop(self, reason):
        self.failure_reason = reason
        self.abort_requested = True
        self.phase = "aborted"
        self.trace.append({"event": "abort", "reason": reason, "step": self.steps})

    def _log(self, event, **values):
        self.trace.append(
            {"event": event, "step": self.steps, **{key: _json_value(value) for key, value in values.items()}}
        )

    # --- shared grasp, release and verification ------------------------------------------------------------

    def _grasp(self, site, early_close=0.0):
        """Approach along the planned approach axis, descend with open fingers and pinch."""
        approach = self.orientation[:, 2]
        self.phase = "approach"
        pre_grasp = site - PRE_GRASP_M * approach
        if not (
            yield from self._move(pre_grasp, speed=TRANSIT_SPEED_MPS, tolerance=0.006, timeout=250, rotation_limit=0.6)
        ):
            self._stop("approach_stalled")
            return False
        # The approach ends inside a loose tolerance while still moving; settle before descending so the residual
        # speed does not carry the fingers sideways onto a neighbouring part (the drill's motor is 6.5 mm away).
        previous = self._eef()
        for _ in range(SETTLE_STEPS):
            yield self._command(pre_grasp, limit=0.01)
            speed = np.linalg.norm(self._eef() - previous) / self.control_dt
            previous = self._eef()
            if speed < SETTLE_SPEED_MPS and np.linalg.norm(self._eef() - pre_grasp) < SETTLE_TOLERANCE_M:
                break
        self.phase = "descend"
        if not (yield from self._move(site, speed=DESCEND_SPEED_MPS, tolerance=0.002, close_within=early_close)):
            self._stop("descend_blocked")
            return False
        self.phase = "grasp"
        self._close()
        for _ in range(CLOSE_STEPS + GRIP_SETTLE_STEPS):
            if self.steps - self._closed_at >= CLOSE_STEPS and self._pinched():
                centre, rotation = self._object()
                self._grip_local = rotation.T @ (self._pad_centre() - centre)
                self._grip_width = self._finger_opening()
                return True
            yield self._command(site)
        self._stop("grasp_missed")
        return False

    def _retreat_leg(self, displacement):
        """Withdraw by a world displacement; legs may end a little short at the workspace boundary."""
        start, goal = self._eef(), self._eef() + displacement
        length2 = float(np.dot(displacement, displacement))
        previous, slow = self._eef(), 0
        for _ in range(RETREAT_TIMEOUT_STEPS):
            current = self._eef()
            delta = goal - current
            distance = np.linalg.norm(delta)
            travelled = float(np.dot(current - start, displacement)) / length2
            slow = slow + 1 if np.linalg.norm(current - previous) / self.control_dt < 0.01 else 0
            if distance < 0.005 or (travelled >= 0.7 and slow >= 3):
                break
            previous = current
            step = min(distance, RETREAT_SPEED_MPS * OSC_LAG_S)
            yield self._command(current + delta * step / distance, limit=max(0.004, step))
        travelled = float(np.dot(self._eef() - start, displacement)) / length2
        return travelled >= 0.7 and not self._touched()

    def _release(self, retreat):
        """Open, withdraw along the given world displacements, and wait for the success latch."""
        self.phase = "release"
        self.grip = OPEN
        hold = self._eef()
        for _ in range(OPEN_STEPS):
            if self._finger_opening() >= self._grip_width + OPEN_CLEARANCE_M:
                break
            yield self._command(hold)
        self.phase = "retreat"
        for displacement in retreat:
            if not (yield from self._retreat_leg(displacement)):
                self._stop("retreat_stalled")
                return
            if self.env.get_metrics()["success"]["passed"]:
                break  # already released, still and upright: the remaining clearance legs add nothing
        self.phase = "verify"
        hold = self._eef()
        for _ in range(VERIFY_STEPS):
            metrics = self.env.get_metrics()
            if metrics["success"]["passed"]:
                self.verified = True
                self._log("verified", up_cosine=metrics["up_cosine"])
                self.phase = "done"
                return
            yield self._command(hold)
        metrics = self.env.get_metrics()
        self._stop(
            "not_upright_after_release"
            if metrics["up_cosine"] < 0.95
            else "not_released" if metrics["touching_gripper"] else "success_not_latched"
        )

    # --- wine bottle: neck pinch and stand-up lift -----------------------------------------------------------

    def _bottle_axis(self):
        """Horizontal bottle axis (base toward neck) and the horizontal axis normal to it, in world coordinates."""
        _, rotation = self._object()
        up = self._table()[1][:, 2]
        axis = _unit(rotation[:, 2] - up * np.dot(rotation[:, 2], up))
        return axis, _unit(np.cross(axis, up))

    def _elevation(self, axis, up):
        """Bottle axis elevation above the table, measured in the vertical plane of the lying bottle."""
        bottle = self._object()[1][:, 2]
        return float(np.arctan2(np.dot(bottle, up), np.dot(bottle, axis)))

    def _base_gap(self, top, up):
        """Height of the bottle's lowest base-rim point above the table."""
        centre, rotation = self._object()
        bottle = rotation[:, 2]
        tilt = up - bottle * np.dot(up, bottle)
        rim = -tilt / np.linalg.norm(tilt) if np.linalg.norm(tilt) > 1e-6 else np.zeros(3)
        return float(np.dot(centre + BOTTLE_BASE_Z_M * bottle + BOTTLE_BASE_RADIUS_M * rim - top, up))

    def _toward_robot(self, point):
        """Horizontal unit vector from a world point toward the robot base."""
        up = self._table()[1][:, 2]
        offset = self.base - point
        return _unit(offset - up * np.dot(offset, up))

    def _plan_bottle(self, kinematics):
        """Pointing-down neck pinch and stand point, with the closing-axis sign that keeps the best joint margin."""
        top, table = self._table()
        up = table[:, 2]
        centre, rotation = self._object()
        axis, turn = self._bottle_axis()
        site = centre + BOTTLE_GRASP_Z_M * rotation[:, 2] - PAD_CENTRE_BEHIND_SITE_M * up
        # The bottle ends standing under the final pinch: keep that point inside the safe table box.
        local = table.T @ (site - top)
        stand = np.array([np.clip(local[index], *BOTTLE_STAND_BOX_T[index]) for index in (0, 1)])
        stand_site = top + table @ np.r_[stand, abs(BOTTLE_BASE_Z_M) + BOTTLE_GRASP_Z_M - PAD_CENTRE_BEHIND_SITE_M]
        best = None
        for sign in (1.0, -1.0):
            orientation = _frame(sign * turn, -up)
            poses = [
                (site - PRE_GRASP_M * orientation[:, 2], orientation),
                (site, orientation),
                (0.5 * (site + stand_site), orientation),
                (stand_site + LIFT_OVERSHOOT_M * up, orientation),
                (stand_site + BOTTLE_RISE_M * up + BOTTLE_BACK_OFF_M * self._toward_robot(stand_site), orientation),
            ]
            margin = kinematics.path_margin(poses)
            # Among feasible signs, prefer the smaller wrist turn from the current pose (the approach's slowest part).
            cost = _turn_angle(kinematics.start[1], orientation)
            if margin >= IK_MARGIN_MIN_RAD and (best is None or cost < best[0]):
                best = (cost, margin, orientation)
        if best is None:
            return None
        return {"site": site, "stand_site": stand_site, "orientation": best[2], "axis": axis, "margin_rad": best[1]}

    def _bottle(self):
        plan = self._plan_bottle(_ArmKinematics(self.env))
        if plan is None:
            self._stop("no_reachable_bottle_plan")
            return
        shift = plan["stand_site"] - plan["site"]
        self._log("plan", strategy="neck_lift", shift_m=float(np.hypot(*shift[:2])), margin_rad=plan["margin_rad"])
        self.orientation = plan["orientation"]
        if not (yield from self._grasp(plan["site"])):
            return
        self.phase = "lift"
        top, table = self._table()
        up, axis = table[:, 2], plan["axis"]
        start = self._eef()
        start_height = float(np.dot(start - top, up))
        stand_height = float(np.dot(plan["stand_site"] - top, up))
        horizontal = (plan["stand_site"] - start) - up * np.dot(plan["stand_site"] - start, up)
        height, best, stale = start_height, -np.inf, 0
        for _ in range(LIFT_TIMEOUT_STEPS):
            elevation = self._elevation(axis, up)
            gap = self._base_gap(top, up)
            if elevation >= STAND_ELEVATION_RAD or (gap > BASE_LIFTOFF_M and elevation >= STAND_MIN_ELEVATION_RAD):
                break
            if not self._holding():
                if elevation >= STAND_MIN_ELEVATION_RAD:
                    break  # past the tipping point: the bottle falls onto its base by itself
                self._stop("lift_lost_grasp")
                return
            if gap > BASE_LIFTOFF_M:
                # The base left the table before the bottle turned upright: set it back down to slide on.
                height = min(height, float(np.dot(self._eef() - top, up))) - (gap - BASE_LIFTOFF_M)
            else:
                rate = LIFT_RATE_M if elevation < LIFT_SLOW_ELEVATION_RAD else LIFT_SLOW_RATE_M
                height = min(height + rate, stand_height + LIFT_OVERSHOOT_M)
            fraction = np.clip((height - start_height) / (stand_height - start_height), 0.0, 1.0)
            target = start + fraction * horizontal + (height - start_height) * up
            yield self._command(target, limit=0.02)
            best, stale = (elevation, 0) if elevation > best + np.radians(0.2) else (best, stale + 1)
            if stale >= LIFT_STALL_STEPS:
                self._stop("lift_stalled")
                return
        else:
            self._stop("lift_timeout")
            return
        self.phase = "settle"
        standing = abs(BOTTLE_BASE_Z_M) + BOTTLE_GRASP_Z_M - PAD_CENTRE_BEHIND_SITE_M
        for _ in range(STRAIGHT_STEPS):
            centre, rotation = self._object()
            spin = np.linalg.norm(self.env.sim.data._data.cvel[self.env.object_body_id][:3])
            if abs(self._elevation(axis, up) - np.pi / 2) < STRAIGHT_TOLERANCE_RAD and spin < STRAIGHT_SPIN_MAX_RAD_S:
                break
            base = centre + BOTTLE_BASE_Z_M * rotation[:, 2]
            base = base - up * np.dot(base - top, up)
            yield self._command(base + standing * up, limit=0.01)
        centre, rotation = self._object()
        self._log(
            "stood",
            elevation_rad=self._elevation(axis, up),
            base_gap_m=self._base_gap(top, up),
            base_xy_t=(table.T @ (centre + BOTTLE_BASE_Z_M * rotation[:, 2] - top))[:2],
        )
        yield from self._release([BOTTLE_RISE_M * up + BOTTLE_BACK_OFF_M * self._toward_robot(self._eef())])

    # --- mug and boxed drink: rigid lift, turn and place ----------------------------------------------------

    @staticmethod
    def _transfer_pose(lift, above, start, end, bump, fraction, up):
        """Site pose a fraction of the way along the lift-turn-carry transfer."""
        position = lift + fraction * (above - lift) + 4.0 * fraction * (1.0 - fraction) * bump * up
        return position, _slerp(start, end, fraction)

    def _transfer_bump(self, site, start, end, lift, above, top, up):
        """Smallest height bump that keeps the held object and the gripper clear of the table, or None."""
        centre, rotation = self._object()
        # Object vertices and gripper hull in grip-site coordinates at the planned grasp.
        held = np.vstack(((centre + self._points @ rotation.T - site) @ start, GRIPPER_HULL))
        for bump in TRANSFER_BUMPS_M:
            for fraction in np.linspace(0.0, 1.0, TRANSFER_SAMPLES):
                position, orientation = self._transfer_pose(lift, above, start, end, bump, fraction, up)
                if np.min((position + held @ orientation.T - top) @ up) < TABLE_CLEARANCE_M:
                    break
            else:
                return bump
        return None

    def _plan_rigid(self, kinematics):
        """Top-down pinch and lift-turn-carry transfer whose whole arm path stays clear of the joint limits."""
        top, table = self._table()
        up = table[:, 2]
        centre, rotation = self._object()
        grasp = RIGID_GRASPS[self.object_id]
        site = centre + rotation @ np.asarray(grasp["point"]) + grasp["site_rise_m"] * up
        axis = _unit(rotation[:, 2] - up * np.dot(rotation[:, 2], up))
        turn = _unit(np.cross(axis, up))
        # Upright: the object z axis goes to the table normal; the final side approach is then the lying axis.
        upright = _rotation(turn, np.arccos(np.clip(np.dot(rotation[:, 2], up), -1.0, 1.0)))
        placed = top + table @ np.r_[(table.T @ (centre - top))[:2], -self.spec["upright_lower_z_m"] + 0.001]
        lift = site + TRANSFER_LIFT_M * up
        candidates = []
        for sign in (1.0, -1.0):
            start = _frame(sign * turn, -up)
            for yaw in APPROACH_YAWS_RAD:
                approach = table @ [np.cos(yaw), np.sin(yaw), 0.0]
                twist = np.arctan2(np.dot(np.cross(axis, approach), up), np.dot(axis, approach))
                motion = _rotation(up, twist) @ upright
                end = motion @ start
                final = placed + motion @ (site - centre)
                angle = _turn_angle(start, end)
                # Rank by rotation time: in-air turn plus the approach's wrist turn (which runs ~1.7x faster).
                cost = angle + APPROACH_TURN_WEIGHT * _turn_angle(kinematics.start[1], start)
                candidates.append((cost, angle, start, end, final, approach))
        # Full path checks in order of increasing cost; keep the best margin among the first few feasible.
        candidates.sort(key=lambda item: item[0])
        feasible = []
        for cost, angle, start, end, final, approach in candidates:
            if feasible and (cost > feasible[0][1] + 0.5 or len(feasible) >= 3):
                break
            above = final + PLACE_ABOVE_M * up
            bump = self._transfer_bump(site, start, end, lift, above, top, up)
            if bump is None:
                continue
            slide = grasp.get("slide_out_m")
            retreat = [-slide * approach, RISE_M * up] if slide else [-RETREAT_M * approach + RISE_M * up]
            poses = [(site - PRE_GRASP_M * start[:, 2], start), (site, start), (lift, start)]
            poses += [self._transfer_pose(lift, above, start, end, bump, f, up) for f in (0.25, 0.5, 0.75, 1.0)]
            poses += [(final, end)] + [(final + sum(retreat[: index + 1]), end) for index in range(len(retreat))]
            margin = kinematics.path_margin(poses)
            if margin >= IK_MARGIN_MIN_RAD:
                plan = {
                    "site": site,
                    "start": start,
                    "end": end,
                    "lift": lift,
                    "above": above,
                    "bump": bump,
                    "retreat": retreat,
                    "turn_rad": float(angle),
                    "margin_rad": margin,
                }
                feasible.append((cost - 0.5 * min(margin, 0.6), cost, plan))
        return min(feasible, key=lambda item: item[0])[2] if feasible else None

    def _rigid(self):
        plan = self._plan_rigid(_ArmKinematics(self.env))
        if plan is None:
            self._stop("no_reachable_turn_plan")
            return
        self._log(
            "plan",
            strategy="lift_turn_place",
            turn_rad=plan["turn_rad"],
            bump_m=plan["bump"],
            margin_rad=plan["margin_rad"],
        )
        self.orientation = plan["start"]
        if not (yield from self._grasp(plan["site"], RIGID_GRASPS[self.object_id]["early_close_m"])):
            return
        self.phase = "lift"
        if not (yield from self._move(plan["lift"], speed=CARRY_SPEED_MPS, tolerance=0.008, timeout=60)):
            self._stop("lift_stalled")
            return
        self.phase = "transfer"
        up = self._table()[1][:, 2]
        path = [plan["lift"], plan["above"], plan["start"], plan["end"], plan["bump"]]
        samples = [self._transfer_pose(*path, f, up)[0] for f in np.linspace(0.0, 1.0, TRANSFER_SAMPLES)]
        length = float(np.sum(np.linalg.norm(np.diff(samples, axis=0), axis=1)))
        count = int(np.ceil(max(length / (CARRY_SPEED_MPS * self.control_dt), plan["turn_rad"] / TRANSFER_RATE_RAD)))
        for index in range(1, count + 1):
            position, self.orientation = self._transfer_pose(*path, index / count, up)
            yield self._command(position, limit=0.03, rotation_limit=0.5)
            if not self._holding():
                self._stop("object_slipped_in_transfer")
                return
        # Finish horizontal tracking while levelling out residual in-hand rotation.
        for _ in range(LEVEL_STEPS):
            # Real grasp drift can land the base before the planned above pose; do not lift it again.
            if self._up_cosine() >= 0.995 and self.env.get_metrics()["touching_table"]:
                break
            # Level at the measured height instead of pulling an upright object back up to the planned hover.
            target = plan["above"] + min(0.0, np.dot(self._eef() - plan["above"], up)) * up
            settled = np.linalg.norm(self._eef() - target) < 0.006 and self._orientation_error() < 0.04
            if settled and self._up_cosine() > 0.9995:
                break
            self._level()
            yield self._command(target, limit=0.03, rotation_limit=0.5)
        if not self._holding() or self._up_cosine() < 0.97:
            self._stop("object_slipped_in_turn")
            return
        self.phase = "place"
        top, _ = self._table()
        for _ in range(80):
            gap = self._bottom_gap(top, up)
            if gap <= PLACE_GAP_M or self.env.get_metrics()["touching_table"]:
                break
            self._level()
            speed = PLACE_SPEED_MPS if gap > PLACE_SLOW_GAP_M else PLACE_SLOW_SPEED_MPS
            yield self._command(self._eef() - speed * OSC_LAG_S * up, limit=0.008)
        else:
            self._stop("place_timeout")
            return
        self._log("placed", up_cosine=self._up_cosine())
        yield from self._release(plan["retreat"])

    def _bottom_gap(self, top, up):
        centre, rotation = self._object()
        return float(np.dot(centre + self.spec["upright_lower_z_m"] * rotation[:, 2] - top, up))

    # --- episode ---------------------------------------------------------------------------------------------

    def _run(self):
        if self.object_id == "wine_bottle":
            yield from self._bottle()
        else:
            yield from self._rigid()


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value
