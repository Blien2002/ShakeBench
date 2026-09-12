"""Device-resident ShakeBench substeps; CPU oracle and canonical IMU delivery.

This is a collection backend, not a scoreable replacement for classic MuJoCo.
Requires the versions in requirements-gpu.txt. Private MJWarp kernel helpers are
intentionally version-pinned rather than reimplementing contact force decoding.
"""

import copy
import hashlib

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp
from mujoco_warp._src.support import contact_force_fn
from mujoco_warp._src.types import vec5
from scipy.spatial.transform import Rotation

from robosuite.controllers.parts.arm.osc_warp import (
    Kinematics, PandaOSC, control, point_acceleration, point_velocity, set_goals,
)
from robosuite.utils import transform_utils as T
from robosuite.utils.shakebench_metrics import DEFAULT_SUCCESS_THRESHOLDS
from robosuite.utils.shakebench_providers import (
    COMMON_STATE_KEYS, POLICY_FIELD_CONTRACT, RigidBodyState, SupportState,
    relative_pose_twist_acceleration,
)
from robosuite.utils.shakebench_rotations import wxyz_to_matrix

wp.set_module_options({"enable_backward": False})


@wp.struct
class Evaluation:
    bodies: wp.array[int]  # robot base, gripper body, object, worktable, deck
    finger_geoms: wp.array[int]
    sensor_addresses: wp.array[int]
    object_geoms: wp.array[int]
    bottom_geoms: wp.array[int]
    finger_mask: wp.array[int]
    points: wp.array[wp.vec3]
    target_origin: wp.vec3
    target_half: wp.vec2
    thresholds: wp.array[float]
    aggregate: wp.array2d[float]  # penetration, bottom force, bottom contact, finger contact
    candidate: wp.array[int]
    latched: wp.array[int]
    invalid: wp.array[int]
    imu: wp.array3d[float]
    imu_position: wp.vec3
    imu_rotation: wp.mat33


@wp.kernel
def write_mocap(poses: wp.array3d[float], pos: wp.array2d[wp.vec3], quat: wp.array2d[wp.quat],
                times: wp.array[float], tick: wp.array[int], mocap_id: int,
                steps: int, right: int, dt: float):
    w = wp.tid()
    t = tick[0] % steps + right
    pos[w, mocap_id] = wp.vec3(poses[w, t, 0], poses[w, t, 1], poses[w, t, 2])
    # MJWarp stores wxyz even though the Warp quaternion type normally uses xyzw.
    quat[w, mocap_id] = wp.quat(poses[w, t, 3], poses[w, t, 4], poses[w, t, 5], poses[w, t, 6])
    times[w] = float(wp.float64(tick[0] + right) * wp.float64(dt))


@wp.kernel
def clear_contacts(e: Evaluation, nacon: wp.array[int], ncollision: wp.array[int], nefc: wp.array[int], capacity: int, njmax: int):
    w = wp.tid()
    for j in range(4):
        e.aggregate[w, j] = 0.0
    if nacon[0] > capacity or ncollision[0] > capacity or nefc[w] > njmax:
        e.invalid[w] = 1


@wp.kernel
def reduce_contacts(k: Kinematics, e: Evaluation, cone: int, nacon: wp.array[int],
                    world: wp.array[int], geoms: wp.array[wp.vec2i], dist: wp.array[float],
                    frame: wp.array[wp.mat33], friction: wp.array[vec5], dim: wp.array[int],
                    address: wp.array2d[int], force: wp.array2d[float], njmax: int):
    i = wp.tid()
    if i >= nacon[0]:
        return
    w = world[i]
    if w < 0 or w >= e.aggregate.shape[0]:
        return
    g = geoms[i]
    if g[0] < 0 or g[1] < 0:
        return
    first = e.object_geoms[g[0]] != 0
    second = e.object_geoms[g[1]] != 0
    if first == second:
        return
    other = g[0]
    sign = float(1.0)
    if first:
        other = g[1]
        sign = -1.0
    wp.atomic_max(e.aggregate, w, 0, wp.max(0.0, -dist[i]))
    if e.finger_mask[other] != 0:
        wp.atomic_max(e.aggregate, w, 3, 1.0)
    if e.bottom_geoms[other] != 0:
        wp.atomic_max(e.aggregate, w, 2, 1.0)
        f = contact_force_fn(cone, frame, friction, dim, address, force, njmax, nacon, w, i, True)
        f_table = wp.transpose(k.xmat[w, e.bodies[3]]) @ (sign * wp.spatial_top(f))
        wp.atomic_add(e.aggregate, w, 1, wp.max(0.0, f_table[2]))


@wp.kernel
def evaluate(k: Kinematics, e: Evaluation, tick: wp.array[int], dt: float):
    w = wp.tid()
    obj = e.bodies[2]
    table = e.bodies[3]
    rot = k.xmat[w, table]
    origin = k.xpos[w, table] + rot @ e.target_origin
    p = k.xpos[w, obj]
    object_rot = k.xmat[w, obj]
    contained = bool(True)
    lower = float(1.0e30)
    for i in range(e.points.shape[0]):
        point = wp.transpose(rot) @ (p + object_rot @ e.points[i] - origin)
        lower = wp.min(lower, point[2])
        if wp.abs(point[0]) > e.target_half[0] + e.thresholds[6] or wp.abs(point[1]) > e.target_half[1] + e.thresholds[6]:
            contained = False
    omega_table = wp.spatial_top(k.cvel[w, table])
    relative_vel = (point_velocity(k, w, obj, p) - point_velocity(k, w, table, origin)
                    - wp.cross(omega_table, p - origin))
    relative_omega = wp.spatial_top(k.cvel[w, obj]) - omega_table
    valid = (contained and e.aggregate[w, 2] > 0.0 and e.aggregate[w, 1] > e.thresholds[4]
             and lower >= -e.thresholds[5] and e.aggregate[w, 3] == 0.0
             and wp.length(relative_vel) < e.thresholds[1] and wp.length(relative_omega) < e.thresholds[2]
             and e.aggregate[w, 0] < e.thresholds[3])
    for j in range(k.qpos.shape[1]):
        if not wp.isfinite(k.qpos[w, j]):
            e.invalid[w] = 1
    for j in range(k.qvel.shape[1]):
        if not wp.isfinite(k.qvel[w, j]):
            e.invalid[w] = 1
    if e.latched[w] == 0:
        if valid:
            if e.candidate[w] < 0:
                e.candidate[w] = tick[0] + 1
            if tick[0] + 1 - e.candidate[w] >= int(wp.round(e.thresholds[0] / dt)):
                e.latched[w] = 1
        else:
            e.candidate[w] = -1


@wp.kernel
def sample_imu(k: Kinematics, e: Evaluation, tick: wp.array[int], stride: int, steps: int):
    w = wp.tid()
    if (tick[0] + 1) % stride != 0:
        return
    body = e.bodies[0]
    point = k.xpos[w, body] + k.xmat[w, body] @ e.imu_position
    sensor_from_world = wp.transpose(k.xmat[w, body] @ e.imu_rotation)
    # Gravity-inclusive object acceleration already equals specific force here.
    acceleration = sensor_from_world @ point_acceleration(k, w, body, point)
    angular = sensor_from_world @ wp.spatial_top(k.cvel[w, body])
    slot = (tick[0] % steps + 1) / stride - 1
    for j in range(3):
        e.imu[w, slot, j] = acceleration[j]
        e.imu[w, slot, j + 3] = angular[j]


@wp.kernel
def advance_tick(tick: wp.array[int]):
    tick[0] += 1


@wp.kernel
def pack_state(k: Kinematics, e: Evaluation, actuator_force: wp.array2d[float],
               gravity: wp.vec3, output: wp.array2d[float]):
    w = wp.tid()
    for i in range(5):
        b = e.bodies[i]
        p = k.xpos[w, b]
        v = point_velocity(k, w, b, p)
        omega = wp.spatial_top(k.cvel[w, b])
        a = point_acceleration(k, w, b, p) + gravity
        alpha = wp.spatial_top(k.cacc[w, b])
        for j in range(3):
            output[w, i * 24 + j] = p[j]
            output[w, i * 24 + 12 + j] = v[j]
            output[w, i * 24 + 15 + j] = omega[j]
            output[w, i * 24 + 18 + j] = a[j]
            output[w, i * 24 + 21 + j] = alpha[j]
            for l in range(3):
                output[w, i * 24 + 3 + j * 3 + l] = k.xmat[w, b][j, l]
    for i in range(2):
        for j in range(3):
            output[w, 120 + i * 3 + j] = k.geom_pos[w, e.finger_geoms[i]][j]
            output[w, 126 + i * 3 + j] = k.sensor[w, e.sensor_addresses[i] + j]
    offset = int(132)
    for j in range(k.qpos.shape[1]):
        output[w, offset + j] = k.qpos[w, j]
    offset += k.qpos.shape[1]
    for j in range(k.qvel.shape[1]):
        output[w, offset + j] = k.qvel[w, j]
    offset += k.qvel.shape[1]
    for j in range(k.ctrl.shape[1]):
        output[w, offset + j] = k.ctrl[w, j]
        output[w, offset + k.ctrl.shape[1] + j] = actuator_force[w, j]


class MJWarpBatch:
    """One homogeneous batch, with one CPU/GPU exchange per policy period.

    Environments must already be reset. They supply model construction and static
    task metadata only; ``step`` never invokes a CPU environment step or forward.
    """

    def __init__(self, envs, programs, *, device="cuda:0", nconmax=128, njmax=512, capture=True):
        if not envs or len(envs) != len(programs):
            raise ValueError("one excitation program is required per environment")
        if (mujoco.__version__, mjw.__version__, wp.__version__) != ("3.9.0", "3.9.0", "1.13.0"):
            raise ValueError("MJWarp collection requires the pinned versions in requirements-gpu.txt")
        if (isinstance(nconmax, bool) or isinstance(njmax, bool) or int(nconmax) != nconmax
                or nconmax < 1 or int(njmax) != njmax or njmax < 1):
            raise ValueError("contact and constraint capacities must be positive integers")
        wp.init()
        self.device = wp.get_device(device)
        if str(device).startswith("cuda") and not self.device.is_cuda:
            raise RuntimeError("CUDA requested but unavailable; use --device cpu only for debugging")
        self.envs = list(envs)
        self.programs = list(programs)
        self.nworld = len(envs)
        env = envs[0]
        self.raw_model = env.sim.model._model
        self.dt = float(self.raw_model.opt.timestep)
        self.steps = env._control_steps
        self.imu_stride = round(0.005 / self.dt)
        if self.steps != 250 or not np.isclose(self.dt, 0.0002) or self.imu_stride != 25:
            raise ValueError("collector requires the current 5 kHz physics / 20 Hz policy scheduler")
        self.tier = env.observation_tier
        if self.tier not in ("V0", "V1", "V2", "V3"):
            raise ValueError("collector requires a State observation tier V0–V3")
        signature = self._signature(env)
        if any(self._signature(other) != signature for other in envs):
            raise ValueError("batch models, controller configurations, and observation tiers must match")
        if any(abs(float(other.sim.data.time)) > 1e-12 for other in envs):
            raise ValueError("environments must be freshly reset")
        self.model_sha256 = signature[0]
        self.arm = env.robots[0].part_controllers["right"]
        arm = self.arm
        if not (arm.name == "OSC_POSE" and arm.input_type == "delta" and arm.input_ref_frame == "base"
                and arm.impedance_mode == "fixed" and arm.uncoupling and arm._goal_update_mode == "achieved"
                and arm.interpolator_pos is None and arm.interpolator_ori is None
                and arm.position_limits is None and arm.orientation_limits is None):
            raise ValueError("only default fixed Panda OSC_POSE is supported")
        if not (np.all(arm.input_max == 1) and np.all(arm.input_min == -1)
                and np.array_equal(arm.output_min, -arm.output_max)):
            raise ValueError("collector requires symmetric normalized OSC action scaling")
        self.initial_observations = [copy.deepcopy(other._get_observations(force_update=True)) for other in envs]
        from robosuite.utils.shakebench_tasks import legacy_oracle_observation
        self.initial_observations = [legacy_oracle_observation(o) for o in self.initial_observations]
        self.initial_imus = [copy.deepcopy(other.vibration_provider.imu) if self.tier != "V0" else None for other in envs]
        self.robots = env.robots[0]
        self.initial_arrays = {name: np.stack([np.array(getattr(other.sim.data._data, name)) for other in envs])
                               for name in ("qpos", "qvel", "ctrl", "act", "mocap_pos", "mocap_quat", "qacc_warmstart")}
        with wp.ScopedDevice(self.device):
            self.model = mjw.put_model(self.raw_model)
            if self.model.is_sparse:
                raise ValueError("Panda collector currently requires dense MJWarp inertia storage")
            self.data = mjw.make_data(self.raw_model, nworld=self.nworld, nconmax=nconmax, njmax=njmax)
            self.k = self._kinematics()
            self.c = self._controller()
            self.e = self._evaluation()
            self.tick = wp.zeros(1, dtype=int)
            self.poses = wp.empty((self.nworld, self.steps + 1, 7), dtype=float)
            size = 132 + self.raw_model.nq + self.raw_model.nv + 2 * self.raw_model.nu
            self.packet = wp.empty((self.nworld, size), dtype=float)
            self.graph = None
            self.reset()
            if capture and self.device.is_cuda:
                # Warm compilation first. Capture a 5 ms block and replay it ten
                # times without a host read; this keeps graph construction small.
                self._upload_poses()
                self._block()
                self.reset()
                with wp.ScopedCapture(device=self.device) as captured:
                    self._block()
                self.graph = captured.graph
                self.reset()

    @staticmethod
    def _signature(env):
        import json
        # XML includes assets and physics options; initial qpos is data, not XML.
        xml = env.sim.model.get_xml()
        return (hashlib.sha256(xml.encode()).hexdigest(), env.observation_tier,
                json.dumps(env.robots[0].composite_controller_config, sort_keys=True))

    def _array(self, data, dtype=float):
        return wp.array(np.asarray(data), dtype=dtype, device=self.device)

    def _kinematics(self):
        k = Kinematics()
        fields = {"bias": "qfrc_bias", "mass": "M", "site_pos": "site_xpos", "site_mat": "site_xmat",
                  "geom_pos": "geom_xpos", "com": "subtree_com", "sensor": "sensordata"}
        for name in ("qpos", "qvel", "ctrl", "bias", "mass", "xpos", "xmat", "site_pos", "site_mat",
                     "geom_pos", "com", "cvel", "cacc", "cdof", "sensor"):
            setattr(k, name, getattr(self.data, fields.get(name, name)))
        k.root = self.model.body_rootid
        k.site_body = self.model.site_bodyid
        k.ancestor = self.model.body_isdofancestor
        return k

    def _controller(self):
        arm = self.arm
        c = PandaOSC()
        c.qpos_ids = self._array(arm.qpos_index, int)
        c.dof_ids = self._array(arm.qvel_index, int)
        ids = self.robots._ref_actuators_indexes_dict
        c.actuator_ids = self._array(list(ids["right"]) + list(ids["right_gripper"]), int)
        c.ctrl_range = self._array(self.raw_model.actuator_ctrlrange)
        c.kp, c.kd, c.scale = (self._array(v) for v in (arm.kp, arm.kd, arm.output_max))
        c.initial = self._array([e.robots[0].part_controllers["right"].initial_joint for e in self.envs])
        c.actions = wp.zeros((self.nworld, 7), dtype=float)
        c.goal_pos = wp.zeros(self.nworld, dtype=wp.vec3)
        c.goal_mat = wp.zeros(self.nworld, dtype=wp.mat33)
        c.gripper = wp.zeros((self.nworld, 2), dtype=float)
        c.eef_site = self.envs[0].sim.model.site_name2id(arm.ref_name)
        c.origin_site = self.envs[0].sim.model.site_name2id("robot0_right_center")
        return c

    def _evaluation(self):
        env = self.envs[0]
        e = Evaluation()
        m = env.sim.model
        e.bodies = self._array([m.body_name2id(name) for name in
                               (env.robot_base_body_name, env.gripper_body_name, env.can_body_name,
                                env.worktable_body_name, env.deck_body_name)], int)
        e.finger_geoms = self._array([m.geom_name2id(name) for name in env.finger_pad_geom_names], int)
        if len(env.finger_pad_geom_names) != 2:
            raise ValueError("expected the two Panda finger pads")
        sensor_types = list(self.raw_model.sensor_type)
        e.sensor_addresses = self._array([self.raw_model.sensor_adr[sensor_types.index(kind)] for kind in
                                         (mujoco.mjtSensor.mjSENS_FORCE, mujoco.mjtSensor.mjSENS_TORQUE)], int)
        for field, names in (("object_geoms", env.can.contact_geoms), ("bottom_geoms", env.target_bottom_geom_names),
                             ("finger_mask", env.finger_pad_geom_names)):
            mask = np.zeros(self.raw_model.ngeom, dtype=np.int32)
            mask[[m.geom_name2id(name) for name in names]] = 1
            setattr(e, field, self._array(mask, int))
        env.metrics.success_snapshot(env.sim)  # reuse the compiled collision support vertices
        e.points = self._array(env.metrics._compiled_can_body_points, wp.vec3)
        e.target_origin = wp.vec3(*env.target_frame_local_origin_m)
        e.target_half = wp.vec2(*(np.asarray(env.target_inner_xy_m) / 2))
        t = DEFAULT_SUCCESS_THRESHOLDS
        e.thresholds = self._array([t.hold_duration_s, t.max_relative_linear_speed_m_s,
                                   t.max_relative_angular_speed_rad_s, t.max_illegal_penetration_m,
                                   t.target_bottom_support_force_threshold_N, t.target_bottom_support_z_tolerance_m,
                                   t.containment_epsilon_m])
        e.aggregate = wp.zeros((self.nworld, 4), dtype=float)
        e.candidate = wp.full(self.nworld, -1, dtype=int)
        e.latched = wp.zeros(self.nworld, dtype=int)
        e.invalid = wp.zeros(self.nworld, dtype=int)
        e.imu = wp.zeros((self.nworld, 10, 6), dtype=float)
        provider = env.vibration_provider
        e.imu_position = wp.vec3(*(provider.sensor_position_body_m if self.tier != "V0" else (0, 0, 0)))
        e.imu_rotation = wp.mat33(wxyz_to_matrix(provider.sensor_quat_body_wxyz).flatten()) if self.tier != "V0" else wp.mat33(np.eye(3).flatten())
        return e

    def reset(self):
        """Reset all worlds, including controller, evaluator, and IMU histories."""
        with wp.ScopedDevice(self.device):
            mjw.reset_data(self.model, self.data)
            for name, values in self.initial_arrays.items():
                target = getattr(self.data, name)
                if target.size:
                    wp.copy(target, self._array(values, target.dtype))
            self.tick.zero_()
            self.c.actions.zero_()
            self.c.goal_pos.zero_()
            self.c.goal_mat.zero_()
            self.c.gripper.zero_()
            self.e.candidate.fill_(-1)
            self.e.latched.zero_()
            self.e.invalid.zero_()
            self.e.aggregate.zero_()
            self.e.imu.zero_()
            self.policy_step = 0
            self.imus = copy.deepcopy(self.initial_imus)
            mjw.forward(self.model, self.data)
        return copy.deepcopy(self.initial_observations)

    def _upload_poses(self):
        times = (self.policy_step * self.steps + np.arange(self.steps + 1)) * self.dt
        config = self.envs[0].deck_config
        base = Rotation.from_quat(np.asarray(config.deck_quat_wxyz)[[1, 2, 3, 0]])
        poses = []
        for program in self.programs:
            q = program.evaluate(times).q.copy()  # SciPy's rotation bindings require writable inputs.
            quaternion = (base * Rotation.from_rotvec(q[:, 3:])).as_quat()[:, [3, 0, 1, 2]]
            poses.append(np.concatenate((base.apply(q[:, :3]) + config.deck_pos_m, quaternion), axis=1))
        wp.copy(self.poses, self._array(poses))

    def _launch(self, kernel, args, dim=None):
        wp.launch(kernel, dim=self.nworld if dim is None else dim, inputs=args, device=self.device)

    def _substep(self):
        d, m, e = self.data, self.model, self.e
        mocap = int(self.raw_model.body_mocapid[self.raw_model.body(self.envs[0].deck_config.driver_body_name).id])
        args = [self.poses, d.mocap_pos, d.mocap_quat, d.time, self.tick, mocap, self.steps]
        self._launch(write_mocap, args + [0, self.dt])
        mjw.step1(m, d)
        # Remember transient overflow before the right-limit forward resets the
        # collision counters. An invalid physics substep invalidates the episode.
        self._launch(clear_contacts, [e, d.nacon, d.ncollision, d.nefc, d.naconmax, d.njmax])
        self._launch(set_goals, [self.k, self.c, self.tick, self.steps])
        self._launch(control, [self.k, self.c])
        mjw.step2(m, d)
        self._launch(write_mocap, args + [1, self.dt])
        mjw.forward(m, d)
        self._launch(clear_contacts, [e, d.nacon, d.ncollision, d.nefc, d.naconmax, d.njmax])
        self._launch(reduce_contacts, [self.k, e, int(m.opt.cone), d.nacon, d.contact.worldid,
                    d.contact.geom, d.contact.dist, d.contact.frame, d.contact.friction, d.contact.dim,
                    d.contact.efc_address, d.efc.force, d.njmax], d.naconmax)
        self._launch(evaluate, [self.k, e, self.tick, self.dt])
        if self.tier != "V0":
            # Force/torque sensors request post-constraint RNE during forward.
            self._launch(sample_imu, [self.k, e, self.tick, self.imu_stride, self.steps])
        self._launch(advance_tick, [self.tick], 1)

    def _block(self):
        for _ in range(self.imu_stride):
            self._substep()

    def step(self, actions):
        actions = np.asarray(actions, dtype=float)
        if actions.shape != (self.nworld, 7) or not np.all(np.isfinite(actions)):
            raise ValueError("actions must be finite with shape (nworld, 7)")
        with wp.ScopedDevice(self.device):
            wp.copy(self.c.actions, self._array(np.clip(actions, -1, 1)))
            self._upload_poses()
            for _ in range(self.steps // self.imu_stride):
                if self.graph is None:
                    self._block()
                else:
                    wp.capture_launch(self.graph)
            self._launch(pack_state, [self.k, self.e, self.data.actuator_force,
                         wp.vec3(*self.raw_model.opt.gravity), self.packet])
            # All downloads are batched and occur only at the policy boundary.
            packet = self.packet.numpy()
            samples = self.e.imu.numpy() if self.tier != "V0" else None
            metrics = {"contacts": self.e.aggregate.numpy(), "success": self.e.latched.numpy().astype(bool),
                       "invalid": self.e.invalid.numpy().astype(bool)}
        self.policy_step += 1
        metrics["invalid"] |= ~np.all(np.isfinite(packet), axis=1)
        obs = []
        for w in range(self.nworld):
            if metrics["invalid"][w] or (samples is not None and not np.all(np.isfinite(samples[w]))):
                metrics["invalid"][w] = True
                obs.append(copy.deepcopy(self.initial_observations[w]))
                continue
            if samples is not None:
                for i, sample in enumerate(samples[w]):
                    self.imus[w].acquire(sample, timestamp_s=((self.policy_step - 1) * 10 + i + 1) * 0.005)
            obs.append(self._observation(w, packet[w]))
        offset = 132
        metrics["qpos"] = packet[:, offset:offset + self.raw_model.nq].copy()
        offset += self.raw_model.nq
        metrics["qvel"] = packet[:, offset:offset + self.raw_model.nv].copy()
        offset += self.raw_model.nv
        metrics["ctrl"] = packet[:, offset:offset + self.raw_model.nu].copy()
        metrics["actuator_force"] = packet[:, offset + self.raw_model.nu:].copy()
        return obs, metrics

    def _observation(self, w, packet):
        values = packet[:120].astype(float).reshape(5, 24)
        bodies = [RigidBodyState(v[:3], v[3:12].reshape(3, 3), v[12:18], v[18:24]) for v in values]
        base, hand, obj, table, deck = bodies
        rotation = base.rotation_world.T
        origin = table.position_world_m + table.rotation_world @ self.envs[0].target_frame_local_origin_m
        nq = self.raw_model.nq
        q, v = packet[132:132 + nq], packet[132 + nq:132 + nq + self.raw_model.nv]
        result = {key: self.initial_observations[w][key].copy() for key in COMMON_STATE_KEYS}
        result.update({
            "robot0_joint_pos": q[self.arm.qpos_index], "robot0_joint_vel": v[self.arm.qvel_index],
            "robot0_eef_pos_robot_base": rotation @ (hand.position_world_m - base.position_world_m),
            "robot0_eef_quat_robot_base": T.mat2quat(rotation @ hand.rotation_world),
            "robot0_gripper_state": np.concatenate((q[self.robots._ref_gripper_joint_pos_indexes["right"]],
                                                    v[self.robots._ref_gripper_joint_vel_indexes["right"]])),
            "robot0_wrist_force": packet[126:129], "robot0_wrist_torque": packet[129:132],
            "robot0_fingertip_pos_robot_base": ((packet[120:126].reshape(2, 3) - base.position_world_m) @ rotation.T).flatten(),
            "can_pos_robot_base": rotation @ (obj.position_world_m - base.position_world_m),
            "can_quat_robot_base": T.mat2quat(rotation @ obj.rotation_world),
            "goal_frame_pos_robot_base": rotation @ (origin - base.position_world_m),
            "goal_frame_quat_robot_base": T.mat2quat(rotation @ table.rotation_world),
        })
        if self.tier != "V0":
            result.update(self.imus[w].to_policy_observation())
        if self.tier in ("V2", "V3"):
            config = self.envs[0].deck_config
            nominal = RigidBodyState(np.array(config.deck_pos_m), wxyz_to_matrix(config.deck_quat_wxyz), np.zeros(6), np.zeros(6))
            d = relative_pose_twist_acceleration(deck, nominal)
            t = relative_pose_twist_acceleration(table, deck)
            result.update(SupportState(d.pose, d.twist, d.acceleration, t.pose, t.twist, t.acceleration).to_policy_observation())
        if self.tier == "V3":
            provider = self.envs[w].vibration_provider
            provider._episode_time_s = self.policy_step / 20
            result.update(provider.public_program_payload())
        return {key: np.asarray(value, dtype=POLICY_FIELD_CONTRACT[key]["dtype"]).copy() for key, value in result.items()}
