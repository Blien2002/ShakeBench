"""Batched fixed-impedance Panda OSC for the optional MJWarp collector.

All per-substep computation stays on the Warp device. Small controller matrices
use float64 to retain the reference controller's pseudoinverse cutoff; simulator
state remains float32. Only the default Panda OSC_POSE configuration is supported.
"""

import warp as wp

wp.set_module_options({"enable_backward": False})

Mat77 = wp.types.matrix(shape=(7, 7), dtype=wp.float64)
Mat67 = wp.types.matrix(shape=(6, 7), dtype=wp.float64)
Mat66 = wp.types.matrix(shape=(6, 6), dtype=wp.float64)
Vec7 = wp.types.vector(length=7, dtype=wp.float64)
Vec6 = wp.types.vector(length=6, dtype=wp.float64)


@wp.struct
class Kinematics:
    qpos: wp.array2d[float]
    qvel: wp.array2d[float]
    ctrl: wp.array2d[float]
    bias: wp.array2d[float]
    mass: wp.array3d[float]
    xpos: wp.array2d[wp.vec3]
    xmat: wp.array2d[wp.mat33]
    site_pos: wp.array2d[wp.vec3]
    site_mat: wp.array2d[wp.mat33]
    geom_pos: wp.array2d[wp.vec3]
    com: wp.array2d[wp.vec3]
    cvel: wp.array2d[wp.spatial_vector]
    cacc: wp.array2d[wp.spatial_vector]
    cdof: wp.array2d[wp.spatial_vector]
    sensor: wp.array2d[float]
    root: wp.array[int]
    site_body: wp.array[int]
    ancestor: wp.array2d[int]


@wp.struct
class PandaOSC:
    qpos_ids: wp.array[int]
    dof_ids: wp.array[int]
    actuator_ids: wp.array[int]
    ctrl_range: wp.array2d[float]
    kp: wp.array[float]
    kd: wp.array[float]
    scale: wp.array[float]
    initial: wp.array2d[float]
    actions: wp.array2d[float]
    goal_pos: wp.array[wp.vec3]
    goal_mat: wp.array[wp.mat33]
    gripper: wp.array2d[float]
    eef_site: int
    origin_site: int


@wp.func
def point_velocity(k: Kinematics, w: int, body: int, point: wp.vec3):
    v = k.cvel[w, body]
    offset = point - k.com[w, k.root[body]]
    return wp.spatial_bottom(v) + wp.cross(wp.spatial_top(v), offset)


@wp.func
def point_acceleration(k: Kinematics, w: int, body: int, point: wp.vec3):
    """MuJoCo object acceleration, including its gravity convention."""
    a = k.cacc[w, body]
    offset = point - k.com[w, k.root[body]]
    return (wp.spatial_bottom(a) + wp.cross(wp.spatial_top(a), offset)
            + wp.cross(wp.spatial_top(k.cvel[w, body]), point_velocity(k, w, body, point)))


@wp.func
def inverse_spd(a: Mat77):
    # Seven-joint SPD Cholesky solve, with no host linalg or device allocation.
    l = Mat77()
    for i in range(7):
        for j in range(i + 1):
            value = a[i, j]
            for t in range(j):
                value -= l[i, t] * l[j, t]
            if i == j:
                l[i, j] = wp.sqrt(value)
            else:
                l[i, j] = value / l[j, j]
    result = Mat77()
    for col in range(7):
        y = Vec7()
        x = Vec7()
        for i in range(7):
            value = wp.float64(0.0)
            if i == col:
                value = wp.float64(1.0)
            for j in range(i):
                value -= l[i, j] * y[j]
            y[i] = value / l[i, i]
        for reverse in range(7):
            i = 6 - reverse
            value = y[i]
            for j in range(i + 1, 7):
                value -= l[j, i] * x[j]
            x[i] = value / l[i, i]
            result[i, col] = x[i]
    return result


@wp.func
def symmetric_pinv(value: Mat66):
    """Jacobi eigensolve for symmetric 6x6 (or zero-padded 3x3) matrices."""
    a = value
    vectors = Mat66()
    for i in range(6):
        vectors[i, i] = wp.float64(1.0)
    for sweep in range(24):
        largest = wp.float64(0.0)
        for p in range(6):
            for q in range(p + 1, 6):
                off = a[p, q]
                largest = wp.max(largest, wp.abs(off))
                if wp.abs(off) > wp.float64(1.0e-30):
                    tau = (a[q, q] - a[p, p]) / (wp.float64(2.0) * off)
                    sign = wp.float64(1.0)
                    if tau < wp.float64(0.0):
                        sign = wp.float64(-1.0)
                    t = sign / (wp.abs(tau) + wp.sqrt(wp.float64(1.0) + tau * tau))
                    c = wp.float64(1.0) / wp.sqrt(wp.float64(1.0) + t * t)
                    s = t * c
                    a[p, p] -= t * off
                    a[q, q] += t * off
                    a[p, q] = wp.float64(0.0)
                    a[q, p] = wp.float64(0.0)
                    for r in range(6):
                        if r != p and r != q:
                            rp = a[r, p]
                            rq = a[r, q]
                            a[r, p] = c * rp - s * rq
                            a[p, r] = a[r, p]
                            a[r, q] = s * rp + c * rq
                            a[q, r] = a[r, q]
                        vp = vectors[r, p]
                        vq = vectors[r, q]
                        vectors[r, p] = c * vp - s * vq
                        vectors[r, q] = s * vp + c * vq
        if largest < wp.float64(1.0e-25):
            break
    maximum = wp.float64(0.0)
    for i in range(6):
        maximum = wp.max(maximum, wp.abs(a[i, i]))
    result = Mat66()
    for i in range(6):
        if wp.abs(a[i, i]) > maximum * wp.float64(1.0e-15):
            for r in range(6):
                for column in range(6):
                    result[r, column] += vectors[r, i] * vectors[column, i] / a[i, i]
    return result


@wp.kernel
def set_goals(k: Kinematics, c: PandaOSC, tick: wp.array[int], control_steps: int):
    w = wp.tid()
    if tick[0] % control_steps != 0:
        return
    origin = k.site_pos[w, c.origin_site]
    rotation = k.site_mat[w, c.origin_site]
    delta = wp.vec3(c.actions[w, 0] * c.scale[0], c.actions[w, 1] * c.scale[1], c.actions[w, 2] * c.scale[2])
    axis = wp.vec3(c.actions[w, 3] * c.scale[3], c.actions[w, 4] * c.scale[4], c.actions[w, 5] * c.scale[5])
    c.goal_pos[w] = wp.transpose(rotation) @ (k.site_pos[w, c.eef_site] - origin) + delta
    angle = wp.length(axis)
    delta_rotation = wp.identity(n=3, dtype=float)
    if angle > 1.0e-12:
        delta_rotation = wp.quat_to_matrix(wp.quat_from_axis_angle(axis / angle, angle))
    c.goal_mat[w] = delta_rotation @ wp.transpose(rotation) @ k.site_mat[w, c.eef_site]
    for finger in range(2):
        direction = float(2 * finger - 1)
        command = c.actions[w, 6]
        sign = float(0.0)
        if command > 0.0:
            sign = 1.0
        elif command < 0.0:
            sign = -1.0
        c.gripper[w, finger] = wp.clamp(c.gripper[w, finger] + direction * 0.2 * sign, -1.0, 1.0)


@wp.kernel
def control(k: Kinematics, c: PandaOSC):
    w = wp.tid()
    body = k.site_body[c.eef_site]
    base = k.site_body[c.origin_site]
    position = k.site_pos[w, c.eef_site]
    origin = k.site_pos[w, c.origin_site]
    rotation = k.site_mat[w, c.origin_site]
    current = k.site_mat[w, c.eef_site]
    desired = rotation @ c.goal_mat[w]
    orientation_error = wp.vec3(0.0)
    for j in range(3):
        orientation_error += 0.5 * wp.cross(wp.vec3(current[0, j], current[1, j], current[2, j]),
                                            wp.vec3(desired[0, j], desired[1, j], desired[2, j]))
    position_error = origin + rotation @ c.goal_pos[w] - position
    velocity_error = point_velocity(k, w, base, origin) - point_velocity(k, w, body, position)
    angular_error = wp.spatial_top(k.cvel[w, base]) - wp.spatial_top(k.cvel[w, body])
    wrench = Vec6()
    for i in range(3):
        wrench[i] = wp.float64(c.kp[i] * position_error[i] + c.kd[i] * velocity_error[i])
        wrench[i + 3] = wp.float64(c.kp[i + 3] * orientation_error[i] + c.kd[i + 3] * angular_error[i])
    mass = Mat77()
    jac = Mat67()
    offset = position - k.com[w, k.root[body]]
    for j in range(7):
        dof = c.dof_ids[j]
        if k.ancestor[body, dof] != 0:
            axis = wp.spatial_top(k.cdof[w, dof])
            linear = wp.spatial_bottom(k.cdof[w, dof]) + wp.cross(axis, offset)
            for i in range(3):
                jac[i, j] = wp.float64(linear[i])
                jac[i + 3, j] = wp.float64(axis[i])
        for i in range(7):
            mass[i, j] = wp.float64(k.mass[w, c.dof_ids[i], dof])
    inv_mass = inverse_spd(mass)
    lambda_inv = jac @ inv_mass @ wp.transpose(jac)
    lambda_full = symmetric_pinv(lambda_inv)
    pos = Mat66()
    ori = Mat66()
    for i in range(3):
        for j in range(3):
            pos[i, j] = lambda_inv[i, j]
            ori[i + 3, j + 3] = lambda_inv[i + 3, j + 3]
    decoupled = (symmetric_pinv(pos) + symmetric_pinv(ori)) @ wrench
    nullspace = wp.identity(n=7, dtype=wp.float64) - inv_mass @ wp.transpose(jac) @ lambda_full @ jac
    null_acc = Vec7()
    for i in range(7):
        null_acc[i] = (wp.float64(10.0) * wp.float64(c.initial[w, i] - k.qpos[w, c.qpos_ids[i]])
                       - wp.sqrt(wp.float64(40.0)) * wp.float64(k.qvel[w, c.dof_ids[i]]))
    torque = wp.transpose(jac) @ decoupled + wp.transpose(nullspace) @ mass @ null_acc
    for i in range(7):
        actuator = c.actuator_ids[i]
        k.ctrl[w, actuator] = wp.clamp(float(torque[i]) + k.bias[w, c.dof_ids[i]],
                                      c.ctrl_range[actuator, 0], c.ctrl_range[actuator, 1])
    for i in range(2):
        actuator = c.actuator_ids[i + 7]
        low = c.ctrl_range[actuator, 0]
        high = c.ctrl_range[actuator, 1]
        k.ctrl[w, actuator] = 0.5 * (high + low) + 0.5 * (high - low) * c.gripper[w, i]
