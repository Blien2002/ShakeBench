"""Measure the eight-object pick-place set and print registry-ready values.

For every declared start pose this compiles the real asset, then:

* extracts the contact-geometry envelope in the object frame,
* settles the posed object on a plane and reports the residual motion, and
* prints the closing-axis span at each height slice, which is the jaw width
  the grasp plan has to cover (the compiled Panda jaw opens to 80 mm).

Poses that keep moving under gravity are re-declared from their settled pose
and checked again, so an unstable authored orientation cannot reach the task
registry.

Usage: python tools/pick_design.py
"""

from __future__ import annotations

import math
import shutil
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

ASSETS = "shakebench/models/assets/objects/robocasa"
#: RoboCasa instances drop the free joint; the probe re-parents the object body
#: exactly like robosuite's MJCF merge does.
STABILITY_TRANSLATION_M = 0.010
STABILITY_ROTATION_RAD = math.radians(3.0)


def yaw(theta: float) -> tuple[float, float, float, float]:
    return (math.cos(theta / 2.0), 0.0, 0.0, math.sin(theta / 2.0))


#: object_id -> (asset xml, declared start quaternion wxyz)
DECLARED = {
    "mug": (f"{ASSETS}/mug/mug_1/model.xml", yaw(math.pi / 2)),
    "apple": (f"{ASSETS}/apple/apple_0/model.xml", yaw(0.0)),
    "spatula": (f"{ASSETS}/spatula/spatula_0/model.xml", None),
    "bar_soap": (f"{ASSETS}/bar_soap/bar_soap_0/model.xml", yaw(math.pi / 2)),
    "cereal": (f"{ASSETS}/cereal/cereal_0/model.xml", yaw(math.pi / 2)),
    "rolling_pin": (f"{ASSETS}/rolling_pin/rolling_pin_0/model.xml", yaw(math.pi / 2)),
    "potato": (f"{ASSETS}/potato/potato_1/model.xml", yaw(math.pi / 2)),
}


def collision_points(model, data) -> np.ndarray:
    from pick_measure import geom_points

    return np.concatenate(
        [
            geom_points(model, data, geom_id)
            for geom_id in range(model.ngeom)
            if int(model.geom_contype[geom_id]) or int(model.geom_conaffinity[geom_id])
        ]
    )


def quat_from_matrix(rotation: np.ndarray) -> tuple[float, float, float, float]:
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.asarray(rotation, dtype=float).reshape(-1))
    return tuple(float(value) for value in quat)


def flat_pose(points: np.ndarray) -> tuple[float, float, float, float]:
    """Return the pose that lays the thinnest principal axis along world +z."""

    _, _, axes = np.linalg.svd(points - points.mean(axis=0), full_matrices=False)
    long_axis, short_axis, thin_axis = axes[0], axes[1], axes[2]
    if np.linalg.det(np.vstack([long_axis, short_axis, thin_axis])) < 0:
        short_axis = -short_axis
    if thin_axis[2] < 0:
        long_axis, short_axis, thin_axis = -long_axis, -short_axis, -thin_axis
    rotation = np.vstack([long_axis, short_axis, thin_axis])
    return quat_from_matrix(rotation)


def settle(path: str, quat, bottom: float, *, drop_m: float = 0.002, duration_s: float = 2.0) -> dict:
    """Drop the real asset from ``quat`` onto a plane and report the outcome."""

    source = Path(path)
    staging = Path(tempfile.mkdtemp(prefix="pickv2_settle_"))
    staged_dir = staging / source.parent.name
    shutil.copytree(source.parent, staged_dir)
    staged = staged_dir / source.name
    tree = ET.parse(staged)
    worldbody = tree.getroot().find("./worldbody")
    wrapper = worldbody.find("./body")
    body = wrapper.find("./body[@name='object']")
    worldbody.remove(wrapper)
    worldbody.append(body)
    body.insert(0, ET.Element("freejoint"))
    tree.write(staged, encoding="unicode")
    xml = (
        "<mujoco><option timestep='0.0002'/><worldbody>"
        "<geom name='floor' type='plane' size='1 1 0.1' friction='0.3 0.005 0.0001'/></worldbody>"
        f"<include file='{staged}'/></mujoco>"
    )
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    data.qpos[:3] = (0.0, 0.0, -bottom + drop_m)
    data.qpos[3:7] = np.asarray(quat, dtype=float)
    start = np.array(data.qpos, dtype=float, copy=True)
    for _ in range(int(duration_s / model.opt.timestep)):
        mujoco.mj_step(model, data)
    delta = np.asarray(data.qpos, dtype=float) - start
    return {
        "quat_wxyz": tuple(float(value) for value in data.qpos[3:7]),
        "translation_m": float(np.linalg.norm(delta[:3])),
        "rotation_rad": float(2.0 * np.arcsin(min(1.0, float(np.linalg.norm(delta[3:]))))),
    }


def posed_bottom(points: np.ndarray, quat) -> float:
    rotation = np.zeros(9)
    mujoco.mju_quat2Mat(rotation, np.asarray(quat, dtype=float))
    return float(points.dot(rotation.reshape(3, 3).T)[:, 2].min())


def stable_pose(path: str, quat, points: np.ndarray) -> tuple[tuple[float, ...], dict]:
    settled = settle(path, quat, posed_bottom(points, quat))
    for _ in range(4):
        if settled["translation_m"] <= STABILITY_TRANSLATION_M and settled["rotation_rad"] <= STABILITY_ROTATION_RAD:
            break
        settled = settle(path, settled["quat_wxyz"], posed_bottom(points, settled["quat_wxyz"]))
    return settled["quat_wxyz"], settled


def describe(object_id: str, path: str) -> None:
    model = mujoco.MjModel.from_xml_path(path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    points = collision_points(model, data)
    lower = float(points[:, 2].min())
    upper = float(points[:, 2].max())
    radius = float(np.max(np.linalg.norm(points[:, :2], axis=1)))
    declared = DECLARED[object_id][1] or flat_pose(points)
    quat, settled = stable_pose(path, declared, points)
    rotation = np.zeros(9)
    mujoco.mju_quat2Mat(rotation, np.asarray(quat, dtype=float))
    posed = points.dot(rotation.reshape(3, 3).T)
    base = posed[:, 2].min()
    height = posed[:, 2].max() - base
    print(f"== {object_id}")
    print(f"   support: lower {lower:.9f} upper {upper:.9f} radius {radius:.9f}")
    print(f"   object-frame collision bbox m: {np.round(points.max(axis=0) - points.min(axis=0), 5)}")
    print(f"   settled quat_wxyz: ({quat[0]:.9f}, {quat[1]:.9f}, {quat[2]:.9f}, {quat[3]:.9f})")
    print(
        f"   residual: translation {settled['translation_m'] * 1e3:.1f} mm"
        f" rotation {math.degrees(settled['rotation_rad']):.2f} deg"
    )
    print(
        f"   posed extent x={posed[:,0].ptp():.4f} y={posed[:,1].ptp():.4f} z={posed[:,2].ptp():.4f}"
        f"  offset x[{posed[:,0].min():+.4f},{posed[:,0].max():+.4f}] y[{posed[:,1].min():+.4f},{posed[:,1].max():+.4f}]"
    )
    for fraction in np.linspace(0.05, 0.95, 10):
        level = base + fraction * height
        slab = posed[np.abs(posed[:, 2] - level) <= 0.006]
        if slab.shape[0] < 3:
            continue
        print(
            f"   h={fraction * height * 1e3:6.1f} mm  jaw-span(y) {slab[:,1].ptp() * 1e3:6.1f} mm"
            f"  x-span {slab[:,0].ptp() * 1e3:7.1f} mm  n={slab.shape[0]}"
        )


def main() -> None:
    for object_id, (path, _) in DECLARED.items():
        describe(object_id, path)


if __name__ == "__main__":
    main()
