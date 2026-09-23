"""Measure candidate assets for the 8-object pick-and-place task set.

Compiles each instance MJCF exactly as MuJoCo would see it, then reports the
collision-geometry bounding box, principal-axis extents (a proxy for the
stable lying pose) and the compiled body mass/inertia that the visual geoms
would contribute if they were left in the inertial computation.

Usage: python tools/pick_measure.py <name>=<xml path> [<name>=<xml> ...]
"""

from __future__ import annotations

import sys

import mujoco
import numpy as np


def geom_points(model, data, geom_id) -> np.ndarray:
    geom_type = int(model.geom_type[geom_id])
    if geom_type == int(mujoco.mjtGeom.mjGEOM_MESH):
        mesh_id = int(model.geom_dataid[geom_id])
        start = int(model.mesh_vertadr[mesh_id])
        count = int(model.mesh_vertnum[mesh_id])
        local = np.asarray(model.mesh_vert[start : start + count], dtype=float).reshape(-1, 3)
    elif geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
        size = np.asarray(model.geom_size[geom_id], dtype=float)
        local = np.asarray(
            [[sx * size[0], sy * size[1], sz * size[2]] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
        )
    elif geom_type == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        r = float(model.geom_size[geom_id][0])
        local = np.asarray([[sx * r, sy * r, sz * r] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    else:
        radius = float(model.geom_size[geom_id][0])
        half = float(model.geom_size[geom_id][1]) if model.geom_size[geom_id].size > 1 else radius
        angles = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)
        local = np.asarray(
            [[radius * np.cos(a), radius * np.sin(a), z] for a in angles for z in (-half, half)], dtype=float
        )
    return np.asarray(data.geom_xpos[geom_id], dtype=float) + local.dot(
        np.asarray(data.geom_xmat[geom_id]).reshape(3, 3).T
    )


def measure(path: str) -> dict:
    model = mujoco.MjModel.from_xml_path(path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    collision, visual = [], []
    for geom_id in range(model.ngeom):
        points = geom_points(model, data, geom_id)
        solid = bool(int(model.geom_contype[geom_id]) or int(model.geom_conaffinity[geom_id]))
        (collision if solid else visual).append(points)
    collision = np.concatenate(collision)
    visual = np.concatenate(visual) if visual else collision
    body = model.body("object").id
    centre = np.asarray(model.body_ipos[body], dtype=float)
    centred = collision - centre
    # Principal axes of the centred collision cloud: the lying pose is the one
    # where the two largest extents are horizontal.
    _, _, axes_T = np.linalg.svd(centred, full_matrices=False)
    principal = centred.dot(axes_T.T)
    return {
        "ngeom": int(model.ngeom),
        "collision_min": collision.min(axis=0),
        "collision_max": collision.max(axis=0),
        "collision_size": collision.max(axis=0) - collision.min(axis=0),
        "visual_size": visual.max(axis=0) - visual.min(axis=0),
        "compiled_mass_kg": float(model.body_mass[body]),
        "compiled_com": centre,
        "compiled_inertia": np.array(model.body_inertia[body]),
        "principal_extents": principal.max(axis=0) - principal.min(axis=0),
        "hull_volume_m3": _hull_volume(centred),
    }


def _hull_volume(points: np.ndarray) -> float:
    """Convex-hull volume via scipy when available; -1 otherwise."""

    try:
        from scipy.spatial import ConvexHull
    except ImportError:
        return -1.0
    return float(ConvexHull(points).volume)


def main() -> None:
    for entry in sys.argv[1:]:
        name, path = entry.split("=", 1)
        result = measure(path)
        size = result["collision_size"]
        print(f"== {name}  geoms={result['ngeom']}")
        print(
            "   collision size m: "
            f"{size[0]:.5f} x {size[1]:.5f} x {size[2]:.5f}   (mm: "
            f"{size[0] * 1e3:.1f} x {size[1] * 1e3:.1f} x {size[2] * 1e3:.1f})"
        )
        print(f"   collision min: {np.round(result['collision_min'], 5)}  max: {np.round(result['collision_max'], 5)}")
        print(f"   visual size m: {np.round(result['visual_size'], 5)}")
        print(f"   principal extents m: {np.round(result['principal_extents'], 5)} (mm {np.round(result['principal_extents'] * 1e3, 1)})")
        print(f"   compiled mass {result['compiled_mass_kg']:.4f} kg (density-derived, not physical)")
        print(f"   compiled com {np.round(result['compiled_com'], 5)} inertia {np.round(result['compiled_inertia'], 8)}")
        print(f"   hull volume {result['hull_volume_m3'] * 1e6:.1f} cm^3")


if __name__ == "__main__":
    main()
