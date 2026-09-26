"""Furniture-style wine rack visuals; contact geometry stays environment-owned."""

import xml.etree.ElementTree as ET
from itertools import product

import numpy as np
from scipy.spatial import ConvexHull

from robosuite.utils.mjcf_utils import array_to_string
from shakebench.models import xml_path_completion

VISUAL_REVISION = "beveled_walnut_oak_v1"
_VISUAL = {"group": "1", "contype": "0", "conaffinity": "0", "mass": "0"}


def _board(asset, body, name, position, size, material, offset):
    """Chamfer a board inside its collision envelope, with face-specific grain UVs."""
    size = np.asarray(size)
    bevel = min(0.0012, size.min() * 0.3)
    vertices = np.array(
        [
            np.asarray(signs) * (size - bevel + np.eye(3)[axis] * bevel)
            for signs in product((-1, 1), repeat=3)
            for axis in range(3)
        ]
    )
    hull = ConvexHull(vertices)
    points, normals, uv = [], [], []
    long_axis = int(np.argmax(size))
    for face, equation in zip(hull.simplices, hull.equations):
        triangle = vertices[face].copy()
        normal = equation[:3]
        if np.dot(np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0]), normal) < 0:
            triangle[[1, 2]] = triangle[[2, 1]]
        face_axis = int(np.argmax(np.abs(normal)))
        along = long_axis if long_axis != face_axis else (face_axis + 1) % 3
        across = next(axis for axis in range(3) if axis not in (face_axis, along))
        points.extend(triangle)
        normals.extend([normal] * 3)
        uv.extend(np.column_stack((triangle[:, across] / 0.12 + offset, triangle[:, along] / 0.24 + offset)))
    ET.SubElement(
        asset,
        "mesh",
        name=name,
        vertex=array_to_string(np.asarray(points).ravel()),
        normal=array_to_string(np.asarray(normals).ravel()),
        face=array_to_string(np.arange(len(points))),
        texcoord=array_to_string(np.asarray(uv).ravel()),
    )
    ET.SubElement(
        body, "geom", name=name, type="mesh", mesh=name, pos=array_to_string(position), material=material, **_VISUAL
    )


def add_rack_visuals(asset, body, boards):
    """Add satin wood, individual floor planks, chamfered edges and flush fasteners."""
    ET.SubElement(
        asset,
        "texture",
        name="wine_rack_grain",
        type="2d",
        file=xml_path_completion("textures/ambientcg_wood095_color_1k.png"),
    )
    for name, tint in (
        ("walnut", [0.60, 0.43, 0.31, 1]),
        ("oak", [0.94, 0.84, 0.69, 1]),
        ("oak_light", [1.0, 0.91, 0.76, 1]),
        ("oak_warm", [0.89, 0.75, 0.58, 1]),
    ):
        ET.SubElement(
            asset,
            "material",
            name=f"wine_rack_{name}",
            texture="wine_rack_grain",
            rgba=array_to_string(tint),
            texrepeat="1 1",
            specular="0.18",
            shininess="0.24",
        )
    ET.SubElement(
        asset, "material", name="wine_rack_fastener", rgba="0.22 0.19 0.15 1", specular="0.45", shininess="0.6"
    )
    for index, (name, position, size) in enumerate(boards):
        if name.endswith("_floor"):
            for plank in range(3):
                center = np.asarray(position, dtype=float)
                center[1] += (plank - 1) * 2 * size[1] / 3
                # A 0.4 mm join reads as a surface seam over the continuous support floor.
                half_size = [size[0], size[1] / 3 - 0.0002, size[2]]
                material = ("oak", "oak_light", "oak_warm")[(plank + index) % 3]
                _board(
                    asset,
                    body,
                    f"{name}_plank_{plank}_visual",
                    center,
                    half_size,
                    f"wine_rack_{material}",
                    index * 0.23 + plank * 0.17,
                )
        else:
            _board(asset, body, f"{name}_visual", position, size, "wine_rack_walnut", index * 0.23)
        if name.endswith("_back"):
            # Screw heads and their slots sit flush with the visible rear face.
            for y in (-0.15, -0.05, 0.05, 0.15):
                for z in (0.022, 0.092):
                    x = position[0] + size[0]
                    suffix = f"{y}_{z}"
                    ET.SubElement(
                        body,
                        "geom",
                        name=f"wine_rack_screw_{suffix}",
                        type="cylinder",
                        pos=array_to_string([x, y, z]),
                        size="0.0022 0.00012",
                        quat="0.70710678 0 0.70710678 0",
                        material="wine_rack_fastener",
                        **_VISUAL,
                    )
                    ET.SubElement(
                        body,
                        "geom",
                        name=f"wine_rack_screw_slot_{suffix}",
                        type="box",
                        pos=array_to_string([x + 0.00013, y, z]),
                        size="0.00002 0.0013 0.0002",
                        rgba="0.045 0.035 0.025 1",
                        **_VISUAL,
                    )
