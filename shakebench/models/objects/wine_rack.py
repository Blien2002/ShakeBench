"""Open oak wine rack with rounded scalloped rails and matching convex contacts."""

import xml.etree.ElementTree as ET
from itertools import product

import numpy as np
from scipy.spatial import ConvexHull

from robosuite.utils.mjcf_utils import array_to_string
from shakebench.models import xml_path_completion

VISUAL_REVISION = "open_scalloped_oak_v2"
SLOT_Y_M = {"middle": 0.0, "left": 0.10, "right": -0.10}
BOTTLE_AXIS_HEIGHT_M = 0.067
RAIL_HALF_THICKNESS_M = 0.014
RAIL_BOTTOM_M = 0.012
# Name, longitudinal position, cut radius, cut centre height, shoulder height.
# The rear rail is cut for the narrower bottle neck.
RAILS = (("body", -0.078, 0.038, 0.0705, 0.065), ("neck", 0.078, 0.022, 0.0745, 0.075))
_VISUAL = {"group": "1", "contype": "0", "conaffinity": "0", "mass": "0"}
_COLLISION = {"group": "0", "contype": "1", "conaffinity": "1", "mass": "0"}


def rail_height(y, radius, center_z, shoulder):
    """Circular cradle bottom with a smooth transition onto the flat shoulders."""
    distance = np.min(np.abs(np.asarray(y)[..., None] - np.asarray(list(SLOT_Y_M.values()))), axis=-1)
    cutoff = np.sqrt(radius**2 - max(center_z - shoulder, 0) ** 2)
    start, end = cutoff - 0.004, cutoff + 0.002
    root = np.sqrt(radius**2 - start**2)
    height, slope = center_z - root, start / root
    t = np.clip((distance - start) / (end - start), 0, 1)
    blend = (2 * t**3 - 3 * t**2 + 1) * height + (t**3 - 2 * t**2 + t) * (end - start) * slope
    blend += (-2 * t**3 + 3 * t**2) * shoulder
    circle = center_z - np.sqrt(np.maximum(radius**2 - distance**2, 0))
    return np.where(distance <= start, circle, np.where(distance < end, blend, shoulder))


def _rounded_rail(asset, body, name, x, radius, center_z, shoulder):
    """Sweep a rounded rectangle along the scalloped top, with smooth analytic normals."""
    y_values = np.linspace(-0.166, 0.166, 333)
    heights = rail_height(y_values, radius, center_z, shoulder)
    slopes = np.gradient(heights, y_values)
    bevel = 0.002
    vertices, normals, uv, faces = [], [], [], []
    # Counterclockwise in x/z; the long grain follows the transverse y axis.
    corners = ((1, False, -90), (1, True, 0), (-1, True, 90), (-1, False, 180))
    for y, top, slope in zip(y_values, heights, slopes):
        for side, upper, start in corners:
            for angle in np.deg2rad(np.linspace(start, start + 90, 6)):
                nx, nz = np.cos(angle), np.sin(angle)
                px = side * (RAIL_HALF_THICKNESS_M - bevel) + bevel * nx
                pz = (top - bevel if upper else RAIL_BOTTOM_M + bevel) + bevel * nz
                vertices.append([px, y, pz])
                normal = np.array([nx, -nz * slope if upper else 0, nz])
                normals.append(normal / np.linalg.norm(normal))
                uv.append([y / 0.8 + 0.5, (pz if abs(nx) > abs(nz) else px) / 0.8 + 0.32 + x])
    count = 24
    for row in range(len(y_values) - 1):
        for j in range(count):
            a, b = row * count + j, (row + 1) * count + j
            c, d = (row + 1) * count + (j + 1) % count, row * count + (j + 1) % count
            faces.extend(((a, b, c), (a, c, d)))
    for row, sign in ((0, -1), (len(y_values) - 1, 1)):
        cap = np.asarray(vertices[row * count : (row + 1) * count])
        center = len(vertices)
        vertices.append([0, y_values[row], (RAIL_BOTTOM_M + heights[row]) / 2])
        vertices.extend(cap)
        normals.extend([[0, sign, 0]] * (count + 1))
        uv.extend([[0.5, 0.5], *np.column_stack((cap[:, 0] / 0.8 + 0.5, cap[:, 2] / 0.8 + 0.5))])
        for j in range(count):
            triangle = [center, center + 1 + j, center + 1 + (j + 1) % count]
            faces.append(triangle if sign < 0 else triangle[::-1])
    ET.SubElement(
        asset,
        "mesh",
        name=name,
        vertex=array_to_string(np.asarray(vertices).ravel()),
        normal=array_to_string(np.asarray(normals).ravel()),
        face=array_to_string(np.asarray(faces).ravel()),
        texcoord=array_to_string(np.asarray(uv).ravel()),
    )
    ET.SubElement(body, "geom", name=name, type="mesh", mesh=name, pos=f"{x} 0 0", material="wine_rack_oak", **_VISUAL)


def _runner(asset, body, name, y):
    """Low longitudinal feet tie the two rails together without closing the bays."""
    size, bevel = np.array([0.124, 0.012, 0.012]), 0.002
    points = np.array(
        [
            np.asarray(signs) * (size - bevel + np.eye(3)[axis] * bevel)
            for signs in product((-1, 1), repeat=3)
            for axis in range(3)
        ]
    )
    hull = ConvexHull(points)
    vertices, normals, uv = [], [], []
    for face, equation in zip(hull.simplices, hull.equations):
        triangle = points[face].copy()
        if np.dot(np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0]), equation[:3]) < 0:
            triangle[[1, 2]] = triangle[[2, 1]]
        vertices.extend(triangle)
        normals.extend([equation[:3]] * 3)
        uv.extend(np.column_stack((triangle[:, 0] / 0.8 + 0.5, triangle[:, 2] / 0.8 + 0.62)))
    ET.SubElement(
        asset,
        "mesh",
        name=name,
        vertex=array_to_string(np.asarray(vertices).ravel()),
        normal=array_to_string(np.asarray(normals).ravel()),
        face=array_to_string(np.arange(len(vertices))),
        texcoord=array_to_string(np.asarray(uv).ravel()),
    )
    position = f"0 {y} 0.012"
    ET.SubElement(
        body, "geom", name=f"{name}_visual", type="mesh", mesh=name, pos=position, material="wine_rack_oak", **_VISUAL
    )
    ET.SubElement(body, "geom", name=name, type="box", size=array_to_string(size), pos=position, **_COLLISION)
    return name


def add_wine_rack(asset, body):
    """Return collision names and per-location, per-rail support names."""
    ET.SubElement(
        asset,
        "texture",
        name="wine_rack_grain",
        type="2d",
        file=xml_path_completion("textures/ambientcg_wood049_color_2k.png"),
    )
    ET.SubElement(
        asset,
        "material",
        name="wine_rack_oak",
        texture="wine_rack_grain",
        rgba="0.93 0.91 0.86 1",
        texrepeat="1 1",
        specular="0.10",
        shininess="0.16",
    )
    ET.SubElement(
        asset,
        "material",
        name="wine_rack_endgrain",
        texture="wine_rack_grain",
        rgba="0.76 0.73 0.67 1",
        texrepeat="1 1",
        specular="0.06",
        shininess="0.1",
    )
    names = [_runner(asset, body, f"wine_rack_runner_{side}", y) for side, y in enumerate((-0.154, 0.154))]
    supports = {location: {} for location in SLOT_Y_M}
    for rail, x, radius, center_z, shoulder in RAILS:
        _rounded_rail(asset, body, f"wine_rack_{rail}_visual", x, radius, center_z, shoulder)
        for location, center_y in SLOT_Y_M.items():
            supports[location][rail] = []
            # Convex strips follow the same cut as the visual mesh. The near-bottom
            # chord error is below 0.3 mm; more segments are unnecessary for contacts.
            edges = center_y + np.linspace(-0.05, 0.05, 21)
            heights = rail_height(edges, radius, center_z, shoulder)
            for segment in range(len(edges) - 1):
                name = f"wine_rack_{location}_{rail}_{segment}"
                vertices = [
                    [px, py, pz]
                    for px in (-RAIL_HALF_THICKNESS_M, RAIL_HALF_THICKNESS_M)
                    for py, top in zip(edges[segment : segment + 2], heights[segment : segment + 2])
                    for pz in (RAIL_BOTTOM_M, top)
                ]
                ET.SubElement(asset, "mesh", name=name, vertex=array_to_string(np.asarray(vertices).ravel()))
                ET.SubElement(body, "geom", name=name, type="mesh", mesh=name, pos=f"{x} 0 0", **_COLLISION)
                names.append(name)
                supports[location][rail].append(name)
        for side, y in enumerate((-0.158, 0.158)):
            # Solid end blocks join the rail to the runners outside the three cutouts.
            name = f"wine_rack_{rail}_end_{side}"
            ET.SubElement(
                body,
                "geom",
                name=name,
                type="box",
                pos=f"{x} {y} {(shoulder + RAIL_BOTTOM_M) / 2}",
                size=f"{RAIL_HALF_THICKNESS_M} 0.008 {(shoulder - RAIL_BOTTOM_M) / 2}",
                **_COLLISION,
            )
            names.append(name)
            # Small inset dowels express the join without conspicuous metal hardware.
            ET.SubElement(
                body,
                "geom",
                name=f"{name}_dowel",
                type="cylinder",
                pos=f"{x - RAIL_HALF_THICKNESS_M - 0.00005} {y} 0.035",
                size="0.0025 0.0001",
                quat="0.70710678 0 0.70710678 0",
                material="wine_rack_endgrain",
                **_VISUAL,
            )
    return names, supports
