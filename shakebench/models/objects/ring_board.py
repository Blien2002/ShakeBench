"""A wooden storage board with three real shallow circular pockets."""

import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial import Delaunay

from robosuite.utils.mjcf_utils import array_to_string, xml_path_completion

BOARD_RADIUS_M = 0.200
BOARD_HEIGHT_M = 0.012
SLOT_DEPTH_M = 0.004
SLOT_CLEARANCE_M = 0.008


def add_ring_board(asset, body, slots, contact_attributes):
    """Attach a fixed board whose top is z=0; return its collision geom names.

    The top is triangulated around the pockets. Each triangle is a convex prism
    for collision, so MuJoCo cannot close the holes with a single convex hull.
    """
    ET.SubElement(
        asset,
        "texture",
        name="ring_wood_texture",
        type="2d",
        file=xml_path_completion("textures/light-wood.png"),
    )
    ET.SubElement(
        asset,
        "material",
        name="ring_wood",
        texture="ring_wood_texture",
        texrepeat="1 1",
        specular="0.12",
        shininess="0.15",
    )
    names = ["ring_board_floor_collision"]
    floor_height = BOARD_HEIGHT_M - SLOT_DEPTH_M
    for visual in (False, True):
        ET.SubElement(
            body,
            "geom",
            name="ring_board_floor_visual" if visual else names[0],
            type="cylinder",
            size=f"{BOARD_RADIUS_M} {floor_height / 2}",
            pos=f"0 0 {-SLOT_DEPTH_M - floor_height / 2}",
            material="ring_wood",
            mass="0",
            group=str(int(visual)),
            contype="0" if visual else "1",
            conaffinity="0" if visual else "1",
            **({} if visual else contact_attributes),
        )
    loops = []
    for center, radius, count in [(np.zeros(2), BOARD_RADIUS_M, 96), *[(xy, r, 48) for xy, r in slots]]:
        angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
        loops.append(np.asarray(center) + radius * np.column_stack((np.cos(angles), np.sin(angles))))
    points = np.concatenate(loops)
    triangles = Delaunay(points).simplices
    centers = points[triangles].mean(axis=1)
    keep = np.ones(len(triangles), dtype=bool)
    for xy, radius in slots:
        keep &= np.linalg.norm(centers - xy, axis=1) > radius
    triangles = triangles[keep]
    # Ensure triangulation preserved every pocket boundary, including custom states.
    edges = {tuple(sorted((int(a), int(b)))) for tri in triangles for a, b in zip(tri, np.roll(tri, -1))}
    start = 0
    for loop in loops:
        if any(tuple(sorted((start + i, start + (i + 1) % len(loop)))) not in edges for i in range(len(loop))):
            raise ValueError("storage slots are too close to triangulate the board")
        start += len(loop)
    vertices = np.concatenate(
        (
            np.column_stack((points, np.zeros(len(points)))),
            np.column_stack((points, np.full(len(points), -SLOT_DEPTH_M))),
        )
    )
    faces = triangles.tolist()
    start, count = 0, len(points)
    for loop_index, loop in enumerate(loops):
        for i in range(len(loop)):
            a, b = start + i, start + (i + 1) % len(loop)
            walls = [[a, a + count, b + count], [a, b + count, b]]
            faces.extend(walls if loop_index == 0 else [face[::-1] for face in walls])
        start += len(loop)
    ET.SubElement(
        asset,
        "mesh",
        name="ring_board_surface_mesh",
        vertex=array_to_string(vertices.ravel()),
        face=array_to_string(np.asarray(faces).ravel()),
        texcoord=array_to_string((vertices[:, :2] / (2 * BOARD_RADIUS_M) + 0.5).ravel()),
    )
    ET.SubElement(
        body,
        "geom",
        name="ring_board_surface_visual",
        type="mesh",
        mesh="ring_board_surface_mesh",
        material="ring_wood",
        group="1",
        mass="0",
        contype="0",
        conaffinity="0",
    )
    for index, tri in enumerate(triangles):
        name = f"ring_board_surface_{index}"
        ET.SubElement(asset, "mesh", name=name, vertex=array_to_string(vertices[np.r_[tri, tri + count]].ravel()))
        names.append(name + "_collision")
        ET.SubElement(
            body,
            "geom",
            name=names[-1],
            type="mesh",
            mesh=name,
            group="0",
            mass="0",
            **contact_attributes,
        )
    return names
