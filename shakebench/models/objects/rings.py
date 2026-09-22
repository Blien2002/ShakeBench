"""Rounded visual rings with the existing hollow-cylinder collision geometry."""

import xml.etree.ElementTree as ET

import numpy as np

from robosuite.models.objects.composite import HollowCylinderObject
from robosuite.utils.mjcf_utils import array_to_string, xml_path_completion


def make_ring(name, *, outer_radius, inner_radius, half_height, segments, rgba):
    """Keep flat physical faces and replace overlapping box visuals with one mesh.

    Like MetaWorld's assembly ring, the detailed mesh has no collision or mass.
    Small rounded edges retain a broad flat top/bottom for stacking.
    """
    ring = HollowCylinderObject(
        name,
        outer_radius=outer_radius,
        inner_radius=inner_radius,
        height=half_height,
        ngeoms=segments,
        density=700,
        rgba=rgba,
    )
    body = ring.get_obj()
    for geom in list(body.findall("geom")):
        if geom.get("group") == "1":
            body.remove(geom)
    for site in body.iter("site"):
        site.set("rgba", "0 0 0 0")

    bevel = 0.0015
    profile, normals = [], []
    # Counterclockwise rounded rectangle in radial/vertical coordinates.
    for radius, height, start in (
        (outer_radius - bevel, -half_height + bevel, -90),
        (outer_radius - bevel, half_height - bevel, 0),
        (inner_radius + bevel, half_height - bevel, 90),
        (inner_radius + bevel, -half_height + bevel, 180),
    ):
        for angle in np.deg2rad(np.linspace(start, start + 90, 5)):
            normal = np.array([np.cos(angle), np.sin(angle)])
            profile.append(np.array([radius, height]) + bevel * normal)
            normals.append(normal)
    vertices, vertex_normals, faces = [], [], []
    slices, count = 96, len(profile)
    for i, angle in enumerate(np.linspace(0, 2 * np.pi, slices, endpoint=False)):
        cosine, sine = np.cos(angle), np.sin(angle)
        for j, ((radius, height), (radial, vertical)) in enumerate(zip(profile, normals)):
            vertices.append([radius * cosine, radius * sine, height])
            vertex_normals.append([radial * cosine, radial * sine, vertical])
            a, b = i * count + j, ((i + 1) % slices) * count + j
            c, d = ((i + 1) % slices) * count + (j + 1) % count, i * count + (j + 1) % count
            faces.extend([[a, b, c], [a, c, d]])
    mesh_name, material_name = f"{name}_rounded_mesh", f"{name}_satin"
    ET.SubElement(
        ring.asset,
        "mesh",
        name=mesh_name,
        vertex=array_to_string(np.asarray(vertices).ravel()),
        normal=array_to_string(np.asarray(vertex_normals).ravel()),
        face=array_to_string(np.asarray(faces).ravel()),
        texcoord=array_to_string((np.asarray(vertices)[:, :2] / (2 * outer_radius) + 0.5).ravel()),
    )
    ET.SubElement(
        ring.asset, "texture", name=f"{name}_wood", type="2d", file=xml_path_completion("textures/light-wood.png")
    )
    ET.SubElement(
        ring.asset,
        "material",
        name=material_name,
        rgba=array_to_string(rgba),
        texture=f"{name}_wood",
        texrepeat="1 1",
        specular="0.12",
        shininess="0.15",
    )
    ET.SubElement(
        body,
        "geom",
        name=f"{name}_rounded_visual",
        type="mesh",
        mesh=mesh_name,
        material=material_name,
        group="1",
        contype="0",
        conaffinity="0",
        mass="0",
    )
    # Keep robosuite's object segmentation metadata in sync with the replaced geoms.
    ring._visual_geoms = ["rounded_visual"]
    return ring
