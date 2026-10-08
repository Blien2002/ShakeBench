"""One flat water-bottle contact mesh within the existing horizontal envelope."""

import xml.etree.ElementTree as ET

import numpy as np

WATER_SUPPORT_PROFILE = "flat_sole_mesh_v1"
SOLE_RADIUS_M = 0.034
SOLE_HEIGHT_M = 0.002
SOLE_LOWER_EXTENSION_M = 0.0005
SOLE_SIDES = 32


def add_flat_water_sole(bottle, collision_bottom_z):
    """Register a massless collision proxy; visual meshes and fixed body inertia stay intact.

    Mesh encoding preserves the existing mesh-only envelope extraction/scorer.
    Robosuite has no public append-contact-geom API; register its unprefixed
    name once, while using the prefixed name in the already-prefixed XML.
    """
    if not np.isfinite(collision_bottom_z):
        raise ValueError("collision bottom must be finite")
    raw_name = "flat_sole"
    geom_name = bottle.naming_prefix + raw_name
    mesh_name = geom_name + "_mesh"
    if raw_name in bottle._contact_geoms or bottle.get_obj().find(f"geom[@name='{geom_name}']") is not None:
        raise ValueError("flat sole already registered")
    angles = np.arange(SOLE_SIDES) * 2 * np.pi / SOLE_SIDES
    vertices = [
        [SOLE_RADIUS_M * np.cos(angle), SOLE_RADIUS_M * np.sin(angle), z]
        for z in (-SOLE_HEIGHT_M / 2, SOLE_HEIGHT_M / 2)
        for angle in angles
    ]
    ET.SubElement(
        bottle.asset,
        "mesh",
        name=mesh_name,
        vertex=" ".join(format(value, ".17g") for point in vertices for value in point),
    )
    ET.SubElement(
        bottle.get_obj(),
        "geom",
        name=geom_name,
        type="mesh",
        mesh=mesh_name,
        pos=f"0 0 {collision_bottom_z - SOLE_LOWER_EXTENSION_M + SOLE_HEIGHT_M / 2:.17g}",
        mass="0",
        group="0",
        rgba="0 0 0 0",
        contype="0",
        conaffinity="0",
    )
    bottle._contact_geoms.append(raw_name)
    return geom_name
