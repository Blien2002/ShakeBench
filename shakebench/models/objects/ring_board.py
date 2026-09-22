"""A compact solid wooden base for the ring-stacking peg."""

import xml.etree.ElementTree as ET

from robosuite.utils.mjcf_utils import xml_path_completion

BOARD_RADIUS_M = 0.062
BOARD_HEIGHT_M = 0.012


def add_ring_board(asset, body, contact_attributes):
    """Attach one fixed cylinder with its top at z=0; return collision names."""
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
    floor_height = BOARD_HEIGHT_M
    for visual in (False, True):
        ET.SubElement(
            body,
            "geom",
            name="ring_board_floor_visual" if visual else names[0],
            type="cylinder",
            size=f"{BOARD_RADIUS_M} {floor_height / 2}",
            pos=f"0 0 {-floor_height / 2}",
            material="ring_wood",
            mass="0",
            group=str(int(visual)),
            contype="0" if visual else "1",
            conaffinity="0" if visual else "1",
            **({} if visual else contact_attributes),
        )
    return names
