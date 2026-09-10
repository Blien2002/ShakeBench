"""Visual-only ShakeBench arena assembly.

The helpers in this module append display geometry only.  Runtime scene
validation and collision checks remain in ``shakebench_scene``.
"""

from __future__ import annotations

import itertools
import xml.etree.ElementTree as ET
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from robosuite import models
from robosuite.utils.shakebench_scene import (
    DECK_VISUAL_BODY_NAME,
    TABLE_VISUAL_GEOM_NAME,
    SceneConfigError,
    SceneVisualConfig,
    _add_camera,
    _append_unique_geom,
    _configure_materials,
    _fmt,
    _geom_name_set,
    _get_or_append_body,
    _quat_from_z_axis,
    _rgba,
    _stewart_points,
    _table_support_geometry,
    _visual_geom,
)


WORLD_VISUALS_FILENAME = "arenas/shakebench_world_visuals.xml"
DIRECT_WORLD_VISUALS_FILENAME = "arenas/shakebench_direct_mount_world_additions.xml"


def _load_world_visuals(arena: Any, config: SceneVisualConfig) -> None:
    """Expand the fixed world MJCF include for Python-side inventory setup."""

    if getattr(arena, "_shakebench_world_visuals_loaded", False):
        return
    for include in tuple(arena.root.findall("./include[@file='shakebench_world_visuals.xml']")):
        arena.root.remove(include)
    # Each profile needs its complete world, not canonical bodies plus additions.
    filenames = [DIRECT_WORLD_VISUALS_FILENAME if config.geometry_variant == "C" else WORLD_VISUALS_FILENAME]
    for filename in filenames:
        source = Path(models.assets_root) / filename
        try:
            root = ET.parse(source).getroot()
        except (OSError, ET.ParseError) as exc:
            raise SceneConfigError(f"cannot load fixed world visuals: {exc}") from exc
        source_asset = root.find("asset")
        for element in () if source_asset is None else source_asset:
            if arena.asset.find(f"./{element.tag}[@name='{element.get('name')}']") is None:
                arena.asset.append(deepcopy(element))
        source_worldbody = root.find("worldbody")
        for body in () if source_worldbody is None else source_worldbody:
            if arena.worldbody.find(f"./body[@name='{body.get('name')}']") is None:
                arena.worldbody.append(deepcopy(body))
    arena._shakebench_world_visuals_loaded = True

def _build_table_finish(arena, platen_body, table, envelope, platen_center_z, geom_names):
    """Author two-part isolator covers and bolted, non-contact table details.

    Rubber boots and anchors belong to the deck; the short insert, cap and
    frame hardware belong to the isolated table. The insert is concealed
    inside the boot at rest, leaving overlap for relative motion.
    """
    names = []
    steel = (0.43, 0.47, 0.50, 1.0)
    frame = (0.12, 0.155, 0.18, 1.0)
    rubber = (0.075, 0.082, 0.09, 1.0)
    dark = (0.035, 0.041, 0.045, 1.0)
    base_size = np.asarray(table["lower_plate_size_m"], dtype=float)
    cap_size = np.asarray(table["foot_plate_size_m"], dtype=float)
    cap_bottom = envelope["foot_center_world_z"] - cap_size[2] / 2
    base_top = envelope["support_z"] + base_size[2]
    boot_top = cap_bottom - 0.012
    boot_height = boot_top - base_top
    if boot_height <= 0:
        raise SceneConfigError("table isolator boot requires clearance between its base plate and upper cap")
    boot_radius = min(float(table["lower_sleeve_radius_m"]), min(base_size[:2]) * 0.43)

    def emit(parent, name, kind, size, pos, rgba=steel, axis=None, material="shakebench_frame_metal"):
        geom = _visual_geom(
            name,
            kind,
            size,
            pos,
            rgba=rgba,
            material=material,
            quat=None if axis is None else _quat_from_z_axis(axis),
        )
        _append_unique_geom(parent, geom, geom_names)
        names.append(name)

    def bolt(parent, name, pos, axis=(0, 0, 1), radius=0.004):
        pos, axis = np.asarray(pos), np.asarray(axis)
        emit(parent, name + "_washer", "cylinder", (radius * 1.45, 0.0007), pos, axis=axis)
        emit(parent, name, "cylinder", (radius, 0.0016), pos + axis * 0.002, axis=axis)
        emit(parent, name + "_socket", "cylinder", (radius * 0.48, 0.00015), pos + axis * 0.0037, dark, axis)

    # A legible powder coat on the existing square-tube frame.
    for geom in arena.table_body.findall("./geom"):
        name = geom.get("name", "")
        if name.startswith("shakebench_table_upper_"):
            geom.set("rgba", _fmt(frame))

    for index, (x, y) in enumerate(envelope["leg_centers_xy"]):
        deck_x, deck_y = x + arena.center_pos[0], y + arena.center_pos[1]
        lower = "shakebench_table_lower_mount_"
        upper = "shakebench_table_upper_"
        base = platen_body.find(f"./geom[@name='{lower}plate_{index}']")
        base.set("rgba", _fmt((0.25, 0.28, 0.31, 1.0)))
        core = platen_body.find(f"./geom[@name='{lower}sleeve_{index}']")
        core.set("pos", _fmt((deck_x, deck_y, (base_top + boot_top) / 2 - platen_center_z)))
        core.set("size", _fmt((boot_radius * 0.76, boot_height / 2)))
        core.set("rgba", _fmt(rubber))
        for rib in range(5):
            z = base_top + boot_height * (rib + 0.5) / 5 - platen_center_z
            emit(
                platen_body,
                f"{lower}boot_rib_{index}_{rib}",
                "ellipsoid",
                (boot_radius, boot_radius, boot_height / 8),
                (deck_x, deck_y, z),
                rubber,
                material="shakebench_mount_rubber",
            )
        for end, z in (("bottom", base_top + 0.002), ("top", boot_top - 0.002)):
            emit(
                platen_body,
                f"{lower}boot_band_{index}_{end}",
                "cylinder",
                (boot_radius * 0.87, 0.002),
                (deck_x, deck_y, z - platen_center_z),
                dark,
            )
        for corner, (sx, sy) in enumerate(itertools.product((-1, 1), repeat=2)):
            bolt(
                platen_body,
                f"{lower}anchor_{index}_{corner}",
                (
                    deck_x + sx * (base_size[0] / 2 - 0.008),
                    deck_y + sy * (base_size[1] / 2 - 0.008),
                    base_top - platen_center_z + 0.0007,
                ),
            )
        cap = arena.table_body.find(f"./geom[@name='{upper}foot_{index}']")
        cap.set("rgba", _fmt((0.29, 0.33, 0.36, 1.0)))
        insert = arena.table_body.find(f"./geom[@name='{upper}sleeve_{index}']")
        insert.set("pos", _fmt((x, y, cap_bottom - 0.012 - envelope["table_world_z"])))
        insert.set("size", _fmt((boot_radius * 0.62, 0.012)))
        insert.set("material", "shakebench_bolt_metal")
        insert.set("rgba", _fmt(steel))
        for corner, (sx, sy) in enumerate(itertools.product((-1, 1), repeat=2)):
            bolt(
                arena.table_body,
                f"{upper}cap_bolt_{index}_{corner}",
                (
                    x + sx * (cap_size[0] / 2 - 0.008),
                    y + sy * (cap_size[1] / 2 - 0.007),
                    cap_bottom + cap_size[2] - envelope["table_world_z"] + 0.0007,
                ),
                radius=0.003,
            )
        # Flush cheek plates on two outside faces of each leg, below the apron.
        for axis_index, sign in ((0, np.sign(x)), (1, np.sign(y))):
            pos = np.array((x, y, -0.095))
            pos[axis_index] += sign * 0.0265
            size = np.array((0.023, 0.023, 0.034))
            size[axis_index] = 0.0015
            prefix = f"{upper}joint_plate_{index}_{axis_index}"
            emit(arena.table_body, prefix, "box", size, pos, (0.22, 0.26, 0.29, 1.0))
            axis = np.eye(3)[axis_index] * sign
            for row, z in enumerate((-0.079, -0.112)):
                bolt_pos = pos + axis * 0.0022
                bolt_pos[2] = z
                bolt(arena.table_body, f"{prefix}_bolt_{row}", bolt_pos, axis, radius=0.0035)

    hx, hy, hz = envelope["table_half"]
    # Slim protective strips sit on the sides of the slab, below its task face.
    for axis_index, half in ((0, hx), (1, hy)):
        for sign in (-1, 1):
            # The canonical pedestal meets the slab's left edge exactly;
            # leave that installation interface flush without added trim.
            if table["layout_variant"] == "A" and axis_index == 0 and sign == -1:
                continue
            pos = np.array((0.0, 0.0, -hz + 0.008))
            pos[axis_index] = sign * (half + 0.001)
            size = np.array((hx, hy, 0.007))
            size[axis_index] = 0.001
            emit(arena.table_body, f"shakebench_table_upper_edge_{axis_index}_{sign}", "box", size, pos, steel)
    # A recessed-looking identification plate on the front apron, with rivets.
    front_y = -hy - float(table["frame_y_offset_m"]) - float(table["frame_half_thickness_m"])
    emit(
        arena.table_body,
        "shakebench_table_upper_badge",
        "box",
        (0.039, 0.001, 0.012),
        (0, front_y - 0.001, -0.055),
        steel,
    )
    emit(
        arena.table_body,
        "shakebench_table_upper_badge_inset",
        "box",
        (0.028, 0.0003, 0.008),
        (0, front_y - 0.0022, -0.055),
        dark,
    )
    for x in (-0.034, 0.034):
        bolt(
            arena.table_body,
            f"shakebench_table_upper_badge_rivet_{x}",
            (x, front_y - 0.002, -0.055),
            (0, -1, 0),
            0.0015,
        )
    return names




def build_scene_visuals(
    arena: Any, config: SceneVisualConfig
) -> tuple[dict[str, Any], tuple[str, ...], tuple[str, ...]]:
    """Append the configured scene and return body/geom inventory primitives."""

    _load_world_visuals(arena, config)
    worldbody = arena.worldbody
    geom_names = _geom_name_set(arena.root)
    materials = config.section("materials")
    platen = config.section("platen")
    table = config.section("table_support")
    _configure_materials(arena, config)
    envelope = _table_support_geometry(arena, config)
    frame_rgb = _rgba("materials.frame_rgba", (*materials["frame_rgb"], 1.0))
    rubber_rgb = _rgba("materials.rubber_rgba", (*materials["rubber_rgb"], 1.0))
    bolt_rgb = _rgba("materials.bolt_rgba", (*materials["bolt_rgb"], 1.0))

    visual_geoms: list[str] = [TABLE_VISUAL_GEOM_NAME]
    visual_bodies: list[str] = []
    frame_bodies: dict[str, str] = {}

    # The physical plane remains in the model and keeps its contact semantics;
    # its alpha is zero so the four display-only slabs can show a pit opening.
    physical_floor = worldbody.find("./geom[@name='floor']")
    if physical_floor is not None:
        physical_floor.set("rgba", "1 1 1 0")
        physical_floor.set("group", "0")

    world_bodies = config.section("frame_ownership")["world"]["bodies"]
    for name in world_bodies:
        body = worldbody.find(f"./body[@name='{name}']")
        if body is None:
            raise SceneConfigError(f"prebuilt world visual body is missing: {name}")
        frame_bodies[name] = "world"
        visual_bodies.append(name)
        visual_geoms.extend(geom.get("name") for geom in body.iter("geom") if geom.get("name"))

    platen_center_z = float(platen["nominal_top_z_m"]) - float(platen["size_m"][-1]) / 2.0
    stewart = config.section("stewart")
    base_points, platen_points = _stewart_points(config, platen_center_z)
    lengths = np.linalg.norm(platen_points - base_points, axis=1)
    if np.any(lengths < float(stewart["leg_min_m"])) or np.any(lengths > float(stewart["leg_max_m"])):
        raise SceneConfigError(f"nominal Stewart lengths are outside configured limits: {lengths.tolist()}")
    rod_length = float(stewart["rod_length_m"])

    # The platen and all deck-side lower mounts are one explicit role body.
    # Reparenting this body by the existing deck processor makes every visual
    # follow the ordinary dynamic deck without creating a kinematic parent.
    platen_body = _get_or_append_body(worldbody, DECK_VISUAL_BODY_NAME, (0.0, 0.0, platen_center_z))
    platen_body.set("pos", _fmt((0.0, 0.0, platen_center_z)))
    visual_bodies.append(DECK_VISUAL_BODY_NAME)
    frame_bodies[DECK_VISUAL_BODY_NAME] = "dynamic_deck"
    platen_half = np.asarray(platen["size_m"], dtype=float) / 2.0
    for name, size, pos, material in (
        ("shakebench_platen_surface", platen_half, (0.0, 0.0, 0.0), "shakebench_platen_surface"),
        (
            "shakebench_platen_edge",
            (platen_half[0], platen_half[1], 0.008),
            (0.0, 0.0, -platen_half[2] - 0.008),
            "shakebench_platen_edge",
        ),
    ):
        _append_unique_geom(
            platen_body, _visual_geom(name, "box", size, pos, material=material, rgba=platen["rgba"]), geom_names
        )
        visual_geoms.append(name)
    # Small top fasteners provide the visual cue of a laboratory shaker plate.
    for index, (x, y) in enumerate(((-0.65, -0.42), (-0.65, 0.42), (0.65, -0.42), (0.65, 0.42))):
        name = f"shakebench_platen_bolt_{index}"
        _append_unique_geom(
            platen_body,
            _visual_geom(
                name,
                "cylinder",
                (0.012, 0.003),
                (x, y, platen_half[2] + 0.003),
                material="shakebench_bolt_metal",
                rgba=bolt_rgb,
            ),
            geom_names,
        )
        visual_geoms.append(name)
    lower_plate_size = np.asarray(table["lower_plate_size_m"], dtype=float)
    upper_plate_size = np.asarray(table["foot_plate_size_m"], dtype=float)
    for index, (x, y) in enumerate(envelope["leg_centers_xy"]):
        # Deck-side plates use world XY; upper supports remain worktable-local.
        x += float(arena.center_pos[0])
        y += float(arena.center_pos[1])
        lower_plate_name = f"shakebench_table_lower_mount_plate_{index}"
        lower_plate_z = float(table["support_plane_z_m"]) + lower_plate_size[2] / 2.0 - platen_center_z
        _append_unique_geom(
            platen_body,
            _visual_geom(
                lower_plate_name,
                "box",
                lower_plate_size / 2.0,
                (x, y, lower_plate_z),
                material="shakebench_frame_metal",
            ),
            geom_names,
        )
        visual_geoms.append(lower_plate_name)
        lower_sleeve_name = f"shakebench_table_lower_mount_sleeve_{index}"
        lower_sleeve_z = float(table["lower_sleeve_center_z_m"]) - platen_center_z
        _append_unique_geom(
            platen_body,
            _visual_geom(
                lower_sleeve_name,
                "cylinder",
                (float(table["lower_sleeve_radius_m"]), float(table["lower_sleeve_half_height_m"])),
                (x, y, lower_sleeve_z),
                material="shakebench_mount_rubber",
                rgba=rubber_rgb,
            ),
            geom_names,
        )
        visual_geoms.append(lower_sleeve_name)
    for index, (point, platen_point) in enumerate(zip(base_points, platen_points)):
        direction = (platen_point - point) / lengths[index]
        rod_start = platen_point - rod_length * direction
        rod_name = f"shakebench_stewart_rod_{index}"
        rod_world_centre = 0.5 * (rod_start + platen_point)
        _append_unique_geom(
            platen_body,
            _visual_geom(
                rod_name,
                "cylinder",
                (float(stewart["rod_radius_m"]), rod_length / 2.0),
                rod_world_centre - np.array((0.0, 0.0, platen_center_z)),
                material="shakebench_actuator_rod",
                quat=_quat_from_z_axis(direction),
            ),
            geom_names,
        )
        visual_geoms.append(rod_name)
        joint_name = f"shakebench_stewart_rod_joint_{index}"
        _append_unique_geom(
            platen_body,
            _visual_geom(
                joint_name,
                "sphere",
                (float(stewart["joint_radius_m"]),),
                platen_point - np.array((0.0, 0.0, platen_center_z)),
                material="shakebench_joint",
            ),
            geom_names,
        )
        visual_geoms.append(joint_name)
    # The isolated worktable owns only its upper frame, legs and the upper
    # half of the sleeves.  No geom spans from this body into the deck body.
    frame_half = float(table["frame_half_thickness_m"])
    frame_y = float(table["frame_y_offset_m"])
    frame_specs = (
        (
            "shakebench_table_upper_frame_front",
            (envelope["table_half"][0], frame_half, frame_half),
            (0.0, -envelope["table_half"][1] - frame_y, -0.055),
        ),
        (
            "shakebench_table_upper_frame_back",
            (envelope["table_half"][0], frame_half, frame_half),
            (0.0, envelope["table_half"][1] + frame_y, -0.055),
        ),
        (
            "shakebench_table_upper_frame_left",
            (frame_half, envelope["table_half"][1] - frame_y, frame_half),
            (-envelope["table_half"][0] + float(table["frame_x_inset_m"]), 0.0, -0.055),
        ),
        (
            "shakebench_table_upper_frame_right",
            (frame_half, envelope["table_half"][1] - frame_y, frame_half),
            (envelope["table_half"][0] - float(table["frame_x_inset_m"]), 0.0, -0.055),
        ),
    )
    for name, size, pos in frame_specs:
        _append_unique_geom(
            arena.table_body,
            _visual_geom(name, "box", size, pos, material="shakebench_frame_metal", rgba=frame_rgb),
            geom_names,
        )
        visual_geoms.append(name)
    # Keep the lower stretcher attached to the legs above the isolator caps.
    crossbrace_z = max(
        float(table["crossbrace_z_local_m"]),
        envelope["foot_center_world_z"]
        + upper_plate_size[2] / 2
        + 2 * float(table["crossbrace_half_thickness_m"])
        - envelope["table_world_z"],
    )
    for name, size, pos in (
        (
            "shakebench_table_upper_crossbrace_front",
            (
                float(table["crossbrace_half_length_x_m"]),
                float(table["crossbrace_half_thickness_m"]),
                float(table["crossbrace_half_thickness_m"]),
            ),
            (0.0, -float(table["crossbrace_half_length_y_m"]), crossbrace_z),
        ),
        (
            "shakebench_table_upper_crossbrace_back",
            (
                float(table["crossbrace_half_length_x_m"]),
                float(table["crossbrace_half_thickness_m"]),
                float(table["crossbrace_half_thickness_m"]),
            ),
            (0.0, float(table["crossbrace_half_length_y_m"]), crossbrace_z),
        ),
        (
            "shakebench_table_upper_crossbrace_left",
            (
                float(table["crossbrace_half_thickness_m"]),
                float(table["crossbrace_half_length_y_m"]),
                float(table["crossbrace_half_thickness_m"]),
            ),
            (-float(table["crossbrace_half_length_x_m"]), 0.0, crossbrace_z),
        ),
        (
            "shakebench_table_upper_crossbrace_right",
            (
                float(table["crossbrace_half_thickness_m"]),
                float(table["crossbrace_half_length_y_m"]),
                float(table["crossbrace_half_thickness_m"]),
            ),
            (float(table["crossbrace_half_length_x_m"]), 0.0, crossbrace_z),
        ),
    ):
        _append_unique_geom(
            arena.table_body,
            _visual_geom(name, "box", size, pos, material="shakebench_frame_metal", rgba=frame_rgb),
            geom_names,
        )
        visual_geoms.append(name)
    for index, (x, y) in enumerate(envelope["leg_centers_xy"]):
        leg_name = f"shakebench_table_upper_leg_{index}"
        local_z = envelope["leg_center_world_z"] - envelope["table_world_z"]
        _append_unique_geom(
            arena.table_body,
            _visual_geom(
                leg_name,
                "box",
                (0.025, 0.025, envelope["leg_half_height"]),
                (x, y, local_z),
                material="shakebench_frame_metal",
                rgba=frame_rgb,
            ),
            geom_names,
        )
        visual_geoms.append(leg_name)
        foot_name = f"shakebench_table_upper_foot_{index}"
        foot_z = envelope["foot_center_world_z"] - envelope["table_world_z"]
        _append_unique_geom(
            arena.table_body,
            _visual_geom(
                foot_name,
                "box",
                upper_plate_size / 2.0,
                (x, y, foot_z),
                material="shakebench_frame_metal",
                rgba=frame_rgb,
            ),
            geom_names,
        )
        visual_geoms.append(foot_name)
        sleeve_name = f"shakebench_table_upper_sleeve_{index}"
        sleeve_z = float(table["upper_sleeve_center_z_m"]) - envelope["table_world_z"]
        _append_unique_geom(
            arena.table_body,
            _visual_geom(
                sleeve_name,
                "cylinder",
                (float(table["upper_sleeve_radius_m"]), float(table["upper_sleeve_half_height_m"])),
                (x, y, sleeve_z),
                material="shakebench_mount_rubber",
                rgba=rubber_rgb,
            ),
            geom_names,
        )
        visual_geoms.append(sleeve_name)

    visual_geoms.extend(_build_table_finish(arena, platen_body, table, envelope, platen_center_z, geom_names))

    for camera in config.section("cameras").values():
        _add_camera(worldbody, camera)
    for camera in config.section("render").get("detail_cameras", {}).values():
        _add_camera(worldbody, camera)

    return frame_bodies, tuple(dict.fromkeys(visual_geoms)), tuple(dict.fromkeys(visual_bodies))
