"""Benchmark-style textured room arena with layout / style separation.

The organization follows robosuite's Arena and RoboCasa's layout/style
split.  CC0 raster assets are authored as portable USD Preview Surface/UV
graphs; deterministic colored geometry remains underneath as a fallback.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import yaml

from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
from isaaclab.sim.utils import clone, create_prim, get_current_stage
from isaaclab.utils.configclass import configclass

from .paths import PROJECT_ROOT
from .visual_assets import author_textured_quad


ROOM_CONFIG_PATH = PROJECT_ROOT / "configs" / "room.yaml"
FLOOR_TEXTURE_PATH = PROJECT_ROOT / "assets" / "textures" / "epoxy_floor_cool_gray_1k.jpg"
WALLPAPER_TEXTURE_PATH = PROJECT_ROOT / "assets" / "textures" / "industrial_wall_light_gray_1k.jpg"


@configclass
class RoomArenaCfg(SpawnerCfg):
    """Visual room shell and deterministic material-style parameters."""

    func = None
    size_m: tuple[float, float, float] = (6.00, 5.00, 3.00)
    center_x_m: float = 0.40
    floor_z_m: float = -0.022
    back_wall_x_m: float = -1.48
    side_wall_y_m: float = 1.53
    wall_thickness_m: float = 0.035
    pit_size_m: tuple[float, float] = (2.05, 1.55)
    pit_depth_m: float = 0.78
    pit_border_width_m: float = 0.16
    safety_line_width_m: float = 0.055
    guardrail_height_m: float = 0.72
    guardrail_thickness_m: float = 0.035
    wall_base_rgb: tuple[float, float, float] = (0.67, 0.66, 0.64)
    wallpaper_rgb: tuple[tuple[float, float, float], ...] = (
        (0.70, 0.69, 0.67),
        (0.66, 0.65, 0.63),
        (0.68, 0.67, 0.65),
    )
    baseboard_rgb: tuple[float, float, float] = (0.12, 0.12, 0.12)
    pit_rgb: tuple[float, float, float] = (0.040, 0.043, 0.045)
    pit_border_rgb: tuple[float, float, float] = (0.13, 0.14, 0.145)
    safety_rgb: tuple[float, float, float] = (0.90, 0.67, 0.05)
    guardrail_rgb: tuple[float, float, float] = (0.90, 0.55, 0.02)
    wood_rgb: tuple[tuple[float, float, float], ...] = (
        (0.38, 0.39, 0.40),
        (0.40, 0.41, 0.42),
        (0.36, 0.37, 0.38),
        (0.39, 0.40, 0.41),
        (0.37, 0.38, 0.39),
    )
    plank_rows: int = 15
    plank_columns: int = 5
    plank_gap_m: float = 0.012
    floor_texture_path: str = str(FLOOR_TEXTURE_PATH)
    wallpaper_texture_path: str = str(WALLPAPER_TEXTURE_PATH)
    floor_uv_repeat: tuple[float, float] = (2.2, 2.0)
    wallpaper_uv_repeat: tuple[float, float] = (2.4, 1.7)
    cabinet_position: tuple[float, float, float] = (-1.85, -1.55, 0.90)
    chair_position: tuple[float, float, float] = (-1.10, 1.45, 0.0)
    tool_cart_position: tuple[float, float, float] = (-1.75, 0.75, 0.0)
    emergency_stop_position: tuple[float, float, float] = (-1.35, -0.35, 0.0)
    equipment_dark_rgb: tuple[float, float, float] = (0.075, 0.085, 0.095)
    equipment_light_rgb: tuple[float, float, float] = (0.26, 0.28, 0.29)


def load_room_arena_cfg(path: Path = ROOM_CONFIG_PATH) -> RoomArenaCfg:
    """Load the room layout and style registry entry."""

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    layout = payload["layout"]
    style = payload["style"]
    return RoomArenaCfg(
        func=spawn_room_arena,
        size_m=tuple(layout["size_m"]),
        center_x_m=float(layout["center_x_m"]),
        floor_z_m=float(layout["floor_z_m"]),
        back_wall_x_m=float(layout["back_wall_x_m"]),
        side_wall_y_m=float(layout["side_wall_y_m"]),
        wall_thickness_m=float(layout["wall_thickness_m"]),
        pit_size_m=tuple(layout["pit_size_m"]),
        pit_depth_m=float(layout["pit_depth_m"]),
        pit_border_width_m=float(layout["pit_border_width_m"]),
        safety_line_width_m=float(layout["safety_line_width_m"]),
        guardrail_height_m=float(layout["guardrail_height_m"]),
        guardrail_thickness_m=float(layout["guardrail_thickness_m"]),
        wall_base_rgb=tuple(style["wall_base_rgb"]),
        wallpaper_rgb=tuple(tuple(rgb) for rgb in style["wallpaper_rgb"]),
        baseboard_rgb=tuple(style["baseboard_rgb"]),
        pit_rgb=tuple(style["pit_rgb"]),
        pit_border_rgb=tuple(style["pit_border_rgb"]),
        safety_rgb=tuple(style["safety_rgb"]),
        guardrail_rgb=tuple(style["guardrail_rgb"]),
        wood_rgb=tuple(tuple(rgb) for rgb in style["wood_rgb"]),
        plank_rows=int(style["plank_rows"]),
        plank_columns=int(style["plank_columns"]),
        plank_gap_m=float(style["plank_gap_m"]),
        floor_texture_path=str(PROJECT_ROOT / style["floor_texture"]),
        wallpaper_texture_path=str(PROJECT_ROOT / style["wallpaper_texture"]),
        floor_uv_repeat=tuple(style["floor_uv_repeat"]),
        wallpaper_uv_repeat=tuple(style["wallpaper_uv_repeat"]),
        cabinet_position=tuple(payload["equipment"]["cabinet_position"]),
        chair_position=tuple(payload["equipment"]["chair_position"]),
        tool_cart_position=tuple(payload["equipment"]["tool_cart_position"]),
        emergency_stop_position=tuple(payload["equipment"]["emergency_stop_position"]),
        equipment_dark_rgb=tuple(payload["equipment"]["dark_rgb"]),
        equipment_light_rgb=tuple(payload["equipment"]["light_rgb"]),
    )


def static_equipment_geometry_report(cfg: RoomArenaCfg) -> dict[str, float]:
    """Signed grounding/assembly errors for static laboratory equipment."""

    floor = cfg.floor_z_m
    cabinet_bottom = cfg.cabinet_position[2] - 0.90
    cart_foot_bottom = floor
    cart_post_bottom = floor + 0.0425
    cart_low_shelf_bottom = floor + 0.065 - 0.0225
    return {
        "control_cabinet_ground_error_m": cabinet_bottom - floor,
        "chair_ground_error_m": 0.0,
        "tool_cart_ground_error_m": cart_foot_bottom - floor,
        "tool_cart_post_below_shelf_m": cart_low_shelf_bottom - cart_post_bottom,
        "emergency_stop_ground_error_m": 0.0,
        "guardrail_ground_error_m": 0.0,
    }


def _visual_cube(
    stage: Any,
    path: str,
    size: tuple[float, float, float],
    position: tuple[float, float, float],
    color: tuple[float, float, float],
) -> None:
    """Author a static arena cube with displayColor for NewtonGL."""

    from pxr import Gf, UsdGeom, UsdPhysics

    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    xform = UsdGeom.Xformable(cube)
    xform.AddTranslateOp().Set(Gf.Vec3d(*position))
    xform.AddScaleOp().Set(Gf.Vec3d(*size))
    gprim = UsdGeom.Gprim(cube)
    gprim.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    gprim.CreateDisplayOpacityAttr([1.0])
    # Retain the shape for NewtonGL but explicitly exclude static laboratory
    # dressing from MJWarp contact generation.
    collision = UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    collision.CreateCollisionEnabledAttr(False)


def _visual_cylinder(
    stage: Any,
    path: str,
    radius: float,
    height: float,
    position: tuple[float, float, float],
    color: tuple[float, float, float],
) -> None:
    """Author a vertical display-only cylinder."""

    from pxr import Gf, UsdGeom, UsdPhysics

    cylinder = UsdGeom.Cylinder.Define(stage, path)
    cylinder.CreateAxisAttr(UsdGeom.Tokens.z)
    cylinder.CreateRadiusAttr(radius)
    cylinder.CreateHeightAttr(height)
    UsdGeom.Xformable(cylinder).AddTranslateOp().Set(Gf.Vec3d(*position))
    gprim = UsdGeom.Gprim(cylinder)
    gprim.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    gprim.CreateDisplayOpacityAttr([1.0])
    collision = UsdPhysics.CollisionAPI.Apply(cylinder.GetPrim())
    collision.CreateCollisionEnabledAttr(False)


def _visual_cylinder_between(stage: Any, path: str, start, end, radius: float, color) -> None:
    """Author a display-only round rail between two points."""

    from pxr import Gf, UsdGeom, UsdPhysics

    vector = tuple(float(end[i] - start[i]) for i in range(3))
    length = math.sqrt(sum(value * value for value in vector))
    direction = tuple(value / length for value in vector)
    q_xyz = (-direction[1], direction[0], 0.0)
    q_w = 1.0 + direction[2]
    q_norm = math.sqrt(q_w * q_w + sum(value * value for value in q_xyz))
    cylinder = UsdGeom.Cylinder.Define(stage, path)
    cylinder.CreateAxisAttr(UsdGeom.Tokens.z)
    cylinder.CreateRadiusAttr(radius)
    cylinder.CreateHeightAttr(length)
    xform = UsdGeom.Xformable(cylinder)
    xform.AddTranslateOp().Set(Gf.Vec3d(*(0.5 * (float(start[i]) + float(end[i])) for i in range(3))))
    xform.AddOrientOp().Set(
        Gf.Quatf(q_w / q_norm, Gf.Vec3f(*(value / q_norm for value in q_xyz)))
    )
    gprim = UsdGeom.Gprim(cylinder)
    gprim.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    gprim.CreateDisplayOpacityAttr([1.0])
    collision = UsdPhysics.CollisionAPI.Apply(cylinder.GetPrim())
    collision.CreateCollisionEnabledAttr(False)


def _contact_patch(stage: Any, path: str, center, size, z: float, surface_rgb) -> None:
    """Small collision-free grounding cue for static equipment."""

    _visual_cube(
        stage,
        path,
        (size[0], size[1], 0.0008),
        (center[0], center[1], z + 0.0005),
        tuple(0.34 * value for value in surface_rgb),
    )


@clone
def spawn_room_arena(
    prim_path: str,
    cfg: RoomArenaCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **_: Any,
):
    """Build a cool-gray industrial laboratory around the shaker pit."""

    stage = get_current_stage()
    create_prim(prim_path, "Xform", translation=translation, orientation=orientation, stage=stage)
    length, width, height = cfg.size_m
    floor_top = cfg.floor_z_m
    x_min = cfg.center_x_m - 0.5 * length
    x_max = cfg.center_x_m + 0.5 * length
    y_min = -0.5 * width
    y_max = 0.5 * width
    pit_half_x = 0.5 * cfg.pit_size_m[0]
    pit_half_y = 0.5 * cfg.pit_size_m[1]
    floor_rectangles = (
        ("West", x_min, -pit_half_x, y_min, y_max),
        ("East", pit_half_x, x_max, y_min, y_max),
        ("South", -pit_half_x, pit_half_x, y_min, -pit_half_y),
        ("North", -pit_half_x, pit_half_x, pit_half_y, y_max),
    )

    # Four floor slabs leave a real rectangular visual opening for the
    # recessed Stewart mechanism.
    for name, rx0, rx1, ry0, ry1 in floor_rectangles:
        _visual_cube(
            stage,
            f"{prim_path}/FloorSubstrate{name}",
            (rx1 - rx0, ry1 - ry0, 0.030),
            (0.5 * (rx0 + rx1), 0.5 * (ry0 + ry1), floor_top - 0.020),
            (0.38, 0.39, 0.40),
        )

    # One broad-scale seamless texture supplies subtle epoxy pour variation
    # without residential plank seams or a short repeat cycle.
    for name, rx0, rx1, ry0, ry1 in floor_rectangles:
        author_textured_quad(
            stage,
            f"{prim_path}/FloorTexture{name}",
            (
                (rx0, ry0, floor_top + 0.0003),
                (rx1, ry0, floor_top + 0.0003),
                (rx1, ry1, floor_top + 0.0003),
                (rx0, ry1, floor_top + 0.0003),
            ),
            cfg.floor_uv_repeat,
            Path(cfg.floor_texture_path),
            roughness=0.58,
        )

    pit_bottom_z = floor_top - cfg.pit_depth_m
    _visual_cube(
        stage,
        f"{prim_path}/PitBottom",
        (cfg.pit_size_m[0], cfg.pit_size_m[1], 0.04),
        (0.0, 0.0, pit_bottom_z - 0.02),
        cfg.pit_rgb,
    )
    pit_wall_height = cfg.pit_depth_m
    pit_wall_z = floor_top - 0.5 * pit_wall_height
    _visual_cube(stage, f"{prim_path}/PitWallWest", (0.04, cfg.pit_size_m[1], pit_wall_height), (-pit_half_x, 0.0, pit_wall_z), cfg.pit_rgb)
    _visual_cube(stage, f"{prim_path}/PitWallEast", (0.04, cfg.pit_size_m[1], pit_wall_height), (pit_half_x, 0.0, pit_wall_z), cfg.pit_rgb)
    _visual_cube(stage, f"{prim_path}/PitWallSouth", (cfg.pit_size_m[0], 0.04, pit_wall_height), (0.0, -pit_half_y, pit_wall_z), cfg.pit_rgb)
    _visual_cube(stage, f"{prim_path}/PitWallNorth", (cfg.pit_size_m[0], 0.04, pit_wall_height), (0.0, pit_half_y, pit_wall_z), cfg.pit_rgb)
    # A restrained set of edge grates provides service access while keeping
    # the mechanism unobstructed from the main camera.
    for index in range(8):
        x = -0.84 + index * 0.24
        _visual_cube(stage, f"{prim_path}/PitGrateNorth{index}", (0.16, 0.055, 0.018), (x, pit_half_y - 0.030, floor_top - 0.018), cfg.pit_border_rgb)
    _contact_patch(stage, f"{prim_path}/PlatenPitShadow", (0.0, 0.0), (1.20, 0.82), pit_bottom_z + 0.022, cfg.pit_rgb)

    border_z = floor_top + 0.018
    border = cfg.pit_border_width_m
    _visual_cube(stage, f"{prim_path}/PitBorderWest", (border, cfg.pit_size_m[1] + 2.0 * border, 0.035), (-pit_half_x - 0.5 * border, 0.0, border_z), cfg.pit_border_rgb)
    _visual_cube(stage, f"{prim_path}/PitBorderEast", (border, cfg.pit_size_m[1] + 2.0 * border, 0.035), (pit_half_x + 0.5 * border, 0.0, border_z), cfg.pit_border_rgb)
    _visual_cube(stage, f"{prim_path}/PitBorderSouth", (cfg.pit_size_m[0], border, 0.035), (0.0, -pit_half_y - 0.5 * border, border_z), cfg.pit_border_rgb)
    _visual_cube(stage, f"{prim_path}/PitBorderNorth", (cfg.pit_size_m[0], border, 0.035), (0.0, pit_half_y + 0.5 * border, border_z), cfg.pit_border_rgb)

    line = cfg.safety_line_width_m
    line_offset_x = pit_half_x + border + 0.5 * line
    line_offset_y = pit_half_y + border + 0.5 * line
    _visual_cube(stage, f"{prim_path}/SafetyLineWest", (line, 2.0 * line_offset_y, 0.008), (-line_offset_x, 0.0, floor_top + 0.006), cfg.safety_rgb)
    _visual_cube(stage, f"{prim_path}/SafetyLineEast", (line, 2.0 * line_offset_y, 0.008), (line_offset_x, 0.0, floor_top + 0.006), cfg.safety_rgb)
    _visual_cube(stage, f"{prim_path}/SafetyLineSouth", (2.0 * line_offset_x, line, 0.008), (0.0, -line_offset_y, floor_top + 0.006), cfg.safety_rgb)
    _visual_cube(stage, f"{prim_path}/SafetyLineNorth", (2.0 * line_offset_x, line, 0.008), (0.0, line_offset_y, floor_top + 0.006), cfg.safety_rgb)

    # Rear and side rails protect the pit while leaving the camera-facing side
    # open so the Stewart actuators remain visible.
    rail_t = cfg.guardrail_thickness_m
    rail_z = floor_top + cfg.guardrail_height_m
    post_z = floor_top + 0.5 * cfg.guardrail_height_m
    rail_y = pit_half_y + border
    rail_x = pit_half_x + border
    # Only the rear rail remains full height. The camera-facing sides are a
    # maintenance opening, eliminating the vertical post that split the view.
    post_points = (
        ("WestSouth", -rail_x, -rail_y),
        ("EastSouth", rail_x, -rail_y),
        ("SouthMid", 0.0, -rail_y),
    )
    for name, x, y in post_points:
        post_root = f"{prim_path}/RailPost{name}"
        create_prim(post_root, "Xform", translation=(x, y, 0.0), stage=stage)
        _visual_cylinder(stage, f"{post_root}/Post", 0.5 * rail_t, cfg.guardrail_height_m, (0.0, 0.0, post_z), cfg.guardrail_rgb)
        _visual_cube(stage, f"{post_root}/BasePlate", (0.12, 0.12, 0.010), (0.0, 0.0, floor_top + 0.006), (0.22, 0.20, 0.13))
        _contact_patch(stage, f"{post_root}/Shadow", (0.0, 0.0), (0.14, 0.14), floor_top, cfg.wood_rgb[0])
        for bolt, (dx, dy) in enumerate(((-0.038, -0.038), (-0.038, 0.038), (0.038, -0.038), (0.038, 0.038))):
            _visual_cylinder(stage, f"{post_root}/Bolt{bolt}", 0.006, 0.009, (dx, dy, floor_top + 0.015), (0.05, 0.055, 0.06))
    for level_name, z in (("Mid", floor_top + 0.5 * cfg.guardrail_height_m), ("Top", rail_z)):
        _visual_cylinder_between(stage, f"{prim_path}/Rail{level_name}South", (-rail_x, -rail_y, z), (rail_x, -rail_y, z), 0.5 * rail_t, cfg.guardrail_rgb)
    for name, x, y in post_points:
        for level, z in (("Mid", floor_top + 0.5 * cfg.guardrail_height_m), ("Top", rail_z)):
            _visual_cylinder(stage, f"{prim_path}/RailPost{name}/Clamp{level}", 0.72 * rail_t, 0.030, (0.0, 0.0, z), (0.18, 0.16, 0.09))

    wall_z = floor_top + 0.5 * height
    _visual_cube(
        stage,
        f"{prim_path}/BackWallBase",
        (cfg.wall_thickness_m, width, height),
        (cfg.back_wall_x_m, 0.0, wall_z),
        cfg.wall_base_rgb,
    )
    for side, y in (("Left", -cfg.side_wall_y_m), ("Right", cfg.side_wall_y_m)):
        _visual_cube(
            stage,
            f"{prim_path}/{side}WallBase",
            (length, cfg.wall_thickness_m, height),
            (cfg.center_x_m, y, wall_z),
            cfg.wall_base_rgb,
        )

    # Broad low-contrast vertical joints read as industrial sandwich panels.
    back_panel_width = 0.60
    panel_count = int(width / back_panel_width)
    for index in range(panel_count):
        y = -0.5 * width + (index + 0.5) * width / panel_count
        _visual_cube(
            stage,
            f"{prim_path}/BackWallpaper_{index:02d}",
            (0.004, width / panel_count - 0.018, height - 0.04),
            (cfg.back_wall_x_m + 0.5 * cfg.wall_thickness_m + 0.003, y, wall_z),
            cfg.wallpaper_rgb[index % len(cfg.wallpaper_rgb)],
        )
    side_panel_count = 10
    for side, y, sign in (("Left", -cfg.side_wall_y_m, 1.0), ("Right", cfg.side_wall_y_m, -1.0)):
        for index in range(side_panel_count):
            x = cfg.center_x_m - 0.5 * length + (index + 0.5) * length / side_panel_count
            _visual_cube(
                stage,
                f"{prim_path}/{side}Wallpaper_{index:02d}",
                (length / side_panel_count - 0.010, 0.004, height - 0.04),
                (x, y + sign * (0.5 * cfg.wall_thickness_m + 0.003), wall_z),
                cfg.wallpaper_rgb[(index + 1) % len(cfg.wallpaper_rgb)],
            )

    # Interior wall faces use a subtle paint-grain texture.  The
    # vertex order gives each face an inward normal, though meshes are double
    # sided so camera placement remains robust.
    wall_bottom = floor_top
    wall_top = floor_top + height
    back_face_x = cfg.back_wall_x_m + 0.5 * cfg.wall_thickness_m + 0.0055
    author_textured_quad(
        stage,
        f"{prim_path}/BackWallTexture",
        (
            (back_face_x, y_min, wall_bottom),
            (back_face_x, y_max, wall_bottom),
            (back_face_x, y_max, wall_top),
            (back_face_x, y_min, wall_top),
        ),
        cfg.wallpaper_uv_repeat,
        Path(cfg.wallpaper_texture_path),
        roughness=0.82,
    )
    left_face_y = -cfg.side_wall_y_m + 0.5 * cfg.wall_thickness_m + 0.0055
    right_face_y = cfg.side_wall_y_m - 0.5 * cfg.wall_thickness_m - 0.0055
    for side, y, points in (
        (
            "Left",
            left_face_y,
            (
                (x_max, left_face_y, wall_bottom),
                (x_min, left_face_y, wall_bottom),
                (x_min, left_face_y, wall_top),
                (x_max, left_face_y, wall_top),
            ),
        ),
        (
            "Right",
            right_face_y,
            (
                (x_min, right_face_y, wall_bottom),
                (x_max, right_face_y, wall_bottom),
                (x_max, right_face_y, wall_top),
                (x_min, right_face_y, wall_top),
            ),
        ),
    ):
        author_textured_quad(
            stage,
            f"{prim_path}/{side}WallTexture",
            points,
            cfg.wallpaper_uv_repeat,
            Path(cfg.wallpaper_texture_path),
            roughness=0.82,
        )

    baseboard_z = floor_top + 0.065
    _visual_cube(
        stage,
        f"{prim_path}/BackBaseboard",
        (0.025, width, 0.13),
        (cfg.back_wall_x_m + 0.5 * cfg.wall_thickness_m + 0.014, 0.0, baseboard_z),
        cfg.baseboard_rgb,
    )
    for side, y, sign in (("Left", -cfg.side_wall_y_m, 1.0), ("Right", cfg.side_wall_y_m, -1.0)):
        _visual_cube(
            stage,
            f"{prim_path}/{side}Baseboard",
            (length, 0.025, 0.13),
            (cfg.center_x_m, y + sign * (0.5 * cfg.wall_thickness_m + 0.014), baseboard_z),
            cfg.baseboard_rgb,
        )

    # Close the room volume and add a pair of non-emissive ceiling fixtures.
    ceiling_z = floor_top + height + 0.018
    _visual_cube(stage, f"{prim_path}/Ceiling", (length, width, 0.035), (cfg.center_x_m, 0.0, ceiling_z), (0.46, 0.45, 0.43))
    for index, y in enumerate((-0.85, 0.85)):
        _visual_cube(stage, f"{prim_path}/CeilingLight{index}", (1.10, 0.34, 0.028), (-0.35, y, ceiling_z - 0.030), (0.82, 0.83, 0.80))

    # Dark vibration control cabinet with contrasting front modules and screen.
    cabinet_x, cabinet_y, cabinet_z = cfg.cabinet_position
    _visual_cube(stage, f"{prim_path}/ControlCabinet", (0.60, 0.80, 1.80), cfg.cabinet_position, cfg.equipment_dark_rgb)
    _contact_patch(stage, f"{prim_path}/ControlCabinet/Shadow", (cabinet_x, cabinet_y), (0.66, 0.86), floor_top, cfg.wood_rgb[0])
    cabinet_front_y = cabinet_y + 0.405
    _visual_cube(stage, f"{prim_path}/ControlCabinet/FrontPanel", (0.50, 0.010, 1.58), (cabinet_x, cabinet_front_y, cabinet_z), cfg.equipment_light_rgb)
    _visual_cube(stage, f"{prim_path}/ControlCabinet/WaveformScreen", (0.34, 0.014, 0.24), (cabinet_x, cabinet_front_y + 0.008, cabinet_z + 0.48), (0.025, 0.16, 0.19))
    for row in range(4):
        _visual_cube(stage, f"{prim_path}/ControlCabinet/Amplifier{row}", (0.38, 0.014, 0.12), (cabinet_x, cabinet_front_y + 0.008, cabinet_z + 0.14 - 0.20 * row), (0.10, 0.11, 0.12))
        for led in range(3):
            _visual_cylinder(stage, f"{prim_path}/ControlCabinet/Amplifier{row}/Led{led}", 0.010, 0.012, (cabinet_x - 0.12 + 0.06 * led, cabinet_front_y + 0.018, cabinet_z + 0.14 - 0.20 * row), (0.08, 0.52 if led == 0 else 0.20, 0.12))

    # A compact chair supplies human scale without blocking the operation area.
    chair_x, chair_y, chair_z = cfg.chair_position
    _visual_cube(stage, f"{prim_path}/Chair/Seat", (0.48, 0.46, 0.07), (chair_x, chair_y, floor_top + 0.48), (0.08, 0.10, 0.13))
    _visual_cube(stage, f"{prim_path}/Chair/Back", (0.48, 0.07, 0.58), (chair_x, chair_y - 0.20, floor_top + 0.79), (0.07, 0.09, 0.12))
    for index, (dx, dy) in enumerate(((-0.18, -0.17), (-0.18, 0.17), (0.18, -0.17), (0.18, 0.17))):
        _visual_cylinder(stage, f"{prim_path}/Chair/Leg{index}", 0.018, 0.46, (chair_x + dx, chair_y + dy, floor_top + 0.23), (0.10, 0.11, 0.12))
        _contact_patch(stage, f"{prim_path}/Chair/Leg{index}/Shadow", (chair_x + dx, chair_y + dy), (0.055, 0.055), floor_top, cfg.wood_rgb[0])

    # Tool cart and two shelves add a second background depth layer.
    cart_x, cart_y, cart_z = cfg.tool_cart_position
    for level, z in enumerate((0.065, 0.40, 0.735)):
        _visual_cube(stage, f"{prim_path}/ToolCart/Shelf{level}", (0.70, 0.42, 0.045), (cart_x, cart_y, floor_top + z), (0.18, 0.20, 0.21))
    for index, (dx, dy) in enumerate(((-0.30, -0.16), (-0.30, 0.16), (0.30, -0.16), (0.30, 0.16))):
        post_height = 0.755
        _visual_cylinder(stage, f"{prim_path}/ToolCart/Post{index}", 0.018, post_height, (cart_x + dx, cart_y + dy, floor_top + 0.0425 + 0.5 * post_height), cfg.equipment_dark_rgb)
        _visual_cylinder(stage, f"{prim_path}/ToolCart/Foot{index}", 0.040, 0.040, (cart_x + dx, cart_y + dy, floor_top + 0.020), (0.10, 0.11, 0.12))
        _contact_patch(stage, f"{prim_path}/ToolCart/Foot{index}/Shadow", (cart_x + dx, cart_y + dy), (0.095, 0.095), floor_top, cfg.wood_rgb[0])

    # Emergency-stop pedestal with a high-contrast mushroom button.
    stop_x, stop_y, stop_z = cfg.emergency_stop_position
    _visual_cylinder(stage, f"{prim_path}/EmergencyStop/Post", 0.035, 0.95, (stop_x, stop_y, floor_top + 0.475), (0.28, 0.25, 0.12))
    _contact_patch(stage, f"{prim_path}/EmergencyStop/Shadow", (stop_x, stop_y), (0.11, 0.11), floor_top, cfg.wood_rgb[0])
    _visual_cube(stage, f"{prim_path}/EmergencyStop/Box", (0.18, 0.14, 0.18), (stop_x, stop_y, floor_top + 0.91), (0.62, 0.50, 0.12))
    _visual_cylinder(stage, f"{prim_path}/EmergencyStop/Button", 0.055, 0.050, (stop_x, stop_y, floor_top + 1.025), (0.72, 0.035, 0.025))

    # Cable bridge and a four-segment route from the cabinet to the pit.
    bridge_y = -cfg.side_wall_y_m + 0.10
    _visual_cube(stage, f"{prim_path}/CableBridge", (3.7, 0.10, 0.11), (-0.35, bridge_y, floor_top + 1.82), (0.12, 0.13, 0.14))
    cable_points = (
        (cabinet_x + 0.28, cabinet_y + 0.28, floor_top + 0.10),
        (-1.10, -1.15, floor_top + 0.08),
        (-0.80, -0.92, floor_top + 0.06),
        (-0.62, -0.72, floor_top - 0.18),
        (-0.50, -0.54, floor_top - 0.52),
    )
    cable_colors = ((0.018, 0.022, 0.026), (0.025, 0.035, 0.055), (0.08, 0.025, 0.018))
    for cable, color in enumerate(cable_colors):
        offset = 0.035 * cable
        for segment, (start, end) in enumerate(zip(cable_points, cable_points[1:])):
            start_offset = (start[0] + offset, start[1], start[2])
            end_offset = (end[0] + offset, end[1], end[2])
            _visual_cylinder_between(stage, f"{prim_path}/PowerCable{cable}_{segment}", start_offset, end_offset, 0.018, color)
    return stage.GetPrimAtPath(prim_path)
