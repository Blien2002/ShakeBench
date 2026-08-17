"""Newton-compatible textured surfaces and shallow-bin geometry."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
from isaaclab.sim.utils import clone, create_prim, get_current_stage
from isaaclab.utils.configclass import configclass

from .paths import PROJECT_ROOT
TABLE_TEXTURE_PATH = PROJECT_ROOT / "assets" / "textures" / "phenolic_bench_dark_1k.jpg"
PLATEN_TEXTURE_PATH = PROJECT_ROOT / "assets" / "textures" / "platen_threaded_holes_1k.jpg"


def platform_shadow_layout() -> dict[str, tuple[float, float]]:
    """Current projected contact cues in vibration-floor local coordinates."""

    return {
        "table_foot_0": (-0.09, -0.245),
        "table_foot_1": (-0.09, 0.245),
        "table_foot_2": (0.45, -0.245),
        "table_foot_3": (0.45, 0.245),
        "robot_base": (-0.47, 0.0),
        "target_bin": (0.08, 0.17),
        "workpiece": (0.08, -0.13),
        "platen": (0.0, 0.0),
    }


def _textured_material(stage: Any, path: str, texture_path: Path, roughness: float) -> Any:
    """Author a portable UsdPreviewSurface/UsdUVTexture graph."""

    from pxr import Sdf, UsdShade

    material = UsdShade.Material.Define(stage, path)
    # Newton multiplies albedo textures by the resolved shape color. Without
    # an explicit material color it falls back to a per-shape debug palette,
    # tinting identical wallpaper meshes cyan/orange/red. White preserves the
    # source texture exactly.
    material.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set((1.0, 1.0, 1.0))
    surface = UsdShade.Shader.Define(stage, f"{path}/PreviewSurface")
    surface.CreateIdAttr("UsdPreviewSurface")
    surface.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
    surface.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)

    texture = UsdShade.Shader.Define(stage, f"{path}/AlbedoTexture")
    texture.CreateIdAttr("UsdUVTexture")
    texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(texture_path.resolve())))
    texture.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
    texture.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
    texture.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")

    st_reader = UsdShade.Shader.Define(stage, f"{path}/TexCoordReader")
    st_reader.CreateIdAttr("UsdPrimvarReader_float2")
    st_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    st_output = st_reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
    texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st_output)
    rgb_output = texture.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(rgb_output)
    material.CreateSurfaceOutput().ConnectToSource(
        surface.ConnectableAPI(),
        "surface",
    )
    return material


def author_textured_quad(
    stage: Any,
    path: str,
    points: tuple[tuple[float, float, float], ...],
    uv_repeat: tuple[float, float],
    texture_path: Path,
    roughness: float = 0.55,
) -> None:
    """Create a UV-mapped, closed thin mesh rendered directly by NewtonGL.

    NewtonGL obtains render shapes from the collision model, while MJWarp's
    MuJoCo contact generator rejects exactly planar mesh colliders.  Extruding
    the quad by 0.2 mm keeps it visually flat but makes it a valid closed mesh.
    """

    from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade

    if len(points) != 4:
        raise ValueError("textured quad requires exactly four points")
    edge_a = tuple(points[1][axis] - points[0][axis] for axis in range(3))
    edge_b = tuple(points[2][axis] - points[0][axis] for axis in range(3))
    normal = (
        edge_a[1] * edge_b[2] - edge_a[2] * edge_b[1],
        edge_a[2] * edge_b[0] - edge_a[0] * edge_b[2],
        edge_a[0] * edge_b[1] - edge_a[1] * edge_b[0],
    )
    normal_length = math.sqrt(sum(value * value for value in normal))
    if normal_length < 1.0e-12:
        raise ValueError("textured quad points must span a non-zero area")
    half_thickness = 0.0001
    offset = tuple(half_thickness * value / normal_length for value in normal)
    front = [tuple(point[axis] + offset[axis] for axis in range(3)) for point in points]
    back = [tuple(point[axis] - offset[axis] for axis in range(3)) for point in points]

    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr([Gf.Vec3f(*point) for point in (*front, *back)])
    mesh.CreateFaceVertexCountsAttr([4, 4, 4, 4, 4, 4])
    mesh.CreateFaceVertexIndicesAttr(
        [
            0, 1, 2, 3,
            7, 6, 5, 4,
            0, 4, 5, 1,
            1, 5, 6, 2,
            2, 6, 7, 3,
            3, 7, 4, 0,
        ]
    )
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr(True)
    u, v = uv_repeat
    primvar = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "st",
        Sdf.ValueTypeNames.TexCoord2fArray,
        UsdGeom.Tokens.vertex,
    )
    face_uv = [
        Gf.Vec2f(0.0, 0.0),
        Gf.Vec2f(float(u), 0.0),
        Gf.Vec2f(float(u), float(v)),
        Gf.Vec2f(0.0, float(v)),
    ]
    primvar.Set(face_uv + face_uv)
    material = _textured_material(stage, f"{path}_Material", texture_path, roughness)
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    # ViewerGL retains Newton shapes whose CollisionAPI is explicitly disabled,
    # while MJWarp omits them from geometry/pair generation. Textured skins are
    # visual detail only; robust parent cuboids provide any required contacts.
    collision = UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    collision.CreateCollisionEnabledAttr(False)


def _colored_cube(stage: Any, path: str, size, position, color) -> None:
    from pxr import Gf, UsdGeom, UsdPhysics

    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    xform = UsdGeom.Xformable(cube)
    xform.AddTranslateOp().Set(Gf.Vec3d(*position))
    xform.AddScaleOp().Set(Gf.Vec3d(*size))
    gprim = UsdGeom.Gprim(cube)
    gprim.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    gprim.CreateDisplayOpacityAttr([1.0])
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())


def _visual_cube(stage: Any, path: str, size, position, color, rotate_y_deg: float = 0.0) -> None:
    """Author a display-only cube that NewtonGL keeps out of MJWarp."""

    from pxr import Gf, UsdGeom, UsdPhysics

    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    xform = UsdGeom.Xformable(cube)
    xform.AddTranslateOp().Set(Gf.Vec3d(*position))
    if rotate_y_deg:
        xform.AddRotateYOp().Set(float(rotate_y_deg))
    xform.AddScaleOp().Set(Gf.Vec3d(*size))
    gprim = UsdGeom.Gprim(cube)
    gprim.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    gprim.CreateDisplayOpacityAttr([1.0])
    collision = UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    collision.CreateCollisionEnabledAttr(False)


def _visual_cylinder(stage: Any, path: str, radius: float, height: float, position, color) -> None:
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
    xform.AddOrientOp().Set(Gf.Quatf(q_w / q_norm, Gf.Vec3f(*(value / q_norm for value in q_xyz))))
    gprim = UsdGeom.Gprim(cylinder)
    gprim.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    gprim.CreateDisplayOpacityAttr([1.0])
    collision = UsdPhysics.CollisionAPI.Apply(cylinder.GetPrim())
    collision.CreateCollisionEnabledAttr(False)


def _visual_sphere(stage: Any, path: str, radius: float, position, color) -> None:
    """Author a collision-free sphere for rounded control grips and caps."""

    from pxr import Gf, UsdGeom, UsdPhysics

    sphere = UsdGeom.Sphere.Define(stage, path)
    sphere.CreateRadiusAttr(float(radius))
    UsdGeom.Xformable(sphere).AddTranslateOp().Set(Gf.Vec3d(*position))
    gprim = UsdGeom.Gprim(sphere)
    gprim.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    gprim.CreateDisplayOpacityAttr([1.0])
    collision = UsdPhysics.CollisionAPI.Apply(sphere.GetPrim())
    collision.CreateCollisionEnabledAttr(False)


def _visual_mesh(stage: Any, path: str, points, faces, color) -> None:
    """Author a small closed, flat-shaded display mesh."""

    from pxr import Gf, UsdGeom, UsdPhysics

    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr([Gf.Vec3f(*point) for point in points])
    mesh.CreateFaceVertexCountsAttr([len(face) for face in faces])
    mesh.CreateFaceVertexIndicesAttr([index for face in faces for index in face])
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr(True)
    gprim = UsdGeom.Gprim(mesh)
    gprim.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    gprim.CreateDisplayOpacityAttr([1.0])
    collision = UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    collision.CreateCollisionEnabledAttr(False)


def _visual_wedge_prism(stage: Any, path: str, outline_xz, width: float, color) -> None:
    """Extrude a five-sided console outline along Y."""

    half_width = 0.5 * float(width)
    points = [
        (float(x), -half_width, float(z)) for x, z in outline_xz
    ] + [
        (float(x), half_width, float(z)) for x, z in outline_xz
    ]
    count = len(outline_xz)
    faces = [tuple(reversed(range(count))), tuple(range(count, 2 * count))]
    faces.extend(
        (index, (index + 1) % count, (index + 1) % count + count, index + count)
        for index in range(count)
    )
    _visual_mesh(stage, path, points, faces, color)


def _visual_extruded_polygon(
    stage: Any,
    path: str,
    center,
    tangent,
    lateral,
    normal,
    outline,
    lower_m: float,
    upper_m: float,
    color,
) -> None:
    """Extrude a 2-D control silhouette away from the operator surface."""

    def point(a: float, b: float, height: float):
        return tuple(
            float(center[axis])
            + float(a) * float(tangent[axis])
            + float(b) * float(lateral[axis])
            + float(height) * float(normal[axis])
            for axis in range(3)
        )

    points = [point(a, b, lower_m) for a, b in outline]
    points.extend(point(a, b, upper_m) for a, b in outline)
    count = len(outline)
    faces = [tuple(reversed(range(count))), tuple(range(count, 2 * count))]
    faces.extend(
        (index, (index + 1) % count, (index + 1) % count + count, index + count)
        for index in range(count)
    )
    _visual_mesh(stage, path, points, faces, color)


def _shadow_disc(stage: Any, path: str, center, radii, z: float, surface_rgb) -> None:
    """Approximate a soft decal with opaque concentric, collision-free discs.

    ViewerGL does not currently honor display opacity. Concentric discs avoid
    the opaque-square artifact of an alpha texture while preserving a soft
    contact cue and a zero MJWarp contact-pair delta.
    """

    from pxr import Gf, UsdGeom, UsdPhysics

    colors = (
        tuple(0.18 * value for value in surface_rgb),
        tuple(0.38 * value for value in surface_rgb),
        tuple(0.65 * value for value in surface_rgb),
    )
    scales = (0.45, 0.72, 1.0)
    for layer, (scale, color) in enumerate(zip(scales, colors)):
        disc = UsdGeom.Cylinder.Define(stage, f"{path}_{layer}")
        disc.CreateAxisAttr(UsdGeom.Tokens.z)
        disc.CreateRadiusAttr(1.0)
        disc.CreateHeightAttr(0.0007)
        xform = UsdGeom.Xformable(disc)
        xform.AddTranslateOp().Set(Gf.Vec3d(center[0], center[1], z + layer * 0.0005))
        xform.AddScaleOp().Set(Gf.Vec3d(radii[0] * scale, radii[1] * scale, 1.0))
        gprim = UsdGeom.Gprim(disc)
        gprim.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        gprim.CreateDisplayOpacityAttr([1.0])
        collision = UsdPhysics.CollisionAPI.Apply(disc.GetPrim())
        collision.CreateCollisionEnabledAttr(False)


def _shadow_square(stage: Any, path: str, center, size, z: float, surface_rgb) -> None:
    """Three nested square patches for square-tube feet."""

    colors = (
        tuple(0.18 * value for value in surface_rgb),
        tuple(0.38 * value for value in surface_rgb),
        tuple(0.65 * value for value in surface_rgb),
    )
    for layer, (scale, color) in enumerate(zip((0.48, 0.73, 1.0), colors)):
        _visual_cube(
            stage,
            f"{path}_{layer}",
            (size[0] * scale, size[1] * scale, 0.0007),
            (center[0], center[1], z + layer * 0.0005),
            color,
        )


@configclass
class PlatformAppearanceCfg(SpawnerCfg):
    """Layered platen finish and static contact-shadow cues."""

    func = None
    size_xy: tuple[float, float] = (1.60, 1.10)
    top_z_m: float = 0.043
    top_thickness_m: float = 0.006
    top_rgb: tuple[float, float, float] = (0.46, 0.47, 0.48)
    texture_path: str = str(PLATEN_TEXTURE_PATH)
    skirt_rgb: tuple[float, float, float] = (0.13, 0.14, 0.145)
    skirt_height_m: float = 0.08
    bevel_m: float = 0.005
    robot_xy: tuple[float, float] = (-0.47, 0.0)
    target_xy: tuple[float, float] = (0.08, 0.17)
    table_leg_xy: tuple[tuple[float, float], ...] = (
        (-0.09, -0.245),
        (-0.09, 0.245),
        (0.45, -0.245),
        (0.45, 0.245),
    )


@clone
def spawn_platform_appearance(
    prim_path: str,
    cfg: PlatformAppearanceCfg,
    translation=None,
    orientation=None,
    **_: Any,
):
    """Create a bright top plate over a dark physical skirt plus contact cues."""

    from pxr import Gf, UsdGeom, UsdPhysics

    stage = get_current_stage()
    create_prim(prim_path, "Xform", translation=translation, orientation=orientation, stage=stage)
    _visual_cube(
        stage,
        f"{prim_path}/TopPlate",
        (cfg.size_xy[0] - 0.012, cfg.size_xy[1] - 0.012, cfg.top_thickness_m),
        (0.0, 0.0, cfg.top_z_m),
        cfg.top_rgb,
    )
    # A separate four-sided skirt makes the configured 80 mm platen thickness
    # readable from low angles. Thin cap strips approximate a 5 mm bevel in
    # ViewerGL without introducing collision geometry.
    sx, sy = cfg.size_xy
    panel_h = cfg.skirt_height_m - 2.0 * cfg.bevel_m
    for name, size, position in (
        ("Front", (sx, 0.010, panel_h), (0.0, -0.5 * sy - 0.002, 0.0)),
        ("Rear", (sx, 0.010, panel_h), (0.0, 0.5 * sy + 0.002, 0.0)),
        ("Left", (0.010, sy, panel_h), (-0.5 * sx - 0.002, 0.0, 0.0)),
        ("Right", (0.010, sy, panel_h), (0.5 * sx + 0.002, 0.0, 0.0)),
    ):
        _visual_cube(stage, f"{prim_path}/SideSkirt{name}", size, position, cfg.skirt_rgb)
    for edge, z in (("Upper", 0.5 * cfg.skirt_height_m - 0.5 * cfg.bevel_m), ("Lower", -0.5 * cfg.skirt_height_m + 0.5 * cfg.bevel_m)):
        _visual_cube(stage, f"{prim_path}/Bevel{edge}Front", (sx, 0.016, cfg.bevel_m), (0.0, -0.5 * sy - 0.003, z), tuple(min(1.0, value * (1.14 if edge == "Upper" else 0.78)) for value in cfg.skirt_rgb))
        _visual_cube(stage, f"{prim_path}/Bevel{edge}Rear", (sx, 0.016, cfg.bevel_m), (0.0, 0.5 * sy + 0.003, z), tuple(min(1.0, value * (1.14 if edge == "Upper" else 0.78)) for value in cfg.skirt_rgb))
    shadow_z = cfg.top_z_m + 0.5 * cfg.top_thickness_m + 0.0006
    half_x, half_y = 0.5 * cfg.size_xy[0], 0.5 * cfg.size_xy[1]
    author_textured_quad(
        stage,
        f"{prim_path}/ThreadedHoleSurface",
        (
            (-half_x, -half_y, shadow_z),
            (half_x, -half_y, shadow_z),
            (half_x, half_y, shadow_z),
            (-half_x, half_y, shadow_z),
        ),
        (1.0, 1.0),
        Path(cfg.texture_path),
        roughness=0.34,
    )
    for index, (x, y) in enumerate(
        (
            (-0.5 * cfg.size_xy[0] + 0.09, -0.5 * cfg.size_xy[1] + 0.09),
            (-0.5 * cfg.size_xy[0] + 0.09, 0.5 * cfg.size_xy[1] - 0.09),
            (0.5 * cfg.size_xy[0] - 0.09, -0.5 * cfg.size_xy[1] + 0.09),
            (0.5 * cfg.size_xy[0] - 0.09, 0.5 * cfg.size_xy[1] - 0.09),
        )
    ):
        sensor_root = f"{prim_path}/Accelerometer{index}"
        create_prim(sensor_root, "Xform", translation=(x, y, shadow_z + 0.016), stage=stage)
        _visual_cylinder(stage, f"{sensor_root}/Base", 0.034, 0.005, (0.0, 0.0, -0.013), (0.28, 0.29, 0.30))
        _visual_cube(stage, f"{sensor_root}/Housing", (0.046, 0.040, 0.022), (0.0, 0.0, 0.0), (0.12, 0.16, 0.20))
        sensor = UsdGeom.Cylinder.Define(stage, f"{sensor_root}/Connector")
        sensor.CreateAxisAttr(UsdGeom.Tokens.z)
        sensor.CreateRadiusAttr(0.008)
        sensor.CreateHeightAttr(0.010)
        UsdGeom.Xformable(sensor).AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.016))
        gprim = UsdGeom.Gprim(sensor)
        gprim.CreateDisplayColorAttr([Gf.Vec3f(0.85, 0.42, 0.08)])
        gprim.CreateDisplayOpacityAttr([1.0])
        collision = UsdPhysics.CollisionAPI.Apply(sensor.GetPrim())
        collision.CreateCollisionEnabledAttr(False)
        edge_y = math.copysign(0.5 * cfg.size_xy[1] - 0.015, y)
        _visual_cylinder_between(
            stage,
            f"{sensor_root}/Cable",
            (0.0, 0.0, 0.014),
            (0.0, edge_y - y, -0.006),
            0.004,
            (0.025, 0.030, 0.035),
        )

    side_y = 0.5 * cfg.size_xy[1] + 0.0008
    _visual_cube(stage, f"{prim_path}/Nameplate", (0.30, 0.0015, 0.032), (0.0, side_y + 0.001, 0.014), (0.62, 0.66, 0.68))
    _visual_cube(stage, f"{prim_path}/WarningBandBase", (1.18, 0.0020, 0.034), (0.0, side_y, -0.017), (0.025, 0.030, 0.035))
    for index in range(13):
        x = -0.56 + index * 0.093
        _visual_cube(
            stage,
            f"{prim_path}/WarningStripe{index}",
            (0.052, 0.0024, 0.064),
            (x, side_y + 0.001, -0.017),
            (0.90, 0.67, 0.06),
            rotate_y_deg=35.0,
        )
    for index, point in enumerate(cfg.table_leg_xy):
        _shadow_square(stage, f"{prim_path}/TableFootShadow{index}", point, (0.082, 0.082), shadow_z, cfg.top_rgb)
    _shadow_disc(stage, f"{prim_path}/RobotBaseShadow", cfg.robot_xy, (0.14, 0.115), shadow_z, cfg.top_rgb)
    _shadow_disc(stage, f"{prim_path}/TargetBinShadow", cfg.target_xy, (0.12, 0.105), shadow_z, cfg.top_rgb)
    # Thick robot mounting flange and its bolt circle replace the former
    # sticker-like ellipse while remaining display-only.
    _visual_cylinder(stage, f"{prim_path}/RobotMountFlange", 0.120, 0.014, (cfg.robot_xy[0], cfg.robot_xy[1], shadow_z + 0.008), (0.34, 0.35, 0.35))
    for index in range(8):
        angle = index * math.pi / 4.0
        # Bolt transforms are flange-local.  Supplying platen coordinates here
        # used to apply the robot-base translation twice and left a floating
        # ring over the shaker pit.
        _visual_cylinder(
            stage,
            f"{prim_path}/RobotMountFlange/Bolt{index}",
            0.006,
            0.006,
            (0.092 * math.cos(angle), 0.092 * math.sin(angle), 0.010),
            (0.055, 0.060, 0.065),
        )
    return stage.GetPrimAtPath(prim_path)


@configclass
class WorktableAppearanceCfg(SpawnerCfg):
    """Collision-free laboratory-frame finish layered over robust colliders."""

    func = None
    top_size: tuple[float, float, float] = (0.65, 0.60, 0.06)
    leg_height_m: float = 0.23
    leg_center_z_m: float = -0.145
    leg_xy: tuple[tuple[float, float], ...] = (
        (-0.27, -0.245),
        (-0.27, 0.245),
        (0.27, -0.245),
        (0.27, 0.245),
    )
    frame_rgb: tuple[float, float, float] = (0.095, 0.105, 0.115)
    edge_rgb: tuple[float, float, float] = (0.075, 0.085, 0.095)


@clone
def spawn_worktable_appearance(
    prim_path: str,
    cfg: WorktableAppearanceCfg,
    translation=None,
    orientation=None,
    **_: Any,
):
    """Create visible tabletop edges, tube frame, stretchers and bolted feet."""

    from pxr import Gf, UsdGeom, UsdPhysics

    stage = get_current_stage()
    create_prim(prim_path, "Xform", translation=translation, orientation=orientation, stage=stage)
    sx, sy, sz = cfg.top_size
    band = 0.012
    _visual_cube(stage, f"{prim_path}/EdgeFront", (sx, band, sz), (0.0, -0.5 * sy, 0.0), cfg.edge_rgb)
    _visual_cube(stage, f"{prim_path}/EdgeRear", (sx, band, sz), (0.0, 0.5 * sy, 0.0), cfg.edge_rgb)
    _visual_cube(stage, f"{prim_path}/EdgeLeft", (band, sy - 2.0 * band, sz), (-0.5 * sx, 0.0, 0.0), cfg.edge_rgb)
    _visual_cube(stage, f"{prim_path}/EdgeRight", (band, sy - 2.0 * band, sz), (0.5 * sx, 0.0, 0.0), cfg.edge_rgb)

    foot_z = cfg.leg_center_z_m - 0.5 * cfg.leg_height_m + 0.004
    for index, (x, y) in enumerate(cfg.leg_xy):
        _visual_cube(
            stage,
            f"{prim_path}/TubeLeg{index}",
            (0.065, 0.065, cfg.leg_height_m),
            (x, y, cfg.leg_center_z_m),
            cfg.frame_rgb,
        )
        _visual_cube(stage, f"{prim_path}/FootPlate{index}", (0.105, 0.095, 0.010), (x, y, foot_z), cfg.edge_rgb)
        for bolt, (dx, dy) in enumerate(((-0.036, -0.031), (-0.036, 0.031), (0.036, -0.031), (0.036, 0.031))):
            head = UsdGeom.Cylinder.Define(stage, f"{prim_path}/FootPlate{index}/Bolt{bolt}")
            head.CreateAxisAttr(UsdGeom.Tokens.z)
            head.CreateRadiusAttr(0.006)
            head.CreateHeightAttr(0.006)
            # Bolt is a child of FootPlate, so use plate-local coordinates.
            UsdGeom.Xformable(head).AddTranslateOp().Set(Gf.Vec3d(dx, dy, 0.008))
            gprim = UsdGeom.Gprim(head)
            gprim.CreateDisplayColorAttr([Gf.Vec3f(0.035, 0.040, 0.045)])
            gprim.CreateDisplayOpacityAttr([1.0])
            collision = UsdPhysics.CollisionAPI.Apply(head.GetPrim())
            collision.CreateCollisionEnabledAttr(False)

    x_span = abs(cfg.leg_xy[2][0] - cfg.leg_xy[0][0])
    stretcher_z = foot_z + 0.075
    for side, y in (("Front", cfg.leg_xy[0][1]), ("Rear", cfg.leg_xy[1][1])):
        _visual_cube(stage, f"{prim_path}/LowerStretcher{side}", (x_span, 0.030, 0.035), (0.0, y, stretcher_z), cfg.frame_rgb)
    _visual_cube(stage, f"{prim_path}/LowerCrossbarLeft", (0.030, abs(cfg.leg_xy[1][1] - cfg.leg_xy[0][1]), 0.035), (cfg.leg_xy[0][0], 0.0, stretcher_z), cfg.frame_rgb)
    _visual_cube(stage, f"{prim_path}/LowerCrossbarRight", (0.030, abs(cfg.leg_xy[3][1] - cfg.leg_xy[2][1]), 0.035), (cfg.leg_xy[2][0], 0.0, stretcher_z), cfg.frame_rgb)
    return stage.GetPrimAtPath(prim_path)


@configclass
class TexturedTableSurfaceCfg(SpawnerCfg):
    """Thin UV surface rigidly parented to the kinematic worktable."""

    func = None
    size_xy: tuple[float, float] = (0.65, 0.60)
    top_z_m: float = 0.0304
    texture_path: str = str(TABLE_TEXTURE_PATH)
    uv_repeat: tuple[float, float] = (1.0, 1.0)


@clone
def spawn_textured_table_surface(
    prim_path: str,
    cfg: TexturedTableSurfaceCfg,
    translation=None,
    orientation=None,
    **_: Any,
):
    stage = get_current_stage()
    create_prim(prim_path, "Xform", translation=translation, orientation=orientation, stage=stage)
    half_x, half_y = 0.5 * cfg.size_xy[0], 0.5 * cfg.size_xy[1]
    author_textured_quad(
        stage,
        f"{prim_path}/SurfaceMesh",
        (
            (-half_x, -half_y, cfg.top_z_m),
            (half_x, -half_y, cfg.top_z_m),
            (half_x, half_y, cfg.top_z_m),
            (-half_x, half_y, cfg.top_z_m),
        ),
        cfg.uv_repeat,
        Path(cfg.texture_path),
        roughness=0.46,
    )
    return stage.GetPrimAtPath(prim_path)




@configclass
class ControlPanelAppearanceCfg(SpawnerCfg):
    """Display-only sloped console and compound controls for the panel task.

    The silhouette is an original five-sided industrial console inspired by
    laboratory control desks.  The selector uses the broad-rear/tapered-tip
    visual language of an Apollo-era pointer knob, but no third-party mesh is
    copied or bundled.  Every shape in this review layer is collision-disabled.
    """

    func = None
    board_size: tuple[float, float, float] = (0.050, 0.300, 0.180)
    knob_uv: tuple[float, float] = (-0.085, 0.055)
    lever_uv: tuple[float, float] = (0.0, -0.055)
    button_uv: tuple[float, float] = (0.085, 0.055)
    console_depth_m: float = 0.190
    console_width_m: float = 0.320
    console_height_m: float = 0.180
    front_height_m: float = 0.055
    rear_flat_depth_m: float = 0.025
    housing_rgb: tuple[float, float, float] = (0.24, 0.27, 0.30)
    face_rgb: tuple[float, float, float] = (0.075, 0.085, 0.095)
    edge_rgb: tuple[float, float, float] = (0.36, 0.39, 0.42)
    knob_rgb: tuple[float, float, float] = (0.68, 0.69, 0.67)
    lever_rgb: tuple[float, float, float] = (0.055, 0.060, 0.065)
    button_rgb: tuple[float, float, float] = (0.82, 0.075, 0.050)


@clone
def spawn_control_panel_appearance(
    prim_path: str,
    cfg: ControlPanelAppearanceCfg,
    translation=None,
    orientation=None,
    **_: Any,
):
    """Author a sloped five-prism console with three readable controls."""

    stage = get_current_stage()
    create_prim(prim_path, "Xform", translation=translation, orientation=orientation, stage=stage)
    depth = float(cfg.console_depth_m)
    width = float(cfg.console_width_m)
    height = float(cfg.console_height_m)
    half_depth = 0.5 * depth
    half_height = 0.5 * height
    front_top_z = -half_height + float(cfg.front_height_m)
    shoulder_x = half_depth - float(cfg.rear_flat_depth_m)
    outline = (
        (-half_depth, -half_height),
        (half_depth, -half_height),
        (half_depth, half_height),
        (shoulder_x, half_height),
        (-half_depth, front_top_z),
    )
    _visual_wedge_prism(stage, f"{prim_path}/ConsoleHousing", outline, width, cfg.housing_rgb)

    # The operator face runs from the low front edge to the short high rear deck.
    slope_dx = shoulder_x + half_depth
    slope_dz = half_height - front_top_z
    slope_length = math.hypot(slope_dx, slope_dz)
    tangent = (slope_dx / slope_length, 0.0, slope_dz / slope_length)
    lateral = (0.0, 1.0, 0.0)
    normal = (-tangent[2], 0.0, tangent[0])
    surface_center = (
        0.5 * (shoulder_x - half_depth),
        0.0,
        0.5 * (half_height + front_top_z),
    )

    def add(a, b):
        return tuple(float(a[i]) + float(b[i]) for i in range(3))

    def scale(vector, factor):
        return tuple(float(factor) * float(value) for value in vector)

    def surface_point(uy: float, vz: float, lift: float = 0.0):
        return tuple(
            surface_center[axis]
            + float(uy) * lateral[axis]
            + float(vz) * tangent[axis]
            + float(lift) * normal[axis]
            for axis in range(3)
        )

    face_angle_deg = math.degrees(math.atan2(-tangent[0], -tangent[2]))
    face_center = surface_point(0.0, 0.0, 0.003)
    _visual_cube(
        stage,
        f"{prim_path}/OperatorFace",
        (0.006, width - 0.018, slope_length - 0.016),
        face_center,
        cfg.face_rgb,
        rotate_y_deg=face_angle_deg,
    )
    # Bright metal edge cheeks and a short rear cap make the five-prism
    # silhouette legible from the main three-quarter camera.
    for side, uy in (("Left", -0.5 * width + 0.006), ("Right", 0.5 * width - 0.006)):
        _visual_cube(
            stage,
            f"{prim_path}/FaceEdge{side}",
            (0.008, 0.010, slope_length),
            surface_point(uy, 0.0, 0.006),
            cfg.edge_rgb,
            rotate_y_deg=face_angle_deg,
        )
    _visual_cube(
        stage,
        f"{prim_path}/RearCap",
        (cfg.rear_flat_depth_m, width - 0.018, 0.010),
        (half_depth - 0.5 * cfg.rear_flat_depth_m, 0.0, half_height + 0.004),
        cfg.edge_rgb,
    )

    # Four recessed fasteners, plus a slim central annunciator inspired by
    # the dense industrial reference panel without reproducing its mesh.
    for index, (uy, vz) in enumerate(((-0.142, -0.080), (0.142, -0.080), (-0.142, 0.080), (0.142, 0.080))):
        base = surface_point(uy, vz, 0.006)
        _visual_cylinder_between(
            stage,
            f"{prim_path}/Fastener{index}",
            base,
            add(base, scale(normal, 0.004)),
            0.0045,
            (0.52, 0.55, 0.57),
        )
    _visual_cube(
        stage,
        f"{prim_path}/Annunciator",
        (0.008, 0.060, 0.026),
        surface_point(0.0, 0.067, 0.008),
        (0.015, 0.022, 0.028),
        rotate_y_deg=face_angle_deg,
    )
    for index, (uy, color) in enumerate(((-0.018, (0.18, 0.86, 0.34)), (0.0, (0.96, 0.67, 0.08)), (0.018, (0.82, 0.10, 0.06)))):
        base = surface_point(uy, 0.067, 0.014)
        _visual_cylinder_between(
            stage,
            f"{prim_path}/AnnunciatorLamp{index}",
            base,
            add(base, scale(normal, 0.004)),
            0.0042,
            color,
        )

    # Fixed bezels live on the console.  Moving handles/caps are authored by
    # the corresponding articulation link in panel_controls.py.
    knob = surface_point(*cfg.knob_uv)
    for name, radius, start_m, end_m, color in (
        ("KnobOuterBezel", 0.034, 0.005, 0.011, (0.44, 0.46, 0.47)),
        ("KnobInnerBezel", 0.028, 0.010, 0.016, (0.11, 0.12, 0.13)),
    ):
        _visual_cylinder_between(
            stage, f"{prim_path}/{name}", add(knob, scale(normal, start_m)),
            add(knob, scale(normal, end_m)), radius, color
        )
    # Industrial toggle fixed boot.
    lever = surface_point(*cfg.lever_uv)
    _visual_cylinder_between(
        stage, f"{prim_path}/LeverBezel", add(lever, scale(normal, 0.005)),
        add(lever, scale(normal, 0.012)), 0.029, (0.43, 0.45, 0.46)
    )
    _visual_cylinder_between(
        stage, f"{prim_path}/LeverBoot", add(lever, scale(normal, 0.010)),
        add(lever, scale(normal, 0.028)), 0.017, (0.035, 0.040, 0.045)
    )
    # Fixed pushbutton collar.
    button = surface_point(*cfg.button_uv)
    _visual_cylinder_between(
        stage, f"{prim_path}/ButtonBezel", add(button, scale(normal, 0.005)),
        add(button, scale(normal, 0.012)), 0.032, (0.47, 0.49, 0.50)
    )
    _visual_cylinder_between(
        stage, f"{prim_path}/ButtonCollar", add(button, scale(normal, 0.011)),
        add(button, scale(normal, 0.021)), 0.024, (0.10, 0.11, 0.12)
    )
    # Neutral metal label plaques keep the assembly readable without relying
    # on renderer-dependent text glyphs.
    for kind, (uy, vz) in (("Knob", cfg.knob_uv), ("Lever", cfg.lever_uv), ("Button", cfg.button_uv)):
        label_v = max(-0.088, vz - 0.050)
        _visual_cube(
            stage,
            f"{prim_path}/Label{kind}",
            (0.006, 0.047, 0.014),
            surface_point(uy, label_v, 0.007),
            (0.58, 0.60, 0.60),
            rotate_y_deg=face_angle_deg,
        )
    return stage.GetPrimAtPath(prim_path)


@configclass
class ShallowBinWallsCfg(SpawnerCfg):
    """Four collision walls attached to the target bin's rigid bottom."""

    func = None
    outer_size_xy: tuple[float, float] = (0.18, 0.16)
    wall_thickness_m: float = 0.008
    wall_height_m: float = 0.035
    bottom_thickness_m: float = 0.012
    color: tuple[float, float, float] = (0.72, 0.86, 0.92)


@clone
def spawn_shallow_bin_walls(
    prim_path: str,
    cfg: ShallowBinWallsCfg,
    translation=None,
    orientation=None,
    **_: Any,
):
    stage = get_current_stage()
    create_prim(prim_path, "Xform", translation=translation, orientation=orientation, stage=stage)
    size_x, size_y = cfg.outer_size_xy
    t = cfg.wall_thickness_m
    wall_z = 0.5 * cfg.bottom_thickness_m + 0.5 * cfg.wall_height_m
    _colored_cube(stage, f"{prim_path}/WallFront", (size_x, t, cfg.wall_height_m), (0.0, -0.5 * (size_y - t), wall_z), cfg.color)
    _colored_cube(stage, f"{prim_path}/WallRear", (size_x, t, cfg.wall_height_m), (0.0, 0.5 * (size_y - t), wall_z), cfg.color)
    _colored_cube(stage, f"{prim_path}/WallLeft", (t, size_y - 2.0 * t, cfg.wall_height_m), (-0.5 * (size_x - t), 0.0, wall_z), cfg.color)
    _colored_cube(stage, f"{prim_path}/WallRight", (t, size_y - 2.0 * t, cfg.wall_height_m), (0.5 * (size_x - t), 0.0, wall_z), cfg.color)
    return stage.GetPrimAtPath(prim_path)
