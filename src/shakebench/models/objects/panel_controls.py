"""Procedural one-DoF panel controls for Newton/MJWarp.

Each control is a two-body articulation: a tiny floating base that is moved
with the worktable support frame, plus one collision-enabled moving link.  The
render geometry is deliberately richer than the collider, so the approved
industrial appearance does not make contact generation unnecessarily fragile.
"""

from __future__ import annotations

import math
from pathlib import Path
import struct
from typing import Any

from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
from isaaclab.sim.utils import clone, create_prim, get_current_stage
from isaaclab.utils.configclass import configclass

from ..visual_assets import (
    _visual_cylinder_between,
)
from ...utils.paths import PROJECT_ROOT

APOLLO_KNOB_SOURCE_STL_PATH = (
    PROJECT_ROOT / "assets" / "models" / "apollo_command_module_control_panel_knob.stl"
)
APOLLO_KNOB_RUNTIME_STL_PATH = (
    PROJECT_ROOT
    / "assets"
    / "models"
    / "apollo_command_module_control_panel_knob.runtime.stl"
)

# The runtime LOD preserves the source STL's 49.998 x 20.701 x 49.963 mm
# envelope.  Its lowest point is mounted four millimetres above the fixed
# bezel's 15 mm top.  The proxy fully encloses the 25.737 mm radial and
# 20.701 mm normal extents so robot fingers cannot enter the visible mesh.
APOLLO_KNOB_VISUAL_BASE_NORMAL_M = 0.019
APOLLO_KNOB_COLLIDER_RADIUS_M = 0.026
APOLLO_KNOB_COLLIDER_HEIGHT_M = 0.021

SWITCH_BASE_RUNTIME_STL_PATH = (
    PROJECT_ROOT / "assets" / "models" / "switch_control_base.runtime.stl"
)
SWITCH_HANDLE_RUNTIME_STL_PATH = (
    PROJECT_ROOT / "assets" / "models" / "switch_control_handle.runtime.stl"
)
# FBX source axes are X/Z in the mounting plane and +Y outward.  Both runtime
# components use this shared origin so the fixed base and moving handle retain
# their authored alignment after uniform down-scaling to the panel.
SWITCH_SOURCE_ORIGIN = (0.3810272216796875, 34.99650955200195, 0.7973480224609375)
SWITCH_SCALE_M_PER_SOURCE_UNIT = 0.00056
SWITCH_BASE_NORMAL_M = 0.016
SWITCH_LEVER_PIVOT_NORMAL_M = 0.031094935607910157
SWITCH_LEVER_TOP_NORMAL_M = 0.09216401397705079
SWITCH_HANDLE_COLLIDER_RADIUS_M = 0.0115


@configclass
class PanelControlArticulationCfg(SpawnerCfg):
    """Spawner configuration for one revolute or prismatic panel control."""

    func = None
    kind: str = "knob"
    slope_tangent: tuple[float, float, float] = (0.8, 0.0, 0.6)
    surface_normal: tuple[float, float, float] = (-0.6, 0.0, 0.8)
    goal: float = math.radians(72.0)
    radius_m: float = 0.022
    length_m: float = 0.036
    mass_kg: float = 0.06


@configclass
class PanelConsoleCollisionCfg(SpawnerCfg):
    """Kinematic collision hull for the fixed five-sided console housing.

    The hull shares the exact X-Z outline authored for the visual console, so
    the operator face ends at the articulation pivots instead of cutting
    through the protruding knob/button colliders.  A separate box collider
    behind the sloped face used to overlap those colliders by tens of
    millimetres and freeze the button's prismatic joint at episode start.
    """

    func = None
    console_depth_m: float = 0.190
    console_width_m: float = 0.320
    console_height_m: float = 0.180
    front_height_m: float = 0.055
    rear_flat_depth_m: float = 0.025


def _add(a, b):
    return tuple(float(a[index]) + float(b[index]) for index in range(3))


def _scale(vector, factor):
    return tuple(float(factor) * float(value) for value in vector)


def _visual_binary_stl_mesh(
    stage: Any,
    path: str,
    stl_path: Path,
    tangent,
    lateral,
    normal,
    *,
    base_normal_m: float,
    in_plane_rotation_rad: float,
    color: tuple[float, float, float],
    scale_m_per_source_unit: float = 0.001,
    source_origin: tuple[float, float, float] | None = None,
) -> None:
    """Author a collision-free USD mesh from the original binary STL triangles."""

    from pxr import Gf, UsdGeom, UsdPhysics

    data = stl_path.read_bytes()
    if len(data) < 84:
        raise ValueError(f"binary STL is truncated: {stl_path}")
    triangle_count = struct.unpack_from("<I", data, 80)[0]
    expected_bytes = 84 + 50 * triangle_count
    if len(data) != expected_bytes:
        raise ValueError(
            f"binary STL size mismatch for {stl_path}: "
            f"expected {expected_bytes}, found {len(data)}"
        )

    triangles = []
    source_normals = []
    lowest_height_mm = math.inf
    for triangle_index in range(triangle_count):
        record = struct.unpack_from("<12fH", data, 84 + 50 * triangle_index)
        source_normals.append(record[:3])
        vertices = (
            record[3:6],
            record[6:9],
            record[9:12],
        )
        triangles.append(vertices)
        lowest_height_mm = min(lowest_height_mm, *(vertex[1] for vertex in vertices))

    if source_origin is None:
        source_origin = (0.0, lowest_height_mm, 0.0)
    rotation_cos = math.cos(in_plane_rotation_rad)
    rotation_sin = math.sin(in_plane_rotation_rad)

    def mapped(source_xyz, *, direction: bool = False):
        source_x, source_y, source_z = source_xyz
        if not direction:
            source_x -= source_origin[0]
            source_y -= source_origin[1]
            source_z -= source_origin[2]
        tangent_mm = rotation_cos * source_x - rotation_sin * source_z
        lateral_mm = rotation_sin * source_x + rotation_cos * source_z
        height_m = source_y
        normal_offset = 0.0 if direction else base_normal_m
        return tuple(
            scale_m_per_source_unit
            * (
                tangent_mm * tangent[axis]
                + lateral_mm * lateral[axis]
                + height_m * normal[axis]
            )
            + normal_offset * normal[axis]
            for axis in range(3)
        )

    source_points = []
    point_indices = {}
    faces = []
    for triangle in triangles:
        face = []
        for vertex in triangle:
            point_index = point_indices.get(vertex)
            if point_index is None:
                point_index = len(source_points)
                point_indices[vertex] = point_index
                source_points.append(vertex)
            face.append(point_index)
        faces.append(tuple(face))
    points = [mapped(vertex) for vertex in source_points]
    mapped_normals = [mapped(source_normal, direction=True) for source_normal in source_normals]
    # Direction vectors were scaled along with positions; normalize them for
    # stable lighting while preserving the original STL facet normals.
    mapped_normals = [
        tuple(component / math.sqrt(sum(value * value for value in vector)) for component in vector)
        for vector in mapped_normals
    ]

    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr([Gf.Vec3f(*point) for point in points])
    mesh.CreateFaceVertexCountsAttr([3] * triangle_count)
    mesh.CreateFaceVertexIndicesAttr([index for face in faces for index in face])
    mesh.CreateNormalsAttr([Gf.Vec3f(*value) for value in mapped_normals])
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.uniform)
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr(True)
    gprim = UsdGeom.Gprim(mesh)
    gprim.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    gprim.CreateDisplayOpacityAttr([1.0])
    collision = UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    collision.CreateCollisionEnabledAttr(False)


def _surface_quat(tangent) -> Any:
    """Quaternion mapping local Z to the outward surface normal."""

    from pxr import Gf

    angle = -math.atan2(float(tangent[2]), float(tangent[0]))
    return Gf.Quatf(math.cos(0.5 * angle), Gf.Vec3f(0.0, math.sin(0.5 * angle), 0.0))


def _set_disable_gravity(prim: Any) -> None:
    """Author the standard rigid-body gravity flag without a Kit dependency."""

    from pxr import Sdf

    prim.CreateAttribute("physics:disableGravity", Sdf.ValueTypeNames.Bool).Set(True)


def _collision_cylinder(
    stage: Any,
    path: str,
    radius: float,
    height: float,
    center,
    orientation,
) -> None:
    from pxr import Gf, UsdGeom, UsdPhysics

    cylinder = UsdGeom.Cylinder.Define(stage, path)
    cylinder.CreateAxisAttr(UsdGeom.Tokens.z)
    cylinder.CreateRadiusAttr(float(radius))
    cylinder.CreateHeightAttr(float(height))
    xform = UsdGeom.Xformable(cylinder)
    xform.AddTranslateOp().Set(Gf.Vec3d(*center))
    xform.AddOrientOp().Set(orientation)
    UsdPhysics.CollisionAPI.Apply(cylinder.GetPrim())
    # Collision remains active, but this simple proxy is hidden behind the
    # collision-free approved render geometry.
    UsdGeom.Imageable(cylinder.GetPrim()).CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)


def _collision_sphere(stage: Any, path: str, radius: float, center) -> None:
    """Create an invisible spherical collision proxy on the current link."""

    from pxr import Gf, UsdGeom, UsdPhysics

    sphere = UsdGeom.Sphere.Define(stage, path)
    sphere.CreateRadiusAttr(float(radius))
    UsdGeom.Xformable(sphere).AddTranslateOp().Set(Gf.Vec3d(*center))
    UsdPhysics.CollisionAPI.Apply(sphere.GetPrim())
    UsdGeom.Imageable(sphere.GetPrim()).CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)


def _collision_extruded_prism(stage: Any, path: str, outline_xz, width_m: float) -> None:
    """Author an invisible closed prism collider extruded along local Y.

    The five-sided console outline is convex, so MJWarp's convex-hull import
    preserves it exactly.  The prism is invisible: the richer visual console
    lives in ``ControlPanel/Appearance`` and must stay collision-free.
    """

    from pxr import Gf, UsdGeom, UsdPhysics

    half_width = 0.5 * float(width_m)
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
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr([Gf.Vec3f(*point) for point in points])
    mesh.CreateFaceVertexCountsAttr([len(face) for face in faces])
    mesh.CreateFaceVertexIndicesAttr([index for face in faces for index in face])
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr(True)
    collision = UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    collision.CreateCollisionEnabledAttr(True)
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr(
        UsdPhysics.Tokens.convexHull
    )
    UsdGeom.Imageable(mesh.GetPrim()).CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)


@clone
def spawn_panel_console_collision(
    prim_path: str,
    cfg: PanelConsoleCollisionCfg,
    translation=None,
    orientation=None,
    **_: Any,
):
    """Spawn the kinematic console hull as one convex collision body."""

    from pxr import Gf, UsdPhysics

    stage = get_current_stage()
    root = create_prim(
        prim_path,
        "Xform",
        translation=translation,
        orientation=orientation,
        stage=stage,
    )
    rigid = UsdPhysics.RigidBodyAPI.Apply(root)
    rigid.CreateKinematicEnabledAttr(True)
    rigid.CreateRigidBodyEnabledAttr(True)
    mass = UsdPhysics.MassAPI.Apply(root)
    mass.CreateMassAttr(2.5)
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(0.018, 0.026, 0.018))
    mass.CreatePrincipalAxesAttr(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))

    depth = float(cfg.console_depth_m)
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
    _collision_extruded_prism(
        stage,
        f"{prim_path}/geometry/mesh",
        outline,
        float(cfg.console_width_m),
    )
    return root


def _define_joint(
    stage: Any,
    path: str,
    kind: str,
    body0: str,
    body1: str,
    frame_quat,
    goal: float,
    joint_position=(0.0, 0.0, 0.0),
) -> None:
    from pxr import Gf, Sdf, UsdPhysics

    if kind == "button":
        joint = UsdPhysics.PrismaticJoint.Define(stage, path)
        joint.CreateAxisAttr(UsdPhysics.Tokens.z)
        # Keep a little travel beyond the task threshold.  The controller
        # retreats as soon as the requested travel is reached, so this avoids
        # loading an infinitely stiff mechanical stop on the completion frame.
        joint.CreateLowerLimitAttr(-1.15 * float(goal))
        joint.CreateUpperLimitAttr(0.0)
        drive_name = "linear"
        max_force = 30.0
        stiffness = 180.0
        damping = 2.5
    else:
        joint = UsdPhysics.RevoluteJoint.Define(stage, path)
        joint.CreateAxisAttr(UsdPhysics.Tokens.z if kind == "knob" else UsdPhysics.Tokens.y)
        joint.CreateLowerLimitAttr(0.0)
        # The task goal is deliberately inside the hard limit.  Contact-driven
        # manipulation otherwise produces a force spike exactly when success
        # is detected, even with sub-millimetre penetration.
        joint.CreateUpperLimitAttr(math.degrees(1.10 * float(goal)))
        drive_name = "angular"
        max_force = 2.0
        stiffness = 0.0
        damping = 0.08 if kind == "knob" else 0.05
    joint.CreateBody0Rel().SetTargets([Sdf.Path(body0)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(body1)])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*joint_position))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(*joint_position))
    joint.CreateLocalRot0Attr().Set(frame_quat)
    joint.CreateLocalRot1Attr().Set(frame_quat)
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), drive_name)
    drive.CreateTypeAttr(UsdPhysics.Tokens.force)
    drive.CreateMaxForceAttr(max_force)
    drive.CreateTargetPositionAttr(0.0)
    drive.CreateStiffnessAttr(stiffness)
    drive.CreateDampingAttr(damping)


@clone
def spawn_panel_control_articulation(
    prim_path: str,
    cfg: PanelControlArticulationCfg,
    translation=None,
    orientation=None,
    **_: Any,
):
    """Spawn one physical panel control with a simple convex collider."""

    from pxr import UsdPhysics

    if cfg.kind not in ("knob", "lever", "button"):
        raise ValueError(f"unknown panel control kind: {cfg.kind}")
    stage = get_current_stage()
    root = create_prim(
        prim_path,
        "Xform",
        translation=translation,
        orientation=orientation,
        stage=stage,
    )
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(0.20)
    UsdPhysics.ArticulationRootAPI.Apply(root)
    _set_disable_gravity(root)

    link_path = f"{prim_path}/{cfg.kind}_link"
    link = create_prim(link_path, "Xform", stage=stage)
    UsdPhysics.RigidBodyAPI.Apply(link)
    UsdPhysics.MassAPI.Apply(link).CreateMassAttr(float(cfg.mass_kg))
    _set_disable_gravity(link)

    tangent = tuple(float(value) for value in cfg.slope_tangent)
    normal = tuple(float(value) for value in cfg.surface_normal)
    lateral = (0.0, 1.0, 0.0)
    frame_quat = _surface_quat(tangent)
    _define_joint(
        stage,
        f"{prim_path}/{cfg.kind}_joint",
        cfg.kind,
        prim_path,
        link_path,
        frame_quat,
        cfg.goal,
        joint_position=(
            _scale(normal, SWITCH_LEVER_PIVOT_NORMAL_M)
            if cfg.kind == "lever"
            else (0.0, 0.0, 0.0)
        ),
    )

    if cfg.kind == "knob":
        # The former 22 x 20 mm proxy was smaller than the Apollo visual by
        # 3.74 mm radially and 0.70 mm in height.  Contacts were therefore
        # physically valid while a finger visibly penetrated the render mesh.
        # This cylinder conservatively encloses the certified runtime LOD.
        collider_center = _scale(
            normal,
            APOLLO_KNOB_VISUAL_BASE_NORMAL_M + 0.5 * APOLLO_KNOB_COLLIDER_HEIGHT_M,
        )
        _collision_cylinder(
            stage,
            f"{link_path}/collision",
            max(float(cfg.radius_m), APOLLO_KNOB_COLLIDER_RADIUS_M),
            APOLLO_KNOB_COLLIDER_HEIGHT_M,
            collider_center,
            frame_quat,
        )
        _visual_binary_stl_mesh(
            stage,
            f"{link_path}/ApolloKnob",
            APOLLO_KNOB_RUNTIME_STL_PATH,
            tangent,
            lateral,
            normal,
            base_normal_m=APOLLO_KNOB_VISUAL_BASE_NORMAL_M,
            in_plane_rotation_rad=math.pi,
            color=(0.31, 0.32, 0.31),
        )
    elif cfg.kind == "lever":
        # The source switch has a fixed threaded body/hex collar and a separate
        # moving toggle.  Keeping the 28.1 mm-radius base on the articulation
        # root is essential: rotating that wide collar by 30 degrees would
        # sweep it through the operator face even when its rest pose is clear.
        _visual_binary_stl_mesh(
            stage,
            f"{prim_path}/SwitchBase",
            SWITCH_BASE_RUNTIME_STL_PATH,
            tangent,
            lateral,
            normal,
            base_normal_m=SWITCH_BASE_NORMAL_M,
            in_plane_rotation_rad=0.0,
            color=(0.34, 0.36, 0.37),
            scale_m_per_source_unit=SWITCH_SCALE_M_PER_SOURCE_UNIT,
            source_origin=SWITCH_SOURCE_ORIGIN,
        )
        collider_height = SWITCH_LEVER_TOP_NORMAL_M - SWITCH_LEVER_PIVOT_NORMAL_M
        _collision_cylinder(
            stage,
            f"{link_path}/collision",
            max(float(cfg.radius_m), SWITCH_HANDLE_COLLIDER_RADIUS_M),
            collider_height,
            _scale(
                normal,
                SWITCH_LEVER_PIVOT_NORMAL_M + 0.5 * collider_height,
            ),
            frame_quat,
        )
        _visual_binary_stl_mesh(
            stage,
            f"{link_path}/SwitchHandle",
            SWITCH_HANDLE_RUNTIME_STL_PATH,
            tangent,
            lateral,
            normal,
            base_normal_m=SWITCH_BASE_NORMAL_M,
            in_plane_rotation_rad=0.0,
            color=(0.055, 0.060, 0.065),
            scale_m_per_source_unit=SWITCH_SCALE_M_PER_SOURCE_UNIT,
            source_origin=SWITCH_SOURCE_ORIGIN,
        )
    else:
        _collision_cylinder(
            stage,
            f"{link_path}/collision",
            cfg.radius_m,
            cfg.length_m,
            _scale(normal, 0.5 * cfg.length_m + 0.018),
            frame_quat,
        )
        _visual_cylinder_between(
            stage,
            f"{link_path}/WitnessBand",
            _scale(normal, 0.021),
            _scale(normal, 0.026),
            0.019,
            (0.96, 0.62, 0.08),
        )
        _visual_cylinder_between(
            stage,
            f"{link_path}/Cap",
            _scale(normal, 0.026),
            _scale(normal, 0.038),
            0.020,
            (0.82, 0.075, 0.050),
        )
        _visual_cylinder_between(
            stage,
            f"{link_path}/Face",
            _scale(normal, 0.038),
            _scale(normal, 0.041),
            0.0175,
            (0.96, 0.13, 0.08),
        )
    return root
