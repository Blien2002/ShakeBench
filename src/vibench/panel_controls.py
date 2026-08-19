"""Procedural one-DoF panel controls for Newton/MJWarp.

Each control is a two-body articulation: a tiny floating base that is moved
with the worktable support frame, plus one collision-enabled moving link.  The
render geometry is deliberately richer than the collider, so the approved
industrial appearance does not make contact generation unnecessarily fragile.
"""

from __future__ import annotations

import math
from typing import Any

from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
from isaaclab.sim.utils import clone, create_prim, get_current_stage
from isaaclab.utils.configclass import configclass

from .visual_assets import (
    _visual_cylinder_between,
    _visual_extruded_polygon,
    _visual_sphere,
)


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
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
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
    )

    if cfg.kind == "knob":
        collider_center = _scale(normal, 0.027)
        _collision_cylinder(
            stage,
            f"{link_path}/collision",
            cfg.radius_m,
            0.034,
            collider_center,
            frame_quat,
        )
        pointer_outline = (
            (-0.022, 0.0),
            (-0.018, 0.010),
            (-0.008, 0.014),
            (0.006, 0.012),
            (0.030, 0.003),
            (0.030, -0.003),
            (0.006, -0.012),
            (-0.008, -0.014),
            (-0.018, -0.010),
        )
        _visual_extruded_polygon(
            stage,
            f"{link_path}/Pointer",
            (0.0, 0.0, 0.0),
            tangent,
            lateral,
            normal,
            pointer_outline,
            0.015,
            0.038,
            (0.68, 0.69, 0.67),
        )
        stripe_start = _add(_scale(tangent, -0.004), _scale(normal, 0.039))
        stripe_end = _add(_scale(tangent, 0.027), _scale(normal, 0.039))
        _visual_cylinder_between(
            stage,
            f"{link_path}/IndexStripe",
            stripe_start,
            stripe_end,
            0.0018,
            (0.94, 0.88, 0.66),
        )
    elif cfg.kind == "lever":
        # The shaft proxy starts 2 mm outward of the pivot instead of flush on
        # the operator face.  A flush end cap generates a small but permanent
        # lever<->panel contact at rest under the zeroed NativeCCD margin.
        _collision_cylinder(
            stage,
            f"{link_path}/collision",
            0.011,
            cfg.length_m,
            _scale(normal, 0.5 * cfg.length_m + 0.002),
            frame_quat,
        )
        shaft_start = _scale(normal, 0.020)
        shaft_end = _scale(normal, 0.074)
        _visual_cylinder_between(
            stage,
            f"{link_path}/Shaft",
            shaft_start,
            shaft_end,
            0.006,
            (0.68, 0.70, 0.70),
        )
        grip_start = _scale(normal, 0.060)
        grip_end = _scale(normal, 0.090)
        _collision_sphere(
            stage,
            f"{link_path}/grip_collision",
            0.011,
            grip_end,
        )
        _visual_cylinder_between(
            stage,
            f"{link_path}/Grip",
            grip_start,
            grip_end,
            0.011,
            (0.055, 0.060, 0.065),
        )
        _visual_sphere(stage, f"{link_path}/GripCap", 0.011, grip_end, (0.055, 0.060, 0.065))
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
            f"{link_path}/Cap",
            _scale(normal, 0.018),
            _scale(normal, 0.033),
            0.020,
            (0.82, 0.075, 0.050),
        )
        _visual_sphere(
            stage,
            f"{link_path}/Dome",
            0.0195,
            _scale(normal, 0.034),
            (0.82, 0.075, 0.050),
        )
    return root
