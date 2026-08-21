"""Collision-free visual Stewart platform and analytic leg kinematics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

from isaaclab.assets import RigidObjectCfg, RigidObjectCollectionCfg
from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
from isaaclab.sim.utils import clone, create_prim, get_current_stage
from isaaclab.utils.configclass import configclass


@configclass
class ShakerGeometryCfg:
    """All geometry and stroke limits for the visual 6-3 Stewart platform."""

    base_joint_semi_major: float = 0.85
    base_joint_semi_minor: float = 0.60
    platen_joint_semi_major: float = 0.62
    platen_joint_semi_minor: float = 0.40
    joint_pair_spread: float = math.radians(20.0)
    leg_stroke_nominal: float = 0.82
    leg_stroke_min: float = 0.67
    leg_stroke_max: float = 0.95
    cylinder_radius: float = 0.050
    rod_radius: float = 0.030
    cylinder_length: float = 0.48
    rod_length: float = 0.46
    joint_radius: float = 0.036
    base_center: tuple[float, float, float] = (0.0, 0.0, -0.50)
    platen_joint_z: float = -0.086
    base_plate_radius: float = 0.88
    base_plate_height: float = 0.10
    inertia_base_size: tuple[float, float, float] = (1.85, 1.35, 0.18)
    inertia_base_z: float = -0.69
    air_spring_radius: float = 0.105
    air_spring_height: float = 0.12
    air_spring_ring_radius: float = 0.68
    cylinder_rgb: tuple[float, float, float] = (0.060, 0.065, 0.067)
    rod_rgb: tuple[float, float, float] = (0.16, 0.165, 0.17)
    joint_rgb: tuple[float, float, float] = (0.040, 0.043, 0.045)
    base_rgb: tuple[float, float, float] = (0.050, 0.053, 0.055)
    inertia_base_rgb: tuple[float, float, float] = (0.055, 0.058, 0.060)
    air_spring_rgb: tuple[float, float, float] = (0.030, 0.032, 0.033)

    def __post_init__(self) -> None:
        if not 0.0 < self.joint_pair_spread < math.pi / 3.0:
            raise ValueError("joint_pair_spread must be in (0, pi/3)")
        if not self.leg_stroke_min < self.leg_stroke_nominal < self.leg_stroke_max:
            raise ValueError("leg_stroke_nominal must lie strictly inside the stroke limits")
        if min(self.cylinder_radius, self.rod_radius, self.cylinder_length, self.rod_length) <= 0.0:
            raise ValueError("actuator radii and lengths must be positive")
        report = platen_joint_clearance_report(self)
        failed = {name: value for name, value in report.items() if value < 0.0}
        if failed:
            raise ValueError(f"Stewart-to-platen clearance violation: {failed}")


@dataclass(frozen=True)
class LegTransforms:
    """Batched actuator poses, in Isaac Lab ``(x, y, z, w)`` convention."""

    outer_pose_xyzw: torch.Tensor
    rod_pose_xyzw: torch.Tensor
    lengths_m: torch.Tensor


def _joint_angles(cfg: ShakerGeometryCfg, *, platen: bool, device, dtype) -> torch.Tensor:
    offset = math.pi / 3.0 if platen else 0.0
    values = []
    for group in range(3):
        centre = offset + group * 2.0 * math.pi / 3.0
        values.extend((centre - 0.5 * cfg.joint_pair_spread, centre + 0.5 * cfg.joint_pair_spread))
    return torch.tensor(values, device=device, dtype=dtype)


def joint_points(cfg: ShakerGeometryCfg, *, device="cpu", dtype=torch.float64) -> tuple[torch.Tensor, torch.Tensor]:
    """Return base and platen joint points in their respective local frames."""

    base_angles = _joint_angles(cfg, platen=False, device=device, dtype=dtype)
    platen_angles = _joint_angles(cfg, platen=True, device=device, dtype=dtype)
    base = torch.stack(
        (
            cfg.base_joint_semi_major * torch.cos(base_angles),
            cfg.base_joint_semi_minor * torch.sin(base_angles),
            torch.full_like(base_angles, cfg.base_center[2]),
        ),
        dim=-1,
    )
    base[..., 0] += cfg.base_center[0]
    base[..., 1] += cfg.base_center[1]
    platen = torch.stack(
        (
            cfg.platen_joint_semi_major * torch.cos(platen_angles),
            cfg.platen_joint_semi_minor * torch.sin(platen_angles),
            torch.full_like(platen_angles, cfg.platen_joint_z),
        ),
        dim=-1,
    )
    return base, platen


def platen_joint_clearance_report(
    cfg: ShakerGeometryCfg,
    platform_size: tuple[float, float, float] = (1.60, 1.10, 0.08),
    margin_m: float = 0.06,
) -> dict[str, float]:
    """Return signed joint-to-platen clearances; negative values are violations."""

    _, platen = joint_points(cfg, dtype=torch.float64)
    half_x, half_y, half_z = (0.5 * value for value in platform_size)
    return {
        "x_clearance_m": float(half_x - torch.max(torch.abs(platen[:, 0])).item() - cfg.cylinder_radius - margin_m),
        "y_clearance_m": float(half_y - torch.max(torch.abs(platen[:, 1])).item() - cfg.cylinder_radius - margin_m),
        "z_clearance_m": float((-half_z - cfg.joint_radius) - torch.max(platen[:, 2]).item()),
    }


def actuator_platen_overlap_violations(
    platen_pose_xyzw: torch.Tensor,
    cfg: ShakerGeometryCfg,
    platform_size: tuple[float, float, float] = (1.60, 1.10, 0.08),
) -> list[dict[str, float | int]]:
    """Find actuator centreline samples that enter the platen AABB.

    Samples are evaluated in platen-local coordinates and the AABB is expanded
    by each segment radius.  This deliberately conservative test catches the
    visible rod-through-skirt failure without depending on raster output.
    """

    solved = solve_leg_transforms(platen_pose_xyzw, cfg)
    position = platen_pose_xyzw[..., :3]
    quaternion = platen_pose_xyzw[..., 3:]
    quaternion = quaternion / torch.linalg.norm(quaternion, dim=-1, keepdim=True)
    inverse = torch.cat((-quaternion[..., :3], quaternion[..., 3:]), dim=-1)
    half = torch.tensor(platform_size, dtype=position.dtype, device=position.device) * 0.5
    violations: list[dict[str, float | int]] = []
    for kind, poses, length, radius in (
        ("outer", solved.outer_pose_xyzw, cfg.cylinder_length, cfg.cylinder_radius),
        ("rod", solved.rod_pose_xyzw, cfg.rod_length, cfg.rod_radius),
    ):
        direction = _quat_apply_xyzw(
            poses[..., 3:],
            torch.tensor((0.0, 0.0, 1.0), dtype=position.dtype, device=position.device).expand_as(poses[..., :3]),
        )
        for sample_index, fraction in enumerate(torch.linspace(-0.5, 0.5, 25, device=position.device, dtype=position.dtype)):
            world = poses[..., :3] + fraction * length * direction
            relative = world - position.unsqueeze(-2)
            local = _quat_apply_xyzw(inverse.unsqueeze(-2).expand_as(poses[..., 3:]), relative)
            inside = (
                (torch.abs(local[..., 0]) <= half[0] + radius)
                & (torch.abs(local[..., 1]) <= half[1] + radius)
                & (torch.abs(local[..., 2]) <= half[2] + radius)
            )
            for index in torch.nonzero(inside, as_tuple=False):
                batch_index = int(index[0].item()) if inside.ndim == 2 else 0
                leg_index = int(index[-1].item())
                violations.append(
                    {
                        "batch": batch_index,
                        "leg": leg_index,
                        "sample": sample_index,
                        "kind": kind,
                        "local_x_m": float(local[tuple(index)][0].item()),
                        "local_y_m": float(local[tuple(index)][1].item()),
                        "local_z_m": float(local[tuple(index)][2].item()),
                    }
                )
    return violations


def _quat_apply_xyzw(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    q_xyz = quaternion[..., :3]
    uv = torch.linalg.cross(q_xyz, vector, dim=-1)
    uuv = torch.linalg.cross(q_xyz, uv, dim=-1)
    return vector + 2.0 * (quaternion[..., 3:] * uv + uuv)


def _align_z_quaternion(direction: torch.Tensor) -> torch.Tensor:
    """Return a quaternion that maps local +Z onto ``direction``."""

    z = torch.zeros_like(direction)
    z[..., 2] = 1.0
    xyz = torch.linalg.cross(z, direction, dim=-1)
    w = 1.0 + direction[..., 2:3]
    quaternion = torch.cat((xyz, w), dim=-1)
    norm = torch.linalg.norm(quaternion, dim=-1, keepdim=True)
    fallback = torch.zeros_like(quaternion)
    fallback[..., 0] = 1.0
    return torch.where(norm > 1.0e-8, quaternion / torch.clamp_min(norm, 1.0e-8), fallback)


def solve_leg_transforms(platen_pose_xyzw: torch.Tensor, cfg: ShakerGeometryCfg) -> LegTransforms:
    """Solve all six visual actuators from a platen centre pose analytically.

    Args:
        platen_pose_xyzw: Tensor shaped ``(..., 7)`` containing position and
            a normalized Isaac Lab ``(x, y, z, w)`` quaternion.
        cfg: Stewart geometry and stroke configuration.

    Raises:
        ValueError: If a quaternion is invalid or any leg exceeds its stroke.
    """

    if platen_pose_xyzw.shape[-1] != 7:
        raise ValueError("platen_pose_xyzw must have seven components")
    position = platen_pose_xyzw[..., :3]
    quaternion = platen_pose_xyzw[..., 3:]
    q_norm = torch.linalg.norm(quaternion, dim=-1, keepdim=True)
    if torch.any(q_norm < 1.0e-8):
        raise ValueError("platen quaternion must be non-zero")
    quaternion = quaternion / q_norm
    base, platen_local = joint_points(cfg, device=position.device, dtype=position.dtype)
    batch_shape = position.shape[:-1]
    base = base.expand(*batch_shape, 6, 3)
    platen_local = platen_local.expand(*batch_shape, 6, 3)
    expanded_quaternion = quaternion.unsqueeze(-2).expand(*batch_shape, 6, 4)
    platen_world = position.unsqueeze(-2) + _quat_apply_xyzw(expanded_quaternion, platen_local)
    vector = platen_world - base
    lengths = torch.linalg.norm(vector, dim=-1)
    invalid = (lengths < cfg.leg_stroke_min) | (lengths > cfg.leg_stroke_max)
    if torch.any(invalid):
        index = torch.nonzero(invalid, as_tuple=False)[0]
        leg_index = int(index[-1].item())
        batch_index = tuple(int(value.item()) for value in index[:-1])
        pose = platen_pose_xyzw[batch_index] if batch_index else platen_pose_xyzw
        length = float(lengths[tuple(index)].item())
        raise ValueError(
            f"Stewart leg {leg_index} stroke violation: length={length:.6f} m, "
            f"limits=[{cfg.leg_stroke_min:.6f}, {cfg.leg_stroke_max:.6f}] m, "
            f"platen_pose_xyzw={pose.detach().cpu().tolist()}"
        )
    direction = vector / lengths.unsqueeze(-1)
    orientation = _align_z_quaternion(direction)
    outer_position = base + 0.5 * cfg.cylinder_length * direction
    rod_position = platen_world - 0.5 * cfg.rod_length * direction
    return LegTransforms(
        outer_pose_xyzw=torch.cat((outer_position, orientation), dim=-1),
        rod_pose_xyzw=torch.cat((rod_position, orientation), dim=-1),
        lengths_m=lengths,
    )


@configclass
class ShakerLegSegmentCfg(SpawnerCfg):
    """One kinematic, collision-disabled actuator segment."""

    func = None
    length_m: float = 0.48
    radius_m: float = 0.052
    joint_radius_m: float = 0.072
    joint_at_positive_end: bool = False
    segment_rgb: tuple[float, float, float] = (0.10, 0.13, 0.17)
    joint_rgb: tuple[float, float, float] = (0.18, 0.21, 0.25)
    roughness: float = 0.28
    metallic: float = 0.72


def _set_display(prim, color: tuple[float, float, float]) -> None:
    from pxr import Gf, UsdGeom

    gprim = UsdGeom.Gprim(prim)
    gprim.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    gprim.CreateDisplayOpacityAttr([1.0])


def _disable_collision(prim) -> None:
    from pxr import UsdPhysics

    collision = UsdPhysics.CollisionAPI.Apply(prim)
    collision.CreateCollisionEnabledAttr(False)


def _hex_prism(stage: Any, path: str, radius: float, height: float, z: float, color) -> None:
    """Author a low-cost closed hexagonal prism centered on the Z axis."""

    from pxr import Gf, UsdGeom

    bottom = [
        Gf.Vec3f(radius * math.cos(i * math.pi / 3.0), radius * math.sin(i * math.pi / 3.0), z - 0.5 * height)
        for i in range(6)
    ]
    top = [Gf.Vec3f(point[0], point[1], z + 0.5 * height) for point in bottom]
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(bottom + top)
    mesh.CreateFaceVertexCountsAttr([6, 6, 4, 4, 4, 4, 4, 4])
    indices = [5, 4, 3, 2, 1, 0, 6, 7, 8, 9, 10, 11]
    for index in range(6):
        nxt = (index + 1) % 6
        indices.extend((index, nxt, 6 + nxt, 6 + index))
    mesh.CreateFaceVertexIndicesAttr(indices)
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    _set_display(mesh.GetPrim(), color)
    _disable_collision(mesh.GetPrim())


def _cylinder_between(stage: Any, path: str, start, end, radius: float, color) -> None:
    """Author a collision-free cylinder aligned between two world-local points."""

    from pxr import Gf, UsdGeom

    vector = tuple(float(end[i] - start[i]) for i in range(3))
    length = math.sqrt(sum(value * value for value in vector))
    direction = tuple(value / length for value in vector)
    q_xyz = (-direction[1], direction[0], 0.0)
    q_w = 1.0 + direction[2]
    q_norm = math.sqrt(q_w * q_w + sum(value * value for value in q_xyz))
    if q_norm < 1.0e-8:
        q_w, q_xyz, q_norm = 0.0, (1.0, 0.0, 0.0), 1.0
    cylinder = UsdGeom.Cylinder.Define(stage, path)
    cylinder.CreateAxisAttr(UsdGeom.Tokens.z)
    cylinder.CreateRadiusAttr(radius)
    cylinder.CreateHeightAttr(length)
    xform = UsdGeom.Xformable(cylinder)
    xform.AddTranslateOp().Set(Gf.Vec3d(*(0.5 * (float(start[i]) + float(end[i])) for i in range(3))))
    xform.AddOrientOp().Set(
        Gf.Quatf(
            q_w / q_norm,
            Gf.Vec3f(*(value / q_norm for value in q_xyz)),
        )
    )
    _set_display(cylinder.GetPrim(), color)
    _disable_collision(cylinder.GetPrim())


@clone
def spawn_shaker_leg_segment(
    prim_path: str,
    cfg: ShakerLegSegmentCfg,
    translation=None,
    orientation=None,
    **_: Any,
):
    """Author one renderable actuator body excluded from MJWarp contacts."""

    from pxr import Gf, UsdGeom, UsdPhysics

    stage = get_current_stage()
    create_prim(prim_path, "Xform", translation=translation, orientation=orientation, stage=stage)
    root = stage.GetPrimAtPath(prim_path)
    rigid = UsdPhysics.RigidBodyAPI.Apply(root)
    rigid.CreateKinematicEnabledAttr(True)
    rigid.CreateRigidBodyEnabledAttr(True)
    # Newton requires a complete body declaration even for kinematic visual
    # followers. The mass is never integrated, but prevents the importer from
    # discarding an otherwise collision-disabled rigid root.
    mass = UsdPhysics.MassAPI.Apply(root)
    mass.CreateMassAttr(1.0)
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(0.01, 0.01, 0.01))
    mass.CreatePrincipalAxesAttr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

    cylinder = UsdGeom.Cylinder.Define(stage, f"{prim_path}/Segment")
    cylinder.CreateAxisAttr(UsdGeom.Tokens.z)
    cylinder.CreateHeightAttr(cfg.length_m)
    cylinder.CreateRadiusAttr(cfg.radius_m)
    _set_display(cylinder.GetPrim(), cfg.segment_rgb)
    _disable_collision(cylinder.GetPrim())

    joint = UsdGeom.Sphere.Define(stage, f"{prim_path}/UniversalJoint")
    joint.CreateRadiusAttr(cfg.joint_radius_m)
    endpoint = 0.5 * cfg.length_m if cfg.joint_at_positive_end else -0.5 * cfg.length_m
    UsdGeom.Xformable(joint).AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, endpoint))
    _set_display(joint.GetPrim(), cfg.joint_rgb)
    _disable_collision(joint.GetPrim())
    return root


@configclass
class ShakerBaseVisualCfg(SpawnerCfg):
    """World-fixed base plate, inertia block and six air springs."""

    func = None
    geometry: ShakerGeometryCfg = ShakerGeometryCfg()


@clone
def spawn_shaker_base_visuals(
    prim_path: str,
    cfg: ShakerBaseVisualCfg,
    translation=None,
    orientation=None,
    **_: Any,
):
    """Author the static visual foundation with collision explicitly disabled."""

    from pxr import Gf, UsdGeom

    stage = get_current_stage()
    create_prim(prim_path, "Xform", translation=translation, orientation=orientation, stage=stage)
    geometry = cfg.geometry

    block = UsdGeom.Cube.Define(stage, f"{prim_path}/InertiaBase")
    block.CreateSizeAttr(1.0)
    block_xform = UsdGeom.Xformable(block)
    block_xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, geometry.inertia_base_z))
    block_xform.AddScaleOp().Set(Gf.Vec3d(*geometry.inertia_base_size))
    _set_display(block.GetPrim(), geometry.inertia_base_rgb)
    _disable_collision(block.GetPrim())

    _hex_prism(
        stage,
        f"{prim_path}/HexBasePlate",
        geometry.base_plate_radius,
        geometry.base_plate_height,
        geometry.base_center[2],
        geometry.base_rgb,
    )
    base_points, _ = joint_points(geometry)
    base_points = base_points.detach().cpu().tolist()
    for index, point in enumerate(base_points):
        flange = UsdGeom.Cylinder.Define(stage, f"{prim_path}/JointFlange_{index}")
        flange.CreateAxisAttr(UsdGeom.Tokens.z)
        flange.CreateRadiusAttr(2.8 * geometry.rod_radius)
        flange.CreateHeightAttr(0.035)
        UsdGeom.Xformable(flange).AddTranslateOp().Set(Gf.Vec3d(point[0], point[1], point[2] + 0.025))
        _set_display(flange.GetPrim(), geometry.joint_rgb)
        _disable_collision(flange.GetPrim())
    for index, (lhs, rhs) in enumerate(((0, 3), (1, 4), (2, 5))):
        start = list(base_points[lhs])
        end = list(base_points[rhs])
        start[2] += 0.065
        end[2] += 0.065
        _cylinder_between(
            stage,
            f"{prim_path}/CrossBrace_{index}",
            start,
            end,
            0.022,
            geometry.base_rgb,
        )

    air_z = geometry.inertia_base_z + 0.5 * geometry.inertia_base_size[2] + 0.5 * geometry.air_spring_height
    for index in range(6):
        angle = index * math.pi / 3.0
        air = UsdGeom.Cylinder.Define(stage, f"{prim_path}/AirSpring_{index}")
        air.CreateAxisAttr(UsdGeom.Tokens.z)
        air.CreateRadiusAttr(geometry.air_spring_radius)
        air.CreateHeightAttr(geometry.air_spring_height)
        UsdGeom.Xformable(air).AddTranslateOp().Set(
            Gf.Vec3d(
                geometry.air_spring_ring_radius * math.cos(angle),
                geometry.air_spring_ring_radius * math.sin(angle),
                air_z,
            )
        )
        _set_display(air.GetPrim(), geometry.air_spring_rgb)
        _disable_collision(air.GetPrim())
    return stage.GetPrimAtPath(prim_path)


@configclass
class ShadowFollowerCfg(SpawnerCfg):
    """One kinematic square contact cue, sharing the Stewart collection."""

    func = None
    size_xy: tuple[float, float] = (0.070, 0.050)
    surface_rgb: tuple[float, float, float] = (0.46, 0.47, 0.48)


@clone
def spawn_shadow_follower(
    prim_path: str,
    cfg: ShadowFollowerCfg,
    translation=None,
    orientation=None,
    **_: Any,
):
    from pxr import Gf, UsdGeom, UsdPhysics

    stage = get_current_stage()
    create_prim(prim_path, "Xform", translation=translation, orientation=orientation, stage=stage)
    root = stage.GetPrimAtPath(prim_path)
    rigid = UsdPhysics.RigidBodyAPI.Apply(root)
    rigid.CreateKinematicEnabledAttr(True)
    rigid.CreateRigidBodyEnabledAttr(True)
    mass = UsdPhysics.MassAPI.Apply(root)
    mass.CreateMassAttr(0.01)
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(1.0e-5, 1.0e-5, 1.0e-5))
    mass.CreatePrincipalAxesAttr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    for layer, scale in enumerate((0.48, 0.73, 1.0)):
        cube = UsdGeom.Cube.Define(stage, f"{prim_path}/Patch_{layer}")
        cube.CreateSizeAttr(1.0)
        xform = UsdGeom.Xformable(cube)
        xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, layer * 0.0005))
        xform.AddScaleOp().Set(Gf.Vec3d(cfg.size_xy[0] * scale, cfg.size_xy[1] * scale, 0.0007))
        shade = (0.18, 0.38, 0.65)[layer]
        _set_display(cube.GetPrim(), tuple(shade * value for value in cfg.surface_rgb))
        _disable_collision(cube.GetPrim())
    return root


def make_shaker_leg_collection_cfg(
    cfg: ShakerGeometryCfg,
    nominal_platen_center: tuple[float, float, float],
    shadow_initial_position: tuple[float, float, float] | None = None,
) -> RigidObjectCollectionCfg:
    """Build the twelve ordered rigid visual followers for an Isaac Lab scene."""

    nominal_pose = torch.tensor([(*nominal_platen_center, 0.0, 0.0, 0.0, 1.0)], dtype=torch.float64)
    transforms = solve_leg_transforms(nominal_pose, cfg)
    rigid_objects: dict[str, RigidObjectCfg] = {}
    for kind, poses in (("outer", transforms.outer_pose_xyzw[0]), ("rod", transforms.rod_pose_xyzw[0])):
        for index in range(6):
            is_rod = kind == "rod"
            pose = poses[index].tolist()
            rigid_objects[f"{kind}_{index}"] = RigidObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/ShakerLegs/{kind.capitalize()}_{index}",
                spawn=ShakerLegSegmentCfg(
                    func=spawn_shaker_leg_segment,
                    length_m=cfg.rod_length if is_rod else cfg.cylinder_length,
                    radius_m=cfg.rod_radius if is_rod else cfg.cylinder_radius,
                    joint_radius_m=cfg.joint_radius,
                    joint_at_positive_end=is_rod,
                    segment_rgb=cfg.rod_rgb if is_rod else cfg.cylinder_rgb,
                    joint_rgb=cfg.joint_rgb,
                    roughness=0.16 if is_rod else 0.34,
                    metallic=0.90 if is_rod else 0.58,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(pose[:3]), rot=tuple(pose[3:])),
            )
    if shadow_initial_position is not None:
        rigid_objects["workpiece_shadow"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/ShakerLegs/WorkpieceShadow",
            spawn=ShadowFollowerCfg(func=spawn_shadow_follower),
            init_state=RigidObjectCfg.InitialStateCfg(pos=shadow_initial_position),
        )
    return RigidObjectCollectionCfg(rigid_objects=rigid_objects)
