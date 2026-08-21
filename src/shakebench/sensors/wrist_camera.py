"""Physically modeled Panda wrist camera and NewtonGL sensor frontend."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
from isaaclab.sim.utils import clone, create_prim, get_current_stage
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import quat_apply


# Fixed top mount expressed in Isaac's panda_hand frame.  A camera centred on
# the wrist axis is hidden by the Panda flange when it looks along local +Z.
# The housing is therefore placed just outside the flange silhouette on the
# upper side of the gripper and given a constant 27.8 degree inward pitch.
# Its optical ray intersects the gripper centre at local z=140 mm.  This is one
# rigid hand-camera transform: it never tracks the object and is never
# stabilised in the world frame.
WRIST_CAMERA_FRAME_POS_H = (0.1050, 0.0, -0.0590)
WRIST_CAMERA_AIM_H = (0.0, 0.0, 0.1400)
WRIST_CAMERA_EYE_H = (0.097766738, 0.0, -0.045291247)
WRIST_CAMERA_FORWARD_H = (-0.466662058, 0.0, 0.884435709)
WRIST_CAMERA_UP_H = (0.884435709, 0.0, 0.466662058)
WRIST_CAMERA_ORIENTATION_H = (0.0, -0.240379170, 0.0, 0.970679069)  # xyzw
WRIST_CAMERA_VERTICAL_FOV_DEG = 75.0  # robosuite Panda eye_in_hand fovy


@configclass
class WristCameraAssemblyCfg(SpawnerCfg):
    """Rigidly mounted camera housing, bracket, lens, and colliders."""

    func = None
    body_rgb: tuple[float, float, float] = (0.035, 0.045, 0.055)
    accent_rgb: tuple[float, float, float] = (0.04, 0.38, 0.68)
    lens_rgb: tuple[float, float, float] = (0.01, 0.015, 0.02)
    collision_enabled: bool = True


def _set_color(geom: Any, color: tuple[float, float, float]) -> None:
    from pxr import Gf, UsdGeom

    gprim = UsdGeom.Gprim(geom)
    gprim.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    gprim.CreateDisplayOpacityAttr([1.0])


def _camera_cube(stage: Any, path: str, size, position, color, collision_enabled: bool = True) -> None:
    from pxr import Gf, UsdGeom, UsdPhysics

    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    xform = UsdGeom.Xformable(cube)
    xform.AddTranslateOp().Set(Gf.Vec3d(*position))
    xform.AddScaleOp().Set(Gf.Vec3d(*size))
    _set_color(cube, color)
    collision = UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    collision.CreateCollisionEnabledAttr(bool(collision_enabled))


def _camera_cylinder(
    stage: Any,
    path: str,
    radius,
    height,
    position,
    color,
    orientation=None,
    collision_enabled: bool = True,
) -> None:
    from pxr import Gf, UsdGeom, UsdPhysics

    cylinder = UsdGeom.Cylinder.Define(stage, path)
    cylinder.CreateRadiusAttr(radius)
    cylinder.CreateHeightAttr(height)
    cylinder.CreateAxisAttr("Z")
    xform = UsdGeom.Xformable(cylinder)
    xform.AddTranslateOp().Set(Gf.Vec3d(*position))
    if orientation is not None:
        x, y, z, w = orientation
        xform.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    _set_color(cylinder, color)
    collision = UsdPhysics.CollisionAPI.Apply(cylinder.GetPrim())
    collision.CreateCollisionEnabledAttr(bool(collision_enabled))


@clone
def spawn_wrist_camera_assembly(
    prim_path: str,
    cfg: WristCameraAssemblyCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **_: Any,
):
    """Attach collision-enabled camera geometry above panda_hand/gripper."""

    stage = get_current_stage()
    create_prim(prim_path, "Xform", translation=translation, orientation=orientation, stage=stage)
    # D415-style dimensions follow ManiSkill's camera_link collision model
    # (20.05 x 99 x 23 mm). The rigid bracket bridges the hand to an offset
    # housing; every housing child inherits the same fixed inward pitch.
    _camera_cube(
        stage,
        f"{prim_path}/MountBracket",
        (0.110, 0.026, 0.030),
        (0.0525, 0.0, -0.040),
        cfg.accent_rgb,
        collision_enabled=cfg.collision_enabled,
    )
    housing_path = f"{prim_path}/CameraHousingFrame"
    create_prim(
        housing_path,
        "Xform",
        translation=WRIST_CAMERA_FRAME_POS_H,
        orientation=WRIST_CAMERA_ORIENTATION_H,
        stage=stage,
    )
    _camera_cube(
        stage,
        f"{housing_path}/CameraBody",
        (0.023, 0.099, 0.02005),
        (0.0, 0.0, 0.0),
        cfg.body_rgb,
        collision_enabled=cfg.collision_enabled,
    )
    _camera_cube(
        stage,
        f"{housing_path}/FrontBezel",
        (0.021, 0.097, 0.003),
        (0.0, 0.0, 0.0115),
        cfg.accent_rgb,
        collision_enabled=cfg.collision_enabled,
    )
    _camera_cylinder(
        stage,
        f"{housing_path}/Lens",
        0.007,
        0.003,
        (0.0, 0.0, 0.0140),
        cfg.lens_rgb,
        collision_enabled=cfg.collision_enabled,
    )
    _camera_cylinder(
        stage,
        f"{housing_path}/InfraredLens",
        0.006,
        0.003,
        (0.0, 0.0260, 0.0140),
        cfg.lens_rgb,
        collision_enabled=cfg.collision_enabled,
    )
    create_prim(
        f"{housing_path}/OpticalFrame",
        "Xform",
        translation=(0.0, 0.0, 0.0155),
        stage=stage,
    )
    return stage.GetPrimAtPath(prim_path)


def wrist_camera_frame_from_hand(hand_pose_w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return eye, look-at point, and image-up from the rigid mount."""

    eye_h = torch.tensor(WRIST_CAMERA_EYE_H, dtype=hand_pose_w.dtype, device=hand_pose_w.device).repeat(
        hand_pose_w.shape[0], 1
    )
    forward_h = torch.tensor(
        WRIST_CAMERA_FORWARD_H, dtype=hand_pose_w.dtype, device=hand_pose_w.device
    ).repeat(hand_pose_w.shape[0], 1)
    up_h = torch.tensor(WRIST_CAMERA_UP_H, dtype=hand_pose_w.dtype, device=hand_pose_w.device).repeat(
        hand_pose_w.shape[0], 1
    )
    eye_w = hand_pose_w[:, :3] + quat_apply(hand_pose_w[:, 3:7], eye_h)
    forward_w = quat_apply(hand_pose_w[:, 3:7], forward_h)
    up_w = quat_apply(hand_pose_w[:, 3:7], up_h)
    return eye_w, eye_w + 0.55 * forward_w, up_w


def wrist_camera_pose_from_hand(hand_pose_w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Backward-compatible eye/look-at pair from the physical extrinsic."""

    eye_w, target_w, _ = wrist_camera_frame_from_hand(hand_pose_w)
    return eye_w, target_w


class NewtonGlWristCameraSensor:
    """Lazy RGB sensor mounted to the physical camera optical frame."""

    def __init__(self, width: int = 384, height: int = 240):
        from isaaclab_newton.video_recording import NewtonGlPerspectiveVideoCfg

        from ..benchmark_rendering import BenchmarkNewtonGlPerspectiveVideo

        self.width = width
        self.height = height
        horizontal_fov_deg = math.degrees(
            2.0
            * math.atan(
                math.tan(math.radians(WRIST_CAMERA_VERTICAL_FOV_DEG) / 2.0)
                * width
                / height
            )
        )
        self._capture = BenchmarkNewtonGlPerspectiveVideo(
            NewtonGlPerspectiveVideoCfg(
                window_width=width,
                window_height=height,
                eye=(0.0, 0.0, 1.0),
                lookat=(0.0, 0.0, 0.0),
                horiz_fov_deg=horizontal_fov_deg,
            )
        )

    def render(self, eye_w: torch.Tensor, target_w: torch.Tensor, up_w: torch.Tensor) -> np.ndarray:
        eye = tuple(float(v) for v in eye_w[0].detach().cpu().tolist())
        target = tuple(float(v) for v in target_w[0].detach().cpu().tolist())
        up = tuple(float(v) for v in up_w[0].detach().cpu().tolist())
        self._capture.update_camera_frame(eye, target, up)
        return np.asarray(self._capture.render_rgb_array(), dtype=np.uint8)
