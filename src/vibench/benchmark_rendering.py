"""Neutral manipulation-benchmark lighting for Newton ViewerGL captures."""

from __future__ import annotations

import types

import numpy as np

from isaaclab_newton.video_recording import NewtonGlPerspectiveVideo


class BenchmarkNewtonGlPerspectiveVideo(NewtonGlPerspectiveVideo):
    """ViewerGL capture with neutral lighting inspired by ManiSkill/robosuite.

    ManiSkill's default manipulation environment uses neutral 0.3 ambient
    illumination plus an oblique and a top-down white directional light.
    robosuite's TableArena likewise uses a neutral directional key and subdued
    specular materials. ViewerGL exposes one sun plus a camera-aligned fill, so
    those concepts map to a white oblique sun, neutral hemispherical ambient,
    and a low-specular white fill.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self._benchmark_lighting_applied = False

    def _ensure_viewer(self) -> None:
        super()._ensure_viewer()
        if self._benchmark_lighting_applied:
            return
        renderer = self._viewer.renderer

        # Remove ViewerGL's photographic environment map so surface color is
        # determined by the authored albedo rather than colored reflections.
        renderer._env_path = None
        renderer._env_texture = None
        renderer._env_intensity = 0.0

        # ManiSkill-style neutral ambient and two-white-light approximation.
        renderer.ambient_sky = (0.36, 0.36, 0.36)
        renderer.ambient_ground = (0.14, 0.14, 0.14)
        renderer._light_color = (1.0, 1.0, 1.0)
        sun = np.asarray((0.85, 0.65, 1.8), dtype=np.float32)
        renderer._sun_direction = sun / np.linalg.norm(sun)
        renderer.spotlight_enabled = True
        # Both robosuite TableArena and ManiSkill's default manipulation setup
        # disable cast shadows. This avoids large viewpoint-dependent shadow
        # masks in RGB observations while the white key/fill still conveys form.
        renderer.draw_shadows = False
        # ViewerGL still passes this uniform to the shape shader even when
        # shadows are disabled, but only initializes it in the shadow pass.
        renderer._light_space_matrix = np.eye(4, dtype=np.float32)

        renderer._diffuse_scale = 1.0
        renderer._specular_scale = 0.30
        renderer._exposure = 0.88
        renderer.sky_upper = (0.82, 0.84, 0.87)
        renderer.sky_lower = (0.54, 0.52, 0.49)
        self._benchmark_lighting_applied = True

    def update_camera_frame(self, position, target, up) -> None:
        """Apply a complete rigid camera frame, including optical roll.

        ViewerGL's public ``set_camera`` API only stores pitch/yaw and derives
        image-up from world Z.  That is suitable for an orbit camera but loses
        one degree of freedom for an eye-in-hand sensor.  Bind a per-viewer up
        basis so the RGB frame follows the physical mount's full orientation.
        """

        self._ensure_viewer()
        self._apply_camera(position, target)
        forward = np.asarray(target, dtype=np.float64) - np.asarray(position, dtype=np.float64)
        forward /= np.linalg.norm(forward)
        up_orthogonal = np.asarray(up, dtype=np.float64)
        up_orthogonal -= np.dot(up_orthogonal, forward) * forward
        up_orthogonal /= np.linalg.norm(up_orthogonal)
        camera = self._viewer.camera
        camera._rigid_mount_up = tuple(float(v) for v in up_orthogonal)
        camera.get_up = types.MethodType(_rigid_mount_get_up, camera)
        camera.get_right = types.MethodType(_rigid_mount_get_right, camera)


def _rigid_mount_get_up(camera):
    """ViewerGL camera method bound only to rigid eye-in-hand captures."""

    from pyglet.math import Vec3

    front = camera.get_front()
    hint = Vec3(*camera._rigid_mount_up)
    right = Vec3.cross(front, hint).normalize()
    return Vec3.cross(right, front).normalize()


def _rigid_mount_get_right(camera):
    """ViewerGL camera method paired with :func:`_rigid_mount_get_up`."""

    from pyglet.math import Vec3

    return Vec3.cross(camera.get_front(), camera.get_up()).normalize()
