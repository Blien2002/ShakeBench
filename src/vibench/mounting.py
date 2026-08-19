"""C2 multi-support mapping for the common vibrating vehicle floor."""

from __future__ import annotations

import torch


def _mount_rotation(motion: torch.Tensor) -> torch.Tensor:
    """Return the exact C2 rotation matrix with the benchmark's axis signs.

    The historical C2 convention uses ``+x * ry`` in the vertical component,
    which is equivalent to a right-handed Y rotation by ``-ry``.  Keeping that
    convention makes the exact transform first-order compatible with existing
    scenarios while retaining all sine/cosine terms.
    """

    rx, ry, rz = motion[..., 3], -motion[..., 4], motion[..., 5]
    cx, sx = torch.cos(rx), torch.sin(rx)
    cy, sy = torch.cos(ry), torch.sin(ry)
    cz, sz = torch.cos(rz), torch.sin(rz)
    row0 = torch.stack((cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx), dim=-1)
    row1 = torch.stack((sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx), dim=-1)
    row2 = torch.stack((-sy, cy * sx, cy * cx), dim=-1)
    return torch.stack((row0, row1, row2), dim=-2)


def motion_at_mount(motion: torch.Tensor, mount_xy_m: tuple[float, float]) -> torch.Tensor:
    """Evaluate the complete SE(3) floor pose at one C2 installation point.

    The returned translation is ``t + R @ r - r`` and the Euler components
    retain the centre pose values.  ``r`` is expressed in the measurement
    frame and is intentionally independent of the compact task layout.
    """

    if motion.shape[-1] != 6:
        raise ValueError("motion must have six components")
    x_m, y_m = mount_xy_m
    out = motion.clone()
    mount = motion.new_tensor((x_m, y_m, 0.0))
    rotated_mount = torch.matmul(_mount_rotation(motion), mount)
    out[..., :3] = motion[..., :3] + rotated_mount - mount
    return out


def velocity_at_mount(
    pose: torch.Tensor,
    velocity: torch.Tensor,
    mount_xy_m: tuple[float, float],
) -> torch.Tensor:
    """Evaluate exact rigid-body velocity at an installed C2 measurement point."""

    if pose.shape[-1] != 6 or velocity.shape[-1] != 6:
        raise ValueError("pose and velocity must have six components")
    mount = pose.new_tensor((*mount_xy_m, 0.0))
    rotated_mount = torch.matmul(_mount_rotation(pose), mount)
    # Match the benchmark's historical Y-axis sign convention used above.
    omega = torch.stack((velocity[..., 3], -velocity[..., 4], velocity[..., 5]), dim=-1)
    out = velocity.clone()
    out[..., :3] = velocity[..., :3] + torch.linalg.cross(omega, rotated_mount, dim=-1)
    return out


def c2_support_motions(
    centre_motion: torch.Tensor,
    arm_mount_xy_m: tuple[float, float],
    table_mount_xy_m: tuple[float, float],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return arm-side and table-side motions from one floor motion."""

    return (
        motion_at_mount(centre_motion, arm_mount_xy_m),
        motion_at_mount(centre_motion, table_mount_xy_m),
    )


def c2_support_velocities(
    centre_pose: torch.Tensor,
    centre_velocity: torch.Tensor,
    arm_mount_xy_m: tuple[float, float],
    table_mount_xy_m: tuple[float, float],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return exact arm-side and table-side rigid-body velocities."""

    return (
        velocity_at_mount(centre_pose, centre_velocity, arm_mount_xy_m),
        velocity_at_mount(centre_pose, centre_velocity, table_mount_xy_m),
    )


def analytic_delta_z(
    centre_motion: torch.Tensor,
    arm_mount_xy_m: tuple[float, float],
    table_mount_xy_m: tuple[float, float],
) -> torch.Tensor:
    """Return exact arm-minus-table Z displacement for compatibility callers."""

    arm, table = c2_support_motions(centre_motion, arm_mount_xy_m, table_mount_xy_m)
    return arm[..., 2] - table[..., 2]
