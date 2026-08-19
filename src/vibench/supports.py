"""Single-coordinate support groups and the only support-pose write path.

The old C2 implementation allowed an object at visible position ``l`` to be
driven by a motion sampled at an unrelated virtual mount point ``r``.  This
module removes that possibility structurally: every driven asset belongs to
one :class:`SupportGroup`, and every member pose is generated from that
group's single ``(q, qd)`` transform.

Stage A implements the hard-mounted deck model: one ``deck`` group contains
the visible floor, Panda root, worktable, table legs and target bin.  The
optional isolated-table model will add a second ``table`` group later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch

from isaaclab.utils.math import quat_apply, quat_from_euler_xyz

from .config import BenchmarkConfig


@dataclass(frozen=True)
class SupportMember:
    """One driven asset or asset collection inside a support group."""

    name: str
    local: tuple[float, float, float]
    write_strategy: Literal["root", "collection"] = "root"


@dataclass(frozen=True)
class SupportGroup:
    """A rigid group sharing one motion source and one rotation anchor."""

    name: str
    motion_source: Literal["deck", "table"]
    rotation_anchor: tuple[float, float, float]
    members: tuple[SupportMember, ...]


def angular_velocity_from_euler_rates(rates: torch.Tensor) -> torch.Tensor:
    """Return exact world angular velocity for intrinsic-XYZ Euler rates.

    ``rates`` is a (..., 6) tensor whose last three components are Euler
    angle rates for the convention used by ``quat_from_euler_xyz``.
    """

    rx, ry, rz = rates[..., 3], rates[..., 4], rates[..., 5]
    return torch.stack(
        (
            rx + rz * torch.sin(ry),
            ry * torch.cos(rx) - rz * torch.sin(rx) * torch.cos(ry),
            ry * torch.sin(rx) + rz * torch.cos(rx) * torch.cos(ry),
        ),
        dim=-1,
    )


def support_pose_velocity(
    local: torch.Tensor,
    q: torch.Tensor,
    qd: torch.Tensor,
    rotation_anchor: tuple[float, float, float],
    env_origins: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (position, xyzw quaternion, 6-D velocity) for one group member."""

    quat = quat_from_euler_xyz(q[:, 3], q[:, 4], q[:, 5])
    anchor = local.new_tensor(rotation_anchor)
    offset = quat_apply(quat, local - anchor)
    position = env_origins + q[:, :3] + anchor + offset
    omega = angular_velocity_from_euler_rates(qd)
    linear = qd[:, :3] + torch.linalg.cross(omega, offset)
    velocity = torch.cat((linear, omega), dim=1)
    return position, quat, velocity


def table_leg_local_positions(cfg: BenchmarkConfig) -> tuple[tuple[float, float, float], ...]:
    """Return nominal leg-centre positions in the visible task frame."""

    leg_x = 0.5 * cfg.worktable_size[0] - 0.055
    leg_y = 0.5 * cfg.worktable_size[1] - 0.055
    floor_top = cfg.platform_center[2] + 0.5 * cfg.platform_size[2]
    table_bottom = cfg.worktable_center[2] - 0.5 * cfg.worktable_size[2]
    leg_z = 0.5 * (table_bottom + floor_top) + cfg.assembly_clearance_m
    positions = []
    for x_sign, y_sign in ((-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0)):
        positions.append(
            (
                cfg.worktable_center[0] + x_sign * leg_x,
                cfg.worktable_center[1] + y_sign * leg_y,
                leg_z,
            )
        )
    return tuple(positions)


def support_group_geometries(cfg: BenchmarkConfig) -> tuple[SupportGroup, ...]:
    """Build the support-group layout from visible configuration only.

    This function is intentionally asset-free so the offline travel replay
    and the runtime writer consume exactly the same member table.
    """

    deck_members = [
        SupportMember("platform", cfg.platform_center),
        SupportMember("robot", cfg.resolved_robot_base),
        SupportMember("worktable", cfg.resolved_worktable_center),
    ]
    if cfg.task != "panel_operation":
        deck_members.append(SupportMember("target", cfg.resolved_target_center))
    deck_members.extend(
        SupportMember(f"table_leg_{index}", position)
        for index, position in enumerate(table_leg_local_positions(cfg))
    )
    deck = SupportGroup(
        name="deck",
        motion_source="deck",
        rotation_anchor=cfg.platform_center,
        members=tuple(deck_members),
    )

    # Stage B will insert a ``table`` group here when the isolated-table
    # model is enabled.  The writer below already supports it.
    return (deck,)


_SUPPORT_ASSET_KEYS = {
    "platform": "platform",
    "robot": "robot",
    "worktable": "worktable",
    "target": "target",
    "table_leg_0": "table_leg_fl",
    "table_leg_1": "table_leg_fr",
    "table_leg_2": "table_leg_rl",
    "table_leg_3": "table_leg_rr",
}


def write_support_groups(
    *,
    groups: tuple[SupportGroup, ...],
    scene: Any,
    q_deck: torch.Tensor,
    qd_deck: torch.Tensor,
    q_table: torch.Tensor | None = None,
    qd_table: torch.Tensor | None = None,
) -> None:
    """Write every support-group member from its group's single transform."""

    for group in groups:
        if group.motion_source == "deck":
            q, qd = q_deck, qd_deck
        else:
            if q_table is None or qd_table is None:
                raise RuntimeError("table motion is required for a table support group")
            q, qd = q_table, qd_table
        for member in group.members:
            asset = scene[_SUPPORT_ASSET_KEYS[member.name]]
            local = q.new_tensor(member.local).repeat(q.shape[0], 1)
            position, quat, velocity = support_pose_velocity(
                local,
                q,
                qd,
                group.rotation_anchor,
                scene.env_origins,
            )
            pose = torch.cat((position, quat), dim=1)
            if member.write_strategy == "collection":
                raise NotImplementedError("collection members are wired by the task wrapper")
            asset.write_root_pose_to_sim_index(root_pose=pose)
            asset.write_root_velocity_to_sim_index(root_velocity=velocity)


HARD_STRUCTURAL_EXCLUSIONS: tuple[tuple[str, str], ...] = (
    ("/panda_link0/collisions", "/VibrationFloor/geometry"),
    ("/WorkTableLeg", "/VibrationFloor/geometry"),
    ("/WorkTableLeg", "/WorkTableTop/geometry"),
)


def install_structural_collision_exclusions(
    exclusions: tuple[tuple[str, str], ...] = HARD_STRUCTURAL_EXCLUSIONS,
) -> None:
    """Register a MODEL_INIT callback that filters structural mounting pairs.

    Two bodies inside the same rigid support group are mechanically joined,
    so their contact pair carries no information.  Newton's ModelBuilder
    exposes per-shape-pair filtering; we install it at model build time and
    then assert the resulting topology in diagnostics.
    """

    from isaaclab_newton.physics import NewtonManager
    from isaaclab.physics import PhysicsEvent

    def _on_model_init(_payload) -> None:
        builder = NewtonManager._builder
        if builder is None:
            raise RuntimeError("Newton model builder was not available at MODEL_INIT")
        labels = [str(label) for label in builder.shape_label]
        for lhs_pattern, rhs_pattern in exclusions:
            lhs = [index for index, label in enumerate(labels) if lhs_pattern in label]
            rhs = [index for index, label in enumerate(labels) if rhs_pattern in label]
            if not lhs or not rhs:
                raise RuntimeError(
                    f"structural exclusion matched no shapes: {lhs_pattern!r} x {rhs_pattern!r}"
                )
            for shape_a in lhs:
                for shape_b in rhs:
                    if shape_a != shape_b:
                        builder.add_shape_collision_filter_pair(shape_a, shape_b)

    NewtonManager.register_callback(
        _on_model_init,
        PhysicsEvent.MODEL_INIT,
        order=-55,
        name="vibench_structural_collision_exclusions",
    )
