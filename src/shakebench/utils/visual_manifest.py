"""Fast, renderer-free visual asset and parent-anchor regression registry."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path

import yaml

from ..models.arenas.room import load_room_arena_cfg, static_equipment_geometry_report
from ..config import BenchmarkConfig
from .paths import PROJECT_ROOT
from ..models.supports.shaker import ShakerGeometryCfg
from ..models.visual_assets import (
    ControlPanelAppearanceCfg,
    PlatformAppearanceCfg,
    WorktableAppearanceCfg,
    platform_shadow_layout,
)


VISUAL_MANIFEST_PATH = PROJECT_ROOT / "configs" / "visual_manifest.yaml"


@dataclass(frozen=True)
class AnchorAuditEntry:
    name: str
    prim_path: str
    parent_prim: str
    local_offset_m: tuple[float, float, float]
    world_position_m: tuple[float, float, float]
    expected_anchor_m: tuple[float, float, float]

    @property
    def error_m(self) -> float:
        return math.dist(self.world_position_m, self.expected_anchor_m)

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["error_m"] = self.error_m
        return payload


def load_visual_manifest(path: Path = VISUAL_MANIFEST_PATH) -> dict[str, object]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def visual_feature_facts() -> dict[str, float]:
    """Values tied to the deterministic scene-authoring implementation."""

    platen = PlatformAppearanceCfg()
    table = WorktableAppearanceCfg()
    shaker = ShakerGeometryCfg()
    room = load_room_arena_cfg()
    return {
        "platen_skirt_height_m": platen.skirt_height_m,
        "platen_bevel_m": platen.bevel_m,
        "platen_threaded_hole_surface_count": 1,
        "platen_nameplate_count": 1,
        "platen_warning_stripe_count": 13,
        "platen_warning_stripe_angle_deg": 35.0,
        "accelerometer_count": 4,
        "accelerometer_cable_count": 4,
        "worktable_top_thickness_m": table.top_size[2],
        "worktable_edge_count": 4,
        "worktable_stretcher_count": 4,
        "worktable_footplate_count": 4,
        "worktable_foot_bolt_count": 16,
        "control_panel_trim_count": 4,
        "control_panel_label_plate_count": 3,
        "control_panel_status_light_count": 3,
        "robot_mount_flange_count": 1,
        "robot_mount_bolt_count": 8,
        "stewart_segment_count": 12,
        "stewart_joint_count": 12,
        "stewart_base_flange_count": 6,
        "stewart_base_x_radius_m": shaker.base_joint_semi_major,
        "stewart_base_y_radius_m": shaker.base_joint_semi_minor,
        "stewart_platen_x_radius_m": shaker.platen_joint_semi_major,
        "stewart_platen_y_radius_m": shaker.platen_joint_semi_minor,
        "inertia_base_count": 1,
        "air_spring_count": 6,
        "pit_power_cable_segment_count": 12,
        "contact_shadow_target_count": float(len(platform_shadow_layout())),
        "guardrail_post_count": 3,
        "guardrail_baseplate_count": 3,
        "guardrail_anchor_bolt_count": 12,
        "guardrail_horizontal_rail_count": 2,
        "floor_safety_line_count": 4,
        "static_equipment_ground_error_max_m": max(
            abs(value)
            for name, value in static_equipment_geometry_report(room).items()
            if name.endswith("ground_error_m")
        ),
    }


def prim_anchor_audit(cfg: BenchmarkConfig | None = None) -> list[AnchorAuditEntry]:
    """Nominal world-space audit of every transform-sensitive attachment."""

    benchmark = cfg or BenchmarkConfig()
    platen = PlatformAppearanceCfg(
        size_xy=benchmark.platform_size[:2],
        top_z_m=0.5 * benchmark.platform_size[2] + 0.003,
        robot_xy=benchmark.robot_base[:2],
        target_xy=benchmark.target_center[:2],
    )
    world_z = benchmark.platform_center[2]
    surface_z = world_z + platen.top_z_m + 0.5 * platen.top_thickness_m + 0.0006
    entries: list[AnchorAuditEntry] = []

    for index, (x, y) in enumerate(
        (
            (-0.5 * platen.size_xy[0] + 0.09, -0.5 * platen.size_xy[1] + 0.09),
            (-0.5 * platen.size_xy[0] + 0.09, 0.5 * platen.size_xy[1] - 0.09),
            (0.5 * platen.size_xy[0] - 0.09, -0.5 * platen.size_xy[1] + 0.09),
            (0.5 * platen.size_xy[0] - 0.09, 0.5 * platen.size_xy[1] - 0.09),
        )
    ):
        position = (x, y, surface_z + 0.016)
        entries.append(
            AnchorAuditEntry(
                f"accelerometer_{index}",
                f"{{ENV_NS}}/VibrationFloor/LayeredAppearance/Accelerometer{index}",
                "{ENV_NS}/VibrationFloor/LayeredAppearance",
                (x, y, platen.top_z_m + 0.5 * platen.top_thickness_m + 0.0166),
                position,
                position,
            )
        )
        cable_anchor = (x, math.copysign(0.5 * platen.size_xy[1] - 0.015, y), surface_z + 0.010)
        entries.append(
            AnchorAuditEntry(
                f"accelerometer_cable_{index}",
                f"{{ENV_NS}}/VibrationFloor/LayeredAppearance/Accelerometer{index}/Cable",
                f"{{ENV_NS}}/VibrationFloor/LayeredAppearance/Accelerometer{index}",
                (0.0, cable_anchor[1] - y, -0.006),
                cable_anchor,
                cable_anchor,
            )
        )

    flange_center = (benchmark.robot_base[0], benchmark.robot_base[1], surface_z + 0.008)
    entries.append(
        AnchorAuditEntry(
            "robot_mount_flange",
            "{ENV_NS}/VibrationFloor/LayeredAppearance/RobotMountFlange",
            "{ENV_NS}/VibrationFloor/LayeredAppearance",
            (benchmark.robot_base[0], benchmark.robot_base[1], platen.top_z_m + 0.5 * platen.top_thickness_m + 0.0086),
            flange_center,
            flange_center,
        )
    )
    for index in range(8):
        angle = index * math.pi / 4.0
        local = (0.092 * math.cos(angle), 0.092 * math.sin(angle), 0.010)
        world = tuple(flange_center[axis] + local[axis] for axis in range(3))
        entries.append(
            AnchorAuditEntry(
                f"robot_mount_bolt_{index}",
                f"{{ENV_NS}}/VibrationFloor/LayeredAppearance/RobotMountFlange/Bolt{index}",
                "{ENV_NS}/VibrationFloor/LayeredAppearance/RobotMountFlange",
                local,
                world,
                world,
            )
        )

    side_y = 0.5 * platen.size_xy[1] + 0.0008
    for name, z in (("nameplate", 0.014), ("warning_band", -0.017)):
        world = (0.0, side_y, world_z + z)
        entries.append(
            AnchorAuditEntry(
                name,
                f"{{ENV_NS}}/VibrationFloor/LayeredAppearance/{'Nameplate' if name == 'nameplate' else 'WarningBandBase'}",
                "{ENV_NS}/VibrationFloor/LayeredAppearance",
                (0.0, side_y, z),
                world,
                world,
            )
        )

    for name, xy in platform_shadow_layout().items():
        world = (xy[0], xy[1], surface_z)
        entries.append(
            AnchorAuditEntry(
                f"shadow_{name}",
                f"{{ENV_NS}}/VibrationFloor/LayeredAppearance/{name}",
                "{ENV_NS}/VibrationFloor/LayeredAppearance",
                (xy[0], xy[1], surface_z - world_z),
                world,
                world,
            )
        )

    room = load_room_arena_cfg()
    pit_half_x = 0.5 * room.pit_size_m[0] + room.pit_border_width_m
    pit_half_y = 0.5 * room.pit_size_m[1] + room.pit_border_width_m
    for name, x, y in (("WestSouth", -pit_half_x, -pit_half_y), ("EastSouth", pit_half_x, -pit_half_y), ("SouthMid", 0.0, -pit_half_y)):
        world = (x, y, room.floor_z_m + 0.006)
        entries.append(
            AnchorAuditEntry(
                f"guardrail_base_{name}",
                f"/World/RoomArena/RailPost{name}/BasePlate",
                f"/World/RoomArena/RailPost{name}",
                (0.0, 0.0, room.floor_z_m + 0.006),
                world,
                world,
            )
        )
    return entries
