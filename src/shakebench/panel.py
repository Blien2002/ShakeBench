"""Control-panel task geometry shared by scene, task, controller, and tests.

The visible console is a five-sided prism fixed to the existing C2 worktable.
Its three controls are arranged on the sloped operator face: knob at the
upper-left, lever at the lower-center, and button at the upper-right.  All
coordinates returned here are in the same local task frame
used by :func:`shakebench.task.VibrationBenchmarkTask._support_state` (i.e. the
undisturbed env frame before the table-mount motion is applied).
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from .config import CONTROL_KINDS, BenchmarkConfig

CONTROL_INDEX = {kind: index for index, kind in enumerate(CONTROL_KINDS)}
CONTROL_KINDS_BY_INDEX = CONTROL_KINDS

PANEL_LAMP_COLORS = {
    "knob": (0.18, 0.86, 0.34),
    "lever": (0.96, 0.67, 0.08),
    "button": (0.82, 0.10, 0.06),
}
PANEL_LAMP_OFF_SCALE = 0.07
PANEL_LAMP_ACTIVE_SCALE = 0.35


def panel_lamp_linear_rgb(
    kind: str,
    progress: float,
    *,
    active: bool,
    completed: bool,
) -> tuple[float, float, float]:
    """Return the lamp's linear RGB for one control state."""

    if kind not in PANEL_LAMP_COLORS:
        raise ValueError(f"unknown panel control: {kind}")
    progress = min(1.0, max(0.0, float(progress)))
    if completed:
        scale = 1.0
    elif active:
        scale = PANEL_LAMP_ACTIVE_SCALE + (1.0 - PANEL_LAMP_ACTIVE_SCALE) * progress
    else:
        scale = PANEL_LAMP_OFF_SCALE
    return tuple(scale * channel for channel in PANEL_LAMP_COLORS[kind])


def linear_rgb_to_srgb(rgb: tuple[float, float, float]) -> tuple[float, float, float]:
    """Encode a linear RGB triplet for NewtonGL's sRGB shape-color buffer."""

    def encode(channel: float) -> float:
        channel = min(1.0, max(0.0, float(channel)))
        if channel <= 0.0031308:
            return 12.92 * channel
        return 1.055 * channel ** (1.0 / 2.4) - 0.055

    return tuple(encode(channel) for channel in rgb)


def panel_table_top_z_m(cfg: BenchmarkConfig) -> float:
    """Physical tabletop height for the configured worktable."""

    return cfg.resolved_worktable_center[2] + 0.5 * cfg.worktable_size[2]


@dataclass(frozen=True)
class ControlLayout:
    """Geometry of one control in the fixed task-local frame."""

    kind: str
    # Root/pivot center.  For the lever this is the bottom pivot, not the COM.
    pivot_xyz: tuple[float, float, float]
    size: tuple[float, float, float] | None
    radius_m: float | None = None
    length_m: float | None = None
    goal: float = 1.0


@dataclass(frozen=True)
class PanelLayout:
    board_center: tuple[float, float, float]
    board_size: tuple[float, float, float]
    face_x_m: float
    knob: ControlLayout
    lever: ControlLayout
    button: ControlLayout

    def control(self, kind: str) -> ControlLayout:
        if kind not in CONTROL_INDEX:
            raise ValueError(f"unknown panel control: {kind}")
        return getattr(self, kind)

    @property
    def knob_pivot(self) -> tuple[float, float, float]:
        return self.knob.pivot_xyz

    @property
    def lever_pivot(self) -> tuple[float, float, float]:
        return self.lever.pivot_xyz

    @property
    def button_pivot(self) -> tuple[float, float, float]:
        return self.button.pivot_xyz


def control_panel_layout(cfg: BenchmarkConfig) -> PanelLayout:
    """Resolve the deterministic panel layout for a benchmark config."""

    table_top_z = panel_table_top_z_m(cfg)
    board_x, board_y, board_z = cfg.panel.board_size
    board_cx, board_cy = cfg.panel.board_center_xy
    board_bottom_z = table_top_z + cfg.panel.board_base_clearance_m
    board_center_z = board_bottom_z + 0.5 * board_z
    face_x = board_cx - 0.5 * cfg.panel.console_depth_m

    # Five-sided X-Z outline: low front wall, bottom, tall rear wall, short
    # rear deck, and the sloped operator face.  UV-v follows that slope while
    # UV-u remains the console's lateral Y direction.
    half_depth = 0.5 * cfg.panel.console_depth_m
    half_height = 0.5 * cfg.panel.console_height_m
    front_x = -half_depth
    shoulder_x = half_depth - cfg.panel.console_rear_flat_depth_m
    front_top_z = -half_height + cfg.panel.console_front_height_m
    slope_dx = shoulder_x - front_x
    slope_dz = half_height - front_top_z
    slope_length = math.hypot(slope_dx, slope_dz)
    tangent_x = slope_dx / slope_length
    tangent_z = slope_dz / slope_length
    normal_x = -tangent_z
    normal_z = tangent_x
    surface_center_x = board_cx + 0.5 * (front_x + shoulder_x)
    surface_center_z = board_center_z + 0.5 * (front_top_z + half_height)

    def control_center(uv, standoff_m: float):
        uy, vz = uv
        return (
            surface_center_x + vz * tangent_x + standoff_m * normal_x,
            board_cy + uy,
            surface_center_z + vz * tangent_z + standoff_m * normal_z,
        )

    # Articulation roots sit on the operator surface.  Render/collision link
    # geometry supplies the outward projection; putting that projection into
    # the root as well would double the standoff and disconnect the bezels.
    knob_center = control_center(cfg.panel.knob_uv, 0.0)
    lever_pivot = control_center(cfg.panel.lever_uv, 0.0)
    button_center = control_center(cfg.panel.button_uv, 0.0)

    return PanelLayout(
        board_center=(board_cx, board_cy, board_center_z),
        board_size=cfg.panel.board_size,
        face_x_m=face_x,
        knob=ControlLayout(
            "knob",
            knob_center,
            size=None,
            radius_m=cfg.panel.knob_radius_m,
            length_m=cfg.panel.knob_length_m,
            goal=cfg.panel.knob_goal_rad,
        ),
        lever=ControlLayout(
            "lever",
            lever_pivot,
            size=(cfg.panel.lever_width_m, cfg.panel.lever_width_m, cfg.panel.lever_length_m),
            radius_m=None,
            length_m=cfg.panel.lever_length_m,
            goal=cfg.panel.lever_goal_rad,
        ),
        button=ControlLayout(
            "button",
            button_center,
            size=None,
            radius_m=cfg.panel.button_radius_m,
            length_m=cfg.panel.button_length_m,
            goal=cfg.panel.button_travel_m,
        ),
    )


def sequence_to_ids(sequence: tuple[str, ...]) -> tuple[int, ...]:
    return tuple(CONTROL_INDEX[kind] for kind in sequence)


def padded_sequence_ids(sequence: tuple[str, ...]) -> tuple[int, ...]:
    """Pad a control instruction to fixed length three with ``-1``."""

    ids = sequence_to_ids(sequence)
    return ids + (-1,) * (len(CONTROL_KINDS) - len(ids))


def control_speed_1_s(cfg: BenchmarkConfig, kind: str) -> float:
    if kind == "knob":
        return cfg.panel.knob_operation_speed_1_s
    if kind == "lever":
        return cfg.panel.lever_operation_speed_1_s
    if kind == "button":
        return cfg.panel.button_operation_speed_1_s
    raise ValueError(f"unknown panel control: {kind}")
