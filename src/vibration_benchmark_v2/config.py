"""Numerical and asset configuration for benchmark-v2."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .shaker import ShakerGeometryCfg


AXES = ("tx", "ty", "tz", "rx", "ry", "rz")


@dataclass(frozen=True)
class SpectralBand:
    """A narrow PSD band represented by random-phase spectral lines."""

    center_hz: float
    rms: float
    bandwidth_ratio: float = 0.08
    tones: int = 12

    def __post_init__(self) -> None:
        if self.center_hz <= 0.0 or self.rms < 0.0:
            raise ValueError("center_hz must be positive and rms must be non-negative")
        if not 0.0 <= self.bandwidth_ratio < 1.0 or self.tones < 1:
            raise ValueError("invalid bandwidth_ratio or tones")


def _default_bands() -> dict[str, tuple[SpectralBand, ...]]:
    # Same legacy excitation family used by the prior MuJoCo/RM75 benchmark.
    return {
        "tx": (SpectralBand(18.0, 0.00050, 0.08, 12),),
        "ty": (SpectralBand(13.0, 0.00025, 0.10, 10),),
        "tz": (SpectralBand(32.0, 0.00150, 0.06, 12),),
        "rx": (SpectralBand(8.0, 0.00400, 0.10, 12),),
        "ry": (SpectralBand(11.0, 0.00200, 0.10, 10),),
        "rz": (SpectralBand(6.0, 0.00120, 0.12, 8),),
    }


@dataclass(frozen=True)
class VibrationConfig:
    """Six-axis sine or seeded spectral excitation.

    At fixed acceleration RMS, peak velocity scales as ``v = a / omega``.
    Lowering frequency while increasing displacement therefore worsens
    contact penetration; official scenarios keep a per-substep displacement
    safety check enabled.
    """

    mode: Literal["off", "sine", "spectral"] = "spectral"
    seed: int = 17
    ramp_s: float = 0.75
    sine_axis: Literal["tx", "ty", "tz", "rx", "ry", "rz"] = "tz"
    sine_amplitude: float = 0.0015
    sine_frequency_hz: float = 5.0
    spectral_scale: float = 1.0
    active_axes: tuple[str, ...] = AXES
    bands: dict[str, tuple[SpectralBand, ...]] = field(default_factory=_default_bands)

    def __post_init__(self) -> None:
        if self.mode not in ("off", "sine", "spectral"):
            raise ValueError("mode must be off, sine, or spectral")
        if self.ramp_s <= 0.0 or self.sine_frequency_hz <= 0.0:
            raise ValueError("ramp and sine frequency must be positive")
        if self.spectral_scale <= 0.0:
            raise ValueError("spectral_scale must be positive")
        if not self.active_axes:
            raise ValueError("active_axes must contain at least one vibration axis")
        invalid_active = set(self.active_axes) - set(AXES)
        if invalid_active:
            raise ValueError(f"unknown active vibration axes: {sorted(invalid_active)}")
        if len(set(self.active_axes)) != len(self.active_axes):
            raise ValueError("active_axes must not contain duplicates")
        unknown = set(self.bands) - set(AXES)
        if unknown:
            raise ValueError(f"unknown vibration axes: {sorted(unknown)}")


YCB_ASSETS = {
    "cracker_box": "003_cracker_box.usd",
    "sugar_box": "004_sugar_box.usd",
    "soup_can": "005_tomato_soup_can.usd",
    "mustard_bottle": "006_mustard_bottle.usd",
}

YCB_DIMENSIONS_M: dict[str, tuple[float, float, float]] = {
    # Nominal unscaled collision bounds.  Runtime initialization reads the
    # converted Newton mesh as the source of truth; these values keep static
    # feasibility checks useful before a simulation exists.
    "cracker_box": (0.159, 0.213, 0.071),
    "sugar_box": (0.092677, 0.176251, 0.044803),
    "soup_can": (0.067, 0.067, 0.101),
    "mustard_bottle": (0.095, 0.058, 0.190),
}


def workpiece_dimensions_m(name: str, scale: float) -> tuple[float, float, float]:
    """Return benchmark X/Y/Z extents after uniform scaling."""

    if name not in YCB_DIMENSIONS_M:
        raise ValueError(f"unsupported workpiece: {name}")
    return tuple(float(scale) * value for value in YCB_DIMENSIONS_M[name])


@dataclass(frozen=True)
class AssetConfig:
    """Versioned selection of standard Isaac Sim/YCB assets."""

    robot: Literal["franka_panda"] = "franka_panda"
    table: Literal["phenolic_worktable_c2"] = "phenolic_worktable_c2"
    workpiece: Literal["cracker_box", "sugar_box", "soup_can", "mustard_bottle"] = "sugar_box"
    workpiece_scale: float = 0.75

    def __post_init__(self) -> None:
        if self.workpiece not in YCB_ASSETS:
            raise ValueError(f"unsupported workpiece: {self.workpiece}")
        if not 0.2 <= self.workpiece_scale <= 1.5:
            raise ValueError("workpiece_scale must be in [0.2, 1.5]")


@dataclass(frozen=True)
class BenchmarkConfig:
    """Complete standalone benchmark configuration."""

    # Official evaluation/recording profile.  The former 240 Hz profile is
    # retained by the CLI for training throughput, but is not scoreable.
    dt: float = 1.0 / 1000.0
    solver_substeps: int = 4
    episode_s: float = 16.0
    num_envs: int = 1
    assets: AssetConfig = field(default_factory=AssetConfig)
    vibration: VibrationConfig = field(default_factory=VibrationConfig)
    shaker: ShakerGeometryCfg = field(default_factory=ShakerGeometryCfg)
    material_mu: float = 1.5
    support_config: Literal["C2"] = "C2"
    # One visible vehicle/shaker floor; arm and table are separate C2 mount
    # points on it.  The vehicle measurement coordinates are intentionally
    # separate from the compact task-layout coordinates below.
    arm_mount_xy_m: tuple[float, float] = (0.75, -0.45)
    table_mount_xy_m: tuple[float, float] = (-0.75, 0.45)
    platform_size: tuple[float, float, float] = (1.60, 1.10, 0.08)
    platform_center: tuple[float, float, float] = (0.0, 0.0, 0.04)
    robot_base: tuple[float, float, float] = (-0.47, 0.0, 0.08)
    robot_mount_dynamic_clearance_m: float = 0.007
    worktable_size: tuple[float, float, float] = (0.65, 0.60, 0.06)
    worktable_center: tuple[float, float, float] = (0.18, 0.0, 0.34)
    table_mount_dynamic_clearance_m: float = 0.007
    workpiece_start: tuple[float, float, float] = (0.08, -0.13, 0.47)
    workpiece_initial_clearance_m: float = 0.001
    target_center: tuple[float, float, float] = (0.08, 0.17, 0.376)
    contact_margin_m: float = 0.001
    contact_solref: tuple[float, float] = (0.00060, 1.0)
    # Official 1000 Hz spectral profile is 0.264 mm at the conservative
    # 3.5-sigma velocity estimate; 0.3 mm admits it while rejecting the
    # 1.10 mm training profile that produced multi-millimetre tunnelling.
    max_substep_displacement_m: float = 0.0003
    descend_contact_threshold_n: float = 0.05
    descend_timeout_s: float = 2.0
    grasp_z_guard_margin_m: float = 0.0
    approach_clearance_m: float = 0.080
    descend_clearance_m: float = 0.004
    finger_table_clearance_m: float = 0.012
    descend_position_tolerance_m: float = 0.001
    arm_linear_speed_m_s: float = 0.15
    lift_takeoff_speed_m_s: float = 0.020
    lift_takeoff_duration_s: float = 0.75
    descend_linear_speed_m_s: float = 0.060
    place_linear_speed_m_s: float = 0.080
    gripper_closing_speed_m_s: float = 0.003
    gripper_contact_recovery_speed_m_s: float = 0.001
    gripper_opening_speed_m_s: float = 0.040
    gripper_contact_preload_m: float = 0.0003
    grasp_timeout_s: float = 2.5
    grasp_contact_loss_timeout_s: float = 0.20
    grasp_slip_tolerance_m: float = 0.008
    grasp_assist: bool = False

    def __post_init__(self) -> None:
        if self.dt <= 0.0 or self.episode_s <= 0.0 or self.num_envs < 1:
            raise ValueError("invalid timestep, episode length, or environment count")
        if self.solver_substeps < 1:
            raise ValueError("solver_substeps must be positive")
        if self.contact_margin_m < 0.0 or self.max_substep_displacement_m <= 0.0:
            raise ValueError("contact margin and displacement limit must be non-negative/positive")
        if self.contact_solref[0] <= 0.0 or self.contact_solref[1] <= 0.0:
            raise ValueError("contact solref time constant and damping ratio must be positive")
        if self.contact_solref[0] < 2.0 * self.dt / self.solver_substeps:
            raise ValueError("contact solref time constant must be >= 2 solver substeps")
        if self.descend_contact_threshold_n <= 0.0 or self.descend_timeout_s <= 0.0:
            raise ValueError("descend contact threshold and timeout must be positive")
        if self.approach_clearance_m <= self.descend_clearance_m or self.descend_clearance_m < 0.0:
            raise ValueError("approach clearance must exceed the non-negative descend clearance")
        if self.finger_table_clearance_m < 0.0 or self.descend_position_tolerance_m <= 0.0:
            raise ValueError("finger table clearance must be non-negative and descend tolerance positive")
        if min(
            self.arm_linear_speed_m_s,
            self.lift_takeoff_speed_m_s,
            self.descend_linear_speed_m_s,
            self.place_linear_speed_m_s,
            self.gripper_closing_speed_m_s,
            self.gripper_contact_recovery_speed_m_s,
            self.gripper_opening_speed_m_s,
        ) <= 0.0:
            raise ValueError("controller speed limits must be positive")
        if self.lift_takeoff_duration_s <= 0.0:
            raise ValueError("lift takeoff duration must be positive")
        if not 0.0 <= self.gripper_contact_preload_m <= 0.002:
            raise ValueError("gripper contact preload must be in [0, 2.0] mm")
        if self.grasp_timeout_s <= 0.0 or self.grasp_contact_loss_timeout_s <= 0.0:
            raise ValueError("grasp timeout and contact-loss timeout must be positive")
        if self.grasp_slip_tolerance_m <= 0.0:
            raise ValueError("grasp slip tolerance must be positive")
        if self.workpiece_initial_clearance_m < 0.0:
            raise ValueError("workpiece_initial_clearance_m must be non-negative")
        if self.robot_mount_dynamic_clearance_m < 0.0 or self.table_mount_dynamic_clearance_m < 0.0:
            raise ValueError("mount dynamic clearances must be non-negative")
        if not 0.05 <= self.material_mu <= 2.0:
            raise ValueError("material_mu must be in [0.05, 2.0]")
        if self.support_config != "C2":
            raise ValueError("benchmark-v2 currently requires the requested C2 support layout")

    @property
    def physics_hz(self) -> int:
        """Outer physics frequency rounded to the nearest integer hertz."""

        return int(round(1.0 / self.dt))

    @property
    def effective_substep_hz(self) -> int:
        return self.physics_hz * self.solver_substeps

    @property
    def resolved_workpiece_start(self) -> tuple[float, float, float]:
        """Place the selected YCB collider just above the physical tabletop."""

        height = workpiece_dimensions_m(self.assets.workpiece, self.assets.workpiece_scale)[2]
        table_top = self.resolved_worktable_center[2] + 0.5 * self.worktable_size[2]
        return (
            self.workpiece_start[0],
            self.workpiece_start[1],
            table_top + 0.5 * height + self.workpiece_initial_clearance_m,
        )

    @property
    def resolved_robot_base(self) -> tuple[float, float, float]:
        """Physical root above the flange, including C2 differential travel."""

        return (
            self.robot_base[0],
            self.robot_base[1],
            self.robot_base[2] + self.robot_mount_dynamic_clearance_m,
        )

    @property
    def resolved_worktable_center(self) -> tuple[float, float, float]:
        return (
            self.worktable_center[0],
            self.worktable_center[1],
            self.worktable_center[2] + self.table_mount_dynamic_clearance_m,
        )

    @property
    def resolved_target_center(self) -> tuple[float, float, float]:
        return (
            self.target_center[0],
            self.target_center[1],
            self.target_center[2] + self.table_mount_dynamic_clearance_m,
        )
