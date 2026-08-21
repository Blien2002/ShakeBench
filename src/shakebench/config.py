"""Numerical and asset configuration for benchmark-v2."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from typing import Any, Literal


AXES = ("tx", "ty", "tz", "rx", "ry", "rz")
CONTROL_KINDS = ("knob", "lever", "button")

# Standard gravity used by the Gamma calibration.  Gamma is the realized peak
# vertical acceleration at the workpiece divided by this exact value.
G = 9.80665

# Hard ceiling of the displacement-limited feasibility wedge.  At and above a
# tz centre frequency of 8 Hz the deck displacement gate no longer binds and
# Gamma is capped at this value.
GAMMA_CEIL = 4.0


def sample_panel_sequence(seed: int, controls: tuple[str, ...] = CONTROL_KINDS) -> tuple[str, ...]:
    """Deterministically sample a random non-empty ordered subset of controls.

    Each rollout receives a fresh sampled instruction: one, two, or all three
    controls may have to be operated, and their required order is shuffled.
    """

    if seed < 0 or len(set(controls)) != len(controls) or not controls:
        raise ValueError("panel sequence seed must be non-negative and controls must be unique/non-empty")
    rng = random.Random(seed)
    count = rng.randint(1, len(controls))
    return tuple(rng.sample(list(controls), count))


@dataclass(frozen=True)
class SpectralBand:
    """One narrow acceleration-PSD band of random-phase spectral lines.

    ``accel_rms`` is a *relative* acceleration RMS (m/s^2 for tx/ty/tz,
    rad/s^2 for rx/ry/rz).  The absolute level is fixed at runtime by the
    Gamma calibration; only the ratios between bands are authored here.
    """

    center_hz: float
    accel_rms: float
    bandwidth_ratio: float = 0.10
    tones: int = 12

    def __post_init__(self) -> None:
        if self.center_hz <= 0.0 or self.accel_rms < 0.0:
            raise ValueError("center_hz must be positive and accel_rms must be non-negative")
        if not 0.0 <= self.bandwidth_ratio < 1.0 or self.tones < 1:
            raise ValueError("invalid bandwidth_ratio or tones")


# Relative spectral shape of the contact-preserving excitation family.  Only
# ratios are meaningful here; the absolute level is supplied at runtime by the
# Gamma calibration.  Translation units are m/s^2 and rotation units rad/s^2.
_TRANSLATION_BAND_SHAPES: dict[str, tuple[float, float, float, int]] = {
    "tz": (8.0, 1.000, 0.10, 12),
    "tx": (5.0, 0.500, 0.10, 12),
    "ty": (6.5, 0.350, 0.10, 10),
}

# Rotation bands are never hand-picked angular amplitudes.  Each rotational
# band induces, at ``reference_lever_m``, a tangential acceleration equal to
# ``kappa_rot`` times the translational band it is linked to:
#     sigma_alpha = kappa_rot * sigma_a_trans / reference_lever_m.
# rx/ry link to tz (vertical differential) and rz links to tx (horizontal
# differential).  ``linked`` stores the translational axis name.
_ROTATION_BAND_SHAPES: dict[str, tuple[float, str, float, int]] = {
    "rx": (3.0, "tz", 0.12, 12),
    "ry": (4.0, "tz", 0.12, 10),
    "rz": (2.5, "tx", 0.12, 8),
}


def _default_bands(
    *,
    kappa_rot: float = 0.30,
    reference_lever_m: float = 0.65,
) -> dict[str, tuple[SpectralBand, ...]]:
    """Return the default relative acceleration-PSD band table.

    Translational entries are authored explicitly; rotational entries are
    derived from the kappa_rot differential/common-mode rule so that
    ``kappa_rot`` remains a physically meaningful, independently scannable
    knob instead of six hand-picked milliradian numbers.
    """

    translational = {
        axis: (SpectralBand(center_hz, accel_rms, bandwidth_ratio, tones),)
        for axis, (center_hz, accel_rms, bandwidth_ratio, tones) in _TRANSLATION_BAND_SHAPES.items()
    }
    rotational = {}
    for axis, (center_hz, linked, bandwidth_ratio, tones) in _ROTATION_BAND_SHAPES.items():
        linked_rms = translational[linked][0].accel_rms
        rotational[axis] = (
            SpectralBand(
                center_hz,
                kappa_rot * linked_rms / reference_lever_m,
                bandwidth_ratio,
                tones,
            ),
        )
    return {**translational, **rotational}


@dataclass(frozen=True)
class VibrationConfig:
    """Gamma-calibrated six-axis excitation.

    Bands store only the relative spectral *shape* (acceleration PSD
    normalized to unit level).  ``calibrate_level_scale`` solves the single
    absolute ``level_scale`` that makes the realized peak vertical workpiece
    acceleration equal ``gamma * G``, so acceleration is the first-class
    quantity and displacement is derived per tone.

    ``mode="sine"`` is a deprecated alias for ``sweep`` and is kept for one
    release cycle.
    """

    mode: Literal["off", "sweep", "sine", "spectral"] = "spectral"
    seed: int = 17
    ramp_s: float = 0.75
    # Absolute difficulty: realized peak vertical acceleration at the
    # workpiece divided by G.  gamma >= 1 enters the ballistic (periodic
    # contact-loss) regime and requires ballistic_allowed=True.
    gamma: float = 0.50
    # Global centre-frequency multiplier for the authored band shapes.
    frequency_scale: float = 1.0
    # Reference arm--table spacing used by the rotational band rule.
    reference_lever_m: float = 0.65
    # Workpiece position relative to the deck centre; the Gamma calibration
    # evaluates a_z = qdd_z + (alpha x r_wp)_z at this point.
    workpiece_offset_m: tuple[float, float, float] = (0.65, 0.0, 0.0)
    # Displacement-limited feasibility gate: deck centre peak displacement
    # must stay below this value, otherwise startup is refused.
    max_deck_displacement_m: float = 0.025
    # Explicit opt-in for the ballistic (Gamma >= 1) regime.
    ballistic_allowed: bool = False
    # Differential/common-mode ratio for rotation bands; only used when the
    # default bands are requested.
    kappa_rot: float = 0.30
    # Runtime-computed absolute level in m/s^2 (rad/s^2 for rotational
    # channels).  The CLI calibrates this before building the simulation.
    level_scale: float = 1.0
    # Scalar offset into the same deterministic infinite realization. It is
    # shared by every axis and tone; independent phase offsets would change
    # the waveform rather than select another time window.
    t0: float = 0.0
    sine_axis: Literal["tx", "ty", "tz", "rx", "ry", "rz"] = "tz"
    sine_frequency_hz: float = 5.0
    active_axes: tuple[str, ...] = AXES
    bands: dict[str, tuple[SpectralBand, ...]] | None = None

    def __post_init__(self) -> None:
        if self.mode not in ("off", "sweep", "sine", "spectral"):
            raise ValueError("mode must be off, sweep, sine (deprecated alias), or spectral")
        if self.ramp_s <= 0.0 or self.sine_frequency_hz <= 0.0:
            raise ValueError("ramp and sine frequency must be positive")
        if self.sine_axis not in AXES:
            raise ValueError(f"unknown sweep axis: {self.sine_axis}")
        if self.gamma <= 0.0:
            raise ValueError("gamma must be positive")
        if self.gamma >= 1.0 and not self.ballistic_allowed:
            raise ValueError(
                f"gamma={self.gamma} puts the workpiece in the ballistic regime "
                "(periodic loss of table contact). Set ballistic_allowed=True to opt in; "
                "such runs are not eligible for the official contact-preserving score."
            )
        if self.frequency_scale <= 0.0:
            raise ValueError("frequency_scale must be positive")
        if self.reference_lever_m <= 0.0:
            raise ValueError("reference_lever_m must be positive")
        if len(self.workpiece_offset_m) != 3:
            raise ValueError("workpiece_offset_m must contain exactly three components")
        if self.max_deck_displacement_m <= 0.0:
            raise ValueError("max_deck_displacement_m must be positive")
        if self.kappa_rot < 0.0:
            raise ValueError("kappa_rot must be non-negative")
        if self.level_scale < 0.0:
            raise ValueError("level_scale must be non-negative")
        if not math.isfinite(self.t0) or self.t0 < 0.0:
            raise ValueError("t0 must be a finite non-negative time offset")
        if not self.active_axes:
            raise ValueError("active_axes must contain at least one vibration axis")
        invalid_active = set(self.active_axes) - set(AXES)
        if invalid_active:
            raise ValueError(f"unknown active vibration axes: {sorted(invalid_active)}")
        if len(set(self.active_axes)) != len(self.active_axes):
            raise ValueError("active_axes must not contain duplicates")
        if self.bands is None:
            object.__setattr__(
                self,
                "bands",
                _default_bands(kappa_rot=self.kappa_rot, reference_lever_m=self.reference_lever_m),
            )
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

# Nominal YCB object masses. Uniform asset scaling preserves density, so the
# runtime mass used by diagnostics scales cubically with the authored mesh.
YCB_MASSES_KG: dict[str, float] = {
    "cracker_box": 0.411,
    "sugar_box": 0.514,
    "soup_can": 0.349,
    "mustard_bottle": 0.603,
}


def workpiece_dimensions_m(name: str, scale: float) -> tuple[float, float, float]:
    """Return benchmark X/Y/Z extents after uniform scaling."""

    if name not in YCB_DIMENSIONS_M:
        raise ValueError(f"unsupported workpiece: {name}")
    return tuple(float(scale) * value for value in YCB_DIMENSIONS_M[name])


def workpiece_mass_kg(name: str, scale: float) -> float:
    """Return the density-preserving mass of a uniformly scaled YCB asset."""

    if name not in YCB_MASSES_KG:
        raise ValueError(f"unsupported workpiece: {name}")
    return float(YCB_MASSES_KG[name]) * float(scale) ** 3


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
class PanelConfig:
    """The fixed sloped control console mounted on the C2 worktable.

    The panel hosts exactly three controls in a triangle on its operator face:
    knob (upper-left), lever (lower-center), and button (upper-right).
    ``sequence`` is the rollout instruction; when empty, a random ordered
    subset is sampled deterministically from ``seed``.
    """

    seed: int = 17
    sequence: tuple[str, ...] = ()
    board_size: tuple[float, float, float] = (0.050, 0.300, 0.180)
    board_center_xy: tuple[float, float] = (0.100, 0.0)
    board_base_clearance_m: float = 0.002
    console_depth_m: float = 0.190
    console_width_m: float = 0.320
    console_height_m: float = 0.180
    console_front_height_m: float = 0.055
    console_rear_flat_depth_m: float = 0.025
    knob_uv: tuple[float, float] = (-0.085, 0.055)
    # Matches the 25.737 mm radial envelope of the Apollo runtime LOD with a
    # small collision guard, preventing fingers from entering the visual mesh.
    knob_radius_m: float = 0.026
    knob_length_m: float = 0.036
    knob_center_standoff_m: float = 0.020
    knob_goal_rad: float = math.radians(72.0)
    lever_uv: tuple[float, float] = (0.0, -0.055)
    lever_width_m: float = 0.018
    lever_length_m: float = 0.085
    lever_pivot_standoff_m: float = 0.010
    lever_goal_rad: float = math.radians(30.0)
    button_uv: tuple[float, float] = (0.085, 0.055)
    button_radius_m: float = 0.018
    button_length_m: float = 0.020
    button_center_standoff_m: float = 0.014
    button_travel_m: float = 0.004
    knob_operation_speed_1_s: float = 0.65
    lever_operation_speed_1_s: float = 0.85
    button_operation_speed_1_s: float = 1.20
    operation_hold_s: float = 0.30
    # Physical contact starts near the end of the Cartesian sweep.  Seven
    # seconds leaves enough time for the rate-limited arm to finish the sweep
    # without weakening the collision/contact checks.
    move_timeout_s: float = 7.0
    operation_timeout_s: float = 4.0
    contact_threshold_n: float = 0.03
    contact_loss_timeout_s: float = 0.30
    bilateral_contact_frames: int = 4

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError("panel seed must be non-negative")
        if any(value <= 0.0 for value in self.board_size):
            raise ValueError("panel board dimensions must be positive")
        if min(
            self.console_depth_m,
            self.console_width_m,
            self.console_height_m,
            self.console_front_height_m,
            self.console_rear_flat_depth_m,
        ) <= 0.0:
            raise ValueError("panel console dimensions must be positive")
        if self.console_front_height_m >= self.console_height_m:
            raise ValueError("panel console front height must be below its rear height")
        if self.console_rear_flat_depth_m >= self.console_depth_m:
            raise ValueError("panel console rear deck must be shorter than its depth")
        if self.board_base_clearance_m < 0.0:
            raise ValueError("panel board clearance must be non-negative")
        for name, kind in (("knob", "knob"), ("lever", "lever"), ("button", "button")):
            uv = getattr(self, f"{kind}_uv")
            if abs(uv[0]) > 0.5 * self.console_width_m or abs(uv[1]) > 0.5 * self.console_depth_m:
                raise ValueError(f"{name} UV position lies outside the panel board")
        if min(
            self.knob_radius_m,
            self.knob_length_m,
            self.knob_center_standoff_m,
            self.knob_goal_rad,
            self.lever_width_m,
            self.lever_length_m,
            self.lever_pivot_standoff_m,
            self.lever_goal_rad,
            self.button_radius_m,
            self.button_length_m,
            self.button_center_standoff_m,
            self.button_travel_m,
        ) <= 0.0:
            raise ValueError("panel control dimensions and goals must be positive")
        if min(
            self.knob_operation_speed_1_s,
            self.lever_operation_speed_1_s,
            self.button_operation_speed_1_s,
            self.operation_hold_s,
            self.move_timeout_s,
            self.operation_timeout_s,
            self.contact_threshold_n,
            self.contact_loss_timeout_s,
        ) <= 0.0:
            raise ValueError("panel control speeds, timeouts, and thresholds must be positive")
        if self.bilateral_contact_frames < 1:
            raise ValueError("bilateral_contact_frames must be positive")
        self._validate_sequence(self.sequence)

    @staticmethod
    def _validate_sequence(sequence: tuple[str, ...]) -> None:
        unknown = set(sequence) - set(CONTROL_KINDS)
        if unknown:
            raise ValueError(f"unknown panel controls: {sorted(unknown)}")
        if len(set(sequence)) != len(sequence):
            raise ValueError("panel sequence must not repeat a control")

    def resolved_sequence(self) -> tuple[str, ...]:
        """Return the per-rollout instruction, sampling it when unspecified."""

        if self.sequence:
            self._validate_sequence(self.sequence)
            return self.sequence
        return sample_panel_sequence(self.seed)


@dataclass(frozen=True)
class BenchmarkConfig:
    """Complete standalone benchmark configuration."""

    # Official evaluation/recording profile.  The former 240 Hz profile is
    # retained by the CLI for training throughput, but is not scoreable.
    dt: float = 1.0 / 1000.0
    solver_substeps: int = 5
    episode_s: float = 16.0
    num_envs: int = 1
    task: Literal["pick_place", "panel_operation"] = "pick_place"
    assets: AssetConfig = field(default_factory=AssetConfig)
    panel: PanelConfig = field(default_factory=PanelConfig)
    vibration: VibrationConfig = field(default_factory=VibrationConfig)
    # Lazy-loaded so BenchmarkConfig can be constructed in the offline
    # Gamma-calibration tests without Isaac Lab.  First attribute access
    # materializes the real ShakerGeometryCfg.
    shaker: Any = field(default=None)
    material_mu: float = 1.5
    support_config: Literal["C2", "C2_CLITE"] = "C2"
    # C2_CLITE only: update the welded mocap drivers every N solver substeps.
    # N=2 keeps the workpiece-table numerical floor at the same level while
    # avoiding the full Isaac-Lab asset write on every substep.
    clite_mocap_update_decimation: int = 2
    # Single-coordinate hard-mounted deck.  Robot and worktable are placed at
    # their visible positions and both belong to the same deck support group.
    platform_size: tuple[float, float, float] = (1.60, 1.10, 0.08)
    platform_center: tuple[float, float, float] = (0.0, 0.0, 0.04)
    robot_base: tuple[float, float, float] = (-0.47, 0.0, 0.08)
    worktable_size: tuple[float, float, float] = (0.65, 0.60, 0.06)
    worktable_center: tuple[float, float, float] = (0.18, 0.0, 0.34)
    workpiece_start: tuple[float, float, float] = (0.08, -0.13, 0.47)
    workpiece_initial_clearance_m: float = 0.001
    target_center: tuple[float, float, float] = (0.08, 0.17, 0.376)
    # Positive assembly tolerance (shim) for mechanically-joined members.  It
    # only keeps same-group surfaces away from dist==0 jitter; it is not a
    # dynamic clearance and must never be used to absorb relative motion.
    assembly_clearance_m: float = 0.0005
    contact_margin_m: float = 0.001
    contact_solref: tuple[float, float] = (0.00060, 1.0)
    # Fixed geometric travel gate: a solver property, not an excitation knob.
    # 0.05 x thinnest task collision feature (8 mm target-bin wall).
    min_task_feature_thickness_m: float = 0.008
    alpha_geometry: float = 0.05
    descend_contact_threshold_n: float = 0.05
    descend_timeout_s: float = 2.0
    # Safety band between the lowest finger and the workpiece centre.  With
    # gravity enabled the floating root sags a few millimetres under the
    # current PD gains; a 2 mm band prevents that tracking sag from being
    # misclassified as table contact while still failing far before the
    # finger can reach the tabletop.
    grasp_z_guard_margin_m: float = 0.002
    approach_clearance_m: float = 0.080
    descend_clearance_m: float = 0.004
    finger_table_clearance_m: float = 0.012
    descend_position_tolerance_m: float = 0.001
    arm_linear_speed_m_s: float = 0.15
    lift_takeoff_speed_m_s: float = 0.012
    lift_takeoff_duration_s: float = 1.20
    descend_linear_speed_m_s: float = 0.060
    place_linear_speed_m_s: float = 0.080
    gripper_closing_speed_m_s: float = 0.010
    gripper_contact_recovery_speed_m_s: float = 0.001
    gripper_opening_speed_m_s: float = 0.040
    gripper_contact_preload_m: float = 0.0003
    grasp_timeout_s: float = 6.0
    # Bilateral contact must remain confirmed for this long before transport;
    # grasp_timeout_s remains a failure deadline rather than a fixed dwell.
    grasp_settle_s: float = 0.30
    grasp_contact_loss_timeout_s: float = 0.20
    grasp_slip_tolerance_m: float = 0.010
    grasp_assist: bool = False

    def __getattribute__(self, name: str):
        """Materialize the Isaac-Lab shaker geometry on first access."""

        value = object.__getattribute__(self, name)
        if name == "shaker" and value is None:
            from .models.supports.shaker import ShakerGeometryCfg

            value = ShakerGeometryCfg()
            object.__setattr__(self, "shaker", value)
        return value

    def __post_init__(self) -> None:
        if self.dt <= 0.0 or self.episode_s <= 0.0 or self.num_envs < 1:
            raise ValueError("invalid timestep, episode length, or environment count")
        if self.solver_substeps < 1:
            raise ValueError("solver_substeps must be positive")
        if self.contact_margin_m < 0.0 or self.max_substep_displacement_m <= 0.0:
            raise ValueError("contact margin and displacement limit must be non-negative/positive")
        if not 0.0 <= self.assembly_clearance_m <= 0.002:
            raise ValueError("assembly clearance must be in [0, 2] mm")
        if self.clite_mocap_update_decimation < 1:
            raise ValueError("clite mocap update decimation must be positive")
        if self.min_task_feature_thickness_m <= 0.0 or self.alpha_geometry <= 0.0:
            raise ValueError("task feature thickness and alpha_geometry must be positive")
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
        if min(
            self.grasp_timeout_s,
            self.grasp_settle_s,
            self.grasp_contact_loss_timeout_s,
        ) <= 0.0:
            raise ValueError("grasp timeout, settle time, and contact-loss timeout must be positive")
        if self.grasp_settle_s >= self.grasp_timeout_s:
            raise ValueError("grasp settle time must be shorter than the grasp timeout")
        if self.grasp_z_guard_margin_m < 0.0:
            raise ValueError("grasp z-guard margin must be non-negative")
        if self.grasp_slip_tolerance_m <= 0.0:
            raise ValueError("grasp slip tolerance must be positive")
        if self.workpiece_initial_clearance_m < 0.0:
            raise ValueError("workpiece_initial_clearance_m must be non-negative")
        if not 0.05 <= self.material_mu <= 2.0:
            raise ValueError("material_mu must be in [0.05, 2.0]")
        if self.support_config not in ("C2", "C2_CLITE"):
            raise ValueError("support_config must be C2 or C2_CLITE")
        if self.task not in ("pick_place", "panel_operation"):
            raise ValueError("task must be pick_place or panel_operation")
        if self.task == "panel_operation" and self.support_config == "C2_CLITE":
            raise ValueError("C2_CLITE support is currently only implemented for pick_place")
        if self.task == "panel_operation":
            self.panel.resolved_sequence()

    @property
    def use_clite_support(self) -> bool:
        """Whether supports are driven through mocap + weld equality constraints."""

        return self.support_config == "C2_CLITE"

    @property
    def physics_hz(self) -> int:
        """Outer physics frequency rounded to the nearest integer hertz."""

        return int(round(1.0 / self.dt))

    @property
    def effective_substep_hz(self) -> int:
        return self.physics_hz * self.solver_substeps

    @property
    def max_substep_displacement_m(self) -> float:
        """Fixed solver/geometry travel gate, independent of the excitation.

        The gate is ``alpha_geometry`` times the thinnest task collision
        feature (currently the 8 mm target-bin wall).  It changes only when
        solver behaviour or task geometry changes, never per seed/spectrum.
        """

        return self.alpha_geometry * self.min_task_feature_thickness_m

    @property
    def score_penetration_threshold_mm(self) -> float:
        """Fixed physical scoring threshold: 1% of the workpiece's thinnest collider dimension."""

        return 0.01 * 1000.0 * min(workpiece_dimensions_m(self.assets.workpiece, self.assets.workpiece_scale))

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
        """Robot root at the deck surface plus the explicit assembly shim."""

        return (
            self.robot_base[0],
            self.robot_base[1],
            self.robot_base[2] + self.assembly_clearance_m,
        )

    @property
    def resolved_worktable_center(self) -> tuple[float, float, float]:
        return self.worktable_center

    @property
    def resolved_target_center(self) -> tuple[float, float, float]:
        return self.target_center
