"""Six-axis excitation, calibration, and synthetic vibration sensors."""

from .spectral import (
    DeckMotionReport,
    SpectralVibration,
    SupportTravelReport,
    _euler_rotation_matrices,
    _synthesize_episode,
    calibrate_level_scale,
    displacement_gate_gamma_max,
    offline_deck_motion_report,
    offline_support_travel_report,
    validate_deck_displacement_gate,
)
from .sensors import SyntheticDeckIMU

__all__ = [
    "DeckMotionReport",
    "SpectralVibration",
    "SupportTravelReport",
    "SyntheticDeckIMU",
    "calibrate_level_scale",
    "displacement_gate_gamma_max",
    "offline_deck_motion_report",
    "offline_support_travel_report",
    "validate_deck_displacement_gate",
]
