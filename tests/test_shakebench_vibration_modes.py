from __future__ import annotations

import numpy as np
import pytest

from robosuite.environments.manipulation.vibration_pick_place import VibrationPickPlace
from robosuite.utils.shakebench_calibration import (
    CalibrationError,
    build_vibration_program,
    calibrate_gamma,
    vibration_record,
)


def test_public_vibration_modes_are_calibrated_and_replayable() -> None:
    config = {
        "mode": "single_sine_v1",
        "gamma": 0.15,
        "seed": 42,
        "t0_s": 0.1,
        "mode_params": {"frequency_scale": 0.8, "active_axes": ["tz"]},
        "gamma_definition": "normal_peak_v1",
    }
    first = build_vibration_program(config)
    replay = build_vibration_program(config)
    assert first.gamma_requested == 0.15
    assert calibrate_gamma(first, gamma_definition=first.gamma_definition).gamma_commanded == pytest.approx(0.15)
    assert np.count_nonzero(first.line_mask) == 1
    np.testing.assert_array_equal(first.evaluate([0.2, 0.8]).qdd, replay.evaluate([0.2, 0.8]).qdd)
    record = vibration_record(first)
    assert record["mode"] == "single_sine_v1"
    assert record["excitation_seed"] == 42
    assert len(record["program_hash"]) == 64


def test_custom_mode_and_gamma_definition_are_explicit() -> None:
    program = build_vibration_program(
        {
            "mode": "custom_multisine_v1",
            "gamma": 0.2,
            "mode_params": {
                "active_axes": ["tx"],
                "bands": {"tx": {"center_hz": 4.0, "bandwidth_ratio": 0.1, "relative_accel_rms": 0.5, "tones": 3}},
            },
            "gamma_definition": "magnitude_peak_v1",
        }
    )
    assert np.count_nonzero(program.line_mask[0]) == 3
    assert program.gamma_definition == "magnitude_peak_v1"


def test_vibration_validation_and_zero_gamma() -> None:
    zero = build_vibration_program({"mode": "single_sine_v1", "gamma": 0.0})
    np.testing.assert_array_equal(zero.line_accel_amplitude, 0.0)
    with pytest.raises(CalibrationError, match="unit vibration Gamma is zero"):
        build_vibration_program(
            {
                "mode": "single_sine_v1",
                "gamma": 0.1,
                "mode_params": {"active_axes": ["tx"]},
                "gamma_definition": "normal_peak_v1",
            }
        )
    with pytest.raises(CalibrationError, match="unknown mode parameter"):
        build_vibration_program({"mode": "multisine_v1", "gamma": 0.1, "mode_params": {"typo": 1}})


def test_gamma_only_scales_amplitude() -> None:
    low = build_vibration_program({"mode": "multisine_v1", "gamma": 0.1, "seed": 7})
    high = build_vibration_program({"mode": "multisine_v1", "gamma": 0.2, "seed": 7})
    np.testing.assert_array_equal(low.line_omega_rad_s, high.line_omega_rad_s)
    np.testing.assert_array_equal(low.line_phase_at_episode_zero, high.line_phase_at_episode_zero)
    np.testing.assert_allclose(high.line_accel_amplitude, 2.0 * low.line_accel_amplitude)


def test_environment_uses_one_vibration_source() -> None:
    kwargs = {
        "robots": "Panda",
        "physics_profile": "probe",
        "has_renderer": False,
        "has_offscreen_renderer": False,
        "use_camera_obs": False,
        "use_object_obs": False,
        "load_model_on_init": False,
        "seed": 7,
    }
    env = VibrationPickPlace(vibration={"mode": "single_sine_v1", "gamma": 0.1}, **kwargs)
    assert env.deck_driver.trajectory.mode == "single_sine_v1"
    with pytest.raises(ValueError, match="cannot be combined"):
        VibrationPickPlace(vibration={"gamma": 0.1}, deck_trajectory=lambda time: time, **kwargs)
