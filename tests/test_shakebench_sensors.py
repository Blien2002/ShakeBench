"""Quantitative tests for the Phase 05 canonical IMU contract."""

from __future__ import annotations

import numpy as np
import pytest

from robosuite.utils.shakebench_sensors import (
    ACCEL_RANGE_M_S2,
    BUTTERWORTH_A,
    BUTTERWORTH_B,
    BUTTERWORTH_CUTOFF_HZ,
    BUTTERWORTH_ENBW_HZ,
    CANONICAL_IMU_PROFILE,
    G0_M_S2,
    GRAVITY_WORLD_M_S2,
    CanonicalIMU,
    CanonicalIMUProfile,
    ButterworthLowpass,
    specific_force_from_rigid_body_motion,
)


def _zero_noise_profile(**overrides):
    values = {
        "accel_noise_density_m_s2_sqrt_hz": 0.0,
        "gyro_noise_density_rad_s_sqrt_hz": 0.0,
        "accel_initial_bias_std_m_s2": 0.0,
        "gyro_initial_bias_std_rad_s": 0.0,
        "accel_bias_diffusion_m_s2_sqrt_s": 0.0,
        "gyro_bias_diffusion_rad_s_sqrt_s": 0.0,
    }
    values.update(overrides)
    return CanonicalIMUProfile(**values)


def test_canonical_profile_matches_frozen_contract():
    profile = CANONICAL_IMU_PROFILE
    assert profile.profile_id == "canonical_midgrade_v1"
    assert profile.sample_rate_hz == pytest.approx(200.0)
    assert profile.policy_rate_hz == pytest.approx(20.0)
    assert profile.window_shape == (10, 6)
    assert profile.delivery_delay_samples == 1
    assert profile.cutoff_3db_hz == pytest.approx(40.0)
    assert profile.enbw_hz == pytest.approx(40.7618155662)
    assert profile.accel_range_m_s2 == pytest.approx(16.0 * G0_M_S2)
    assert profile.gyro_range_rad_s == pytest.approx(np.deg2rad(2000.0))
    with pytest.raises(TypeError):
        GRAVITY_WORLD_M_S2[0] = 1.0
    with pytest.raises(TypeError):
        BUTTERWORTH_A[0] = 0.0
    with pytest.raises(TypeError):
        BUTTERWORTH_B[0] = 0.0


def test_specific_force_stationary_free_fall_and_lever_arm_terms():
    stationary = specific_force_from_rigid_body_motion(np.zeros(3), np.zeros(3), np.zeros(3))
    np.testing.assert_allclose(stationary, [0.0, 0.0, G0_M_S2])

    free_fall = specific_force_from_rigid_body_motion([0.0, 0.0, -G0_M_S2], np.zeros(3), np.zeros(3))
    np.testing.assert_allclose(free_fall, np.zeros(3))

    value = specific_force_from_rigid_body_motion(
        [0.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 0.0, 3.0],
        [0.5, 0.0, 0.0],
        gravity_world_m_s2=np.zeros(3),
    )
    # alpha x r = [0, 0, -1], omega x (omega x r) = [-4.5, 0, 0].
    np.testing.assert_allclose(value, [-4.5, 0.0, -1.0])


def test_frozen_filter_cutoff_and_enbw():
    filt = ButterworthLowpass()
    assert abs(filt.frequency_response(BUTTERWORTH_CUTOFF_HZ)) == pytest.approx(1.0 / np.sqrt(2.0))
    assert abs(filt.frequency_response(9.0)) == pytest.approx(0.999265, rel=0.0, abs=2e-5)
    assert filt.enbw_hz() == pytest.approx(BUTTERWORTH_ENBW_HZ, rel=0.0, abs=1e-12)


def test_filtered_noise_rms_matches_noise_density_times_enbw():
    profile = CanonicalIMUProfile(
        accel_initial_bias_std_m_s2=0.0,
        gyro_initial_bias_std_rad_s=0.0,
        accel_bias_diffusion_m_s2_sqrt_s=0.0,
        gyro_bias_diffusion_rad_s_sqrt_s=0.0,
    )
    sensor = CanonicalIMU(seed=101, profile=profile)
    filtered = np.asarray(
        [sensor.acquire(np.zeros(6)).filtered_measurement for _ in range(30000)],
        dtype=float,
    )[1000:]
    expected = profile.noise_density * np.sqrt(profile.enbw_hz)
    np.testing.assert_allclose(filtered.std(axis=0), expected, rtol=0.06, atol=1e-8)


def test_filtered_noise_rms_matches_noise_density_times_enbw():
    profile = CanonicalIMUProfile(
        accel_initial_bias_std_m_s2=0.0,
        gyro_initial_bias_std_rad_s=0.0,
        accel_bias_diffusion_m_s2_sqrt_s=0.0,
        gyro_bias_diffusion_rad_s_sqrt_s=0.0,
    )
    sensor = CanonicalIMU(seed=101, profile=profile)
    filtered = np.asarray(
        [sensor.acquire(np.zeros(6)).filtered_measurement for _ in range(30000)],
        dtype=float,
    )[1000:]
    expected = profile.noise_density * np.sqrt(profile.enbw_hz)
    np.testing.assert_allclose(filtered.std(axis=0), expected, rtol=0.06, atol=1e-8)


def test_ideal_smoke_has_static_prefill_and_no_noise_or_quantization():
    profile = _zero_noise_profile()
    sensor = CanonicalIMU(seed=11, mode="ideal_smoke", profile=profile)
    static = np.asarray([0.1, -0.2, G0_M_S2, 0.01, -0.02, 0.03])
    sensor.reset(initial_clean_measurement=static)
    np.testing.assert_allclose(sensor.window(), np.tile(static, (10, 1)), rtol=0.0, atol=2e-7)

    sample = sensor.acquire(static, timestamp_s=0.005)
    np.testing.assert_allclose(sample.filtered_measurement, static, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(sample.delivered_measurement, static, rtol=0.0, atol=1e-12)
    assert not sample.clipping.any()
    assert np.all(sample.quantized_codes == 0)
    np.testing.assert_allclose(sensor.window(), np.tile(static, (10, 1)), rtol=0.0, atol=2e-7)


def test_delay_window_timestamps_are_one_sample_oldest_to_newest():
    sensor = CanonicalIMU(seed=0, mode="ideal_smoke", profile=_zero_noise_profile())
    sensor.reset(initial_clean_measurement=np.zeros(6))
    np.testing.assert_allclose(
        sensor.window_acquisition_timestamps_s,
        np.arange(-0.050, 0.0, 0.005),
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(sensor.pending_delivery_acquisition_timestamps_s, [0.0], rtol=0.0, atol=1e-12)
    for index in range(1, 11):
        sensor.acquire(np.full(6, float(index)), timestamp_s=index * 0.005)
    np.testing.assert_allclose(
        sensor.acquisition_timestamps_s,
        np.arange(0.005, 0.055, 0.005),
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        sensor.delivered_acquisition_timestamps_s,
        np.arange(0.0, 0.050, 0.005),
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        sensor.window_acquisition_timestamps_s,
        np.arange(0.0, 0.050, 0.005),
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(sensor.pending_delivery_acquisition_timestamps_s, [0.050], rtol=0.0, atol=1e-12)


def test_quantization_bins_and_symmetric_clipping_are_deterministic():
    sensor = CanonicalIMU(seed=0, mode="canonical_noisy_v1", profile=_zero_noise_profile())
    accel_step = sensor.profile.accel_quantization_step_m_s2
    gyro_step = sensor.profile.gyro_quantization_step_rad_s
    result = sensor.quantize_measurement(
        [
            0.49 * accel_step,
            -0.51 * accel_step,
            ACCEL_RANGE_M_S2,
            0.49 * gyro_step,
            -0.51 * gyro_step,
            -sensor.profile.gyro_range_rad_s,
        ]
    )
    np.testing.assert_array_equal(result["quantized_codes"], [0, -1, 32767, 0, -1, -32768])
    assert result["quantized_codes"].dtype == np.int16
    np.testing.assert_allclose(
        result["quantized_measurement"],
        [0.0, -accel_step, 32767 * accel_step, 0.0, -gyro_step, -32768 * gyro_step],
        rtol=0.0,
        atol=1e-15,
    )
    np.testing.assert_array_equal(result["clipping"], [False, False, False, False, False, False])
    np.testing.assert_array_equal(result["quantizer_saturation"], [False, False, True, False, False, False])

    clipped = sensor.quantize_measurement([3.0 * ACCEL_RANGE_M_S2] * 3 + [3.0 * sensor.profile.gyro_range_rad_s] * 3)
    assert clipped["clipping"].all()
    np.testing.assert_array_equal(clipped["quantized_codes"], [32767] * 3 + [32767] * 3)
    assert clipped["quantizer_saturation"].all()
    int16_codes = clipped["quantized_codes"].astype(np.int16)
    np.testing.assert_array_equal(int16_codes.astype(np.int64), clipped["quantized_codes"])


def test_sensor_samples_delivery_items_and_trace_arrays_are_immutable():
    sensor = CanonicalIMU(seed=5, mode="canonical_noisy_v1", profile=_zero_noise_profile())
    sample = sensor.acquire(np.zeros(6))
    for field in (
        "clean_measurement",
        "bias",
        "bias_increment",
        "white_noise",
        "prefilter_measurement",
        "filtered_measurement",
        "clipped_measurement",
        "quantized_codes",
        "quantized_measurement",
        "clipping",
        "quantizer_saturation",
        "delivered_measurement",
        "filter_state",
    ):
        assert not getattr(sample, field).flags.writeable
    assert not sensor._delivery_queue[0].measurement.flags.writeable
    assert not sensor.filter.state().flags.writeable
    trace = sensor.trace
    assert all(not value.flags.writeable for value in trace.values())

    copied = sample.to_dict()
    copied["clean_measurement"][0] = 123.0
    assert sample.clean_measurement[0] != 123.0
    with pytest.raises(ValueError):
        sample.clean_measurement[0] = 123.0


def test_noise_and_bias_traces_are_reproducible_after_reset():
    profile = CanonicalIMUProfile(
        accel_initial_bias_std_m_s2=0.0,
        gyro_initial_bias_std_rad_s=0.0,
        accel_bias_diffusion_m_s2_sqrt_s=0.0,
        gyro_bias_diffusion_rad_s_sqrt_s=0.0,
    )
    sensor = CanonicalIMU(seed=101, profile=profile)
    sensor.reset(initial_clean_measurement=np.zeros(6))
    first = np.asarray([sensor.acquire(np.zeros(6)).filtered_measurement for _ in range(20)])
    sensor.reset(initial_clean_measurement=np.zeros(6))
    second = np.asarray([sensor.acquire(np.zeros(6)).filtered_measurement for _ in range(20)])
    np.testing.assert_array_equal(first, second)


def test_bias_diffusion_and_sensor_replay_are_seeded():
    q_accel = 1e-3
    q_gyro = 2e-3
    profile = _zero_noise_profile(
        accel_bias_diffusion_m_s2_sqrt_s=q_accel,
        gyro_bias_diffusion_rad_s_sqrt_s=q_gyro,
    )
    first = CanonicalIMU(seed=23, profile=profile)
    second = CanonicalIMU(seed=23, profile=profile)
    increments_first = np.asarray([first.acquire(np.zeros(6)).bias_increment for _ in range(5000)])
    increments_second = np.asarray([second.acquire(np.zeros(6)).bias_increment for _ in range(5000)])
    np.testing.assert_array_equal(increments_first, increments_second)
    expected = profile.bias_diffusion * np.sqrt(profile.dt_s)
    np.testing.assert_allclose(increments_first.std(axis=0), expected, rtol=0.08, atol=1e-12)

    first.reset()
    replay = np.asarray([first.acquire(np.zeros(6)).quantized_measurement for _ in range(20)])
    first.reset()
    replay_again = np.asarray([first.acquire(np.zeros(6)).quantized_measurement for _ in range(20)])
    np.testing.assert_array_equal(replay, replay_again)
