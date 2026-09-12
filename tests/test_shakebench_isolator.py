"""Analytic and physics-only probes for the Phase 03 six-DoF isolator."""

from __future__ import annotations

import math

import mujoco
import numpy as np
import pytest

from robosuite.models.arenas import ShakeBenchArena
from robosuite.utils.shakebench_deck import DeckDriverConfig, DeckDriverError
from robosuite.utils.shakebench_isolator import (
    AXES,
    CANONICAL_WORKTABLE_INERTIA_KG_M2,
    CANONICAL_WORKTABLE_MASS_KG,
    IsolatorConfig,
    IsolatorSafetyError,
    check_isolator_safety,
    derive_isolator_parameters,
    derive_k_c,
    parameter_sweep,
    payload_static_offset,
    relative_transmissibility,
    single_axis_transfer,
    static_equilibrium_offset,
    static_sag_uncompensated_m,
    transfer_function,
    transfer_metrics,
    transmissibility,
    validate_isolator_safety,
    vertical_preload_springref,
)


def test_canonical_k_c_and_preload_are_explicit_and_per_axis():
    config = IsolatorConfig(
        fn_hz=(3.0, 4.0, 5.0, 6.0, 7.0, 8.0),
        zeta=(0.05, 0.10, 0.15, 0.20, 0.25, 0.30),
    )
    parameters = derive_isolator_parameters(config)
    effective_mass = np.asarray((32.0, 32.0, 32.0, *CANONICAL_WORKTABLE_INERTIA_KG_M2))
    omega = 2.0 * math.pi * np.asarray(config.fn_hz)
    np.testing.assert_allclose(parameters.effective_mass, effective_mass, rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(parameters.stiffness, effective_mass * omega**2, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(
        parameters.damping,
        2.0 * np.asarray(config.zeta) * effective_mass * omega,
        rtol=0.0,
        atol=1e-12,
    )
    assert parameters.mass_kg == CANONICAL_WORKTABLE_MASS_KG
    assert parameters.inertia_kg_m2 == CANONICAL_WORKTABLE_INERTIA_KG_M2
    assert parameters.springref[0] == parameters.springref[1] == 0.0
    assert parameters.springref[3:] == (0.0, 0.0, 0.0)
    assert vertical_preload_springref(parameters) == pytest.approx(
        config.mass_kg * config.gravity_m_s2 / parameters.stiffness[2]
    )
    assert static_sag_uncompensated_m(parameters) == pytest.approx(parameters.springref[2])
    np.testing.assert_allclose(derive_k_c(config)[0], parameters.k, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(derive_k_c(config)[1], parameters.c, rtol=0.0, atol=0.0)


def test_payload_changes_equilibrium_but_never_rederives_support():
    parameters = derive_isolator_parameters()
    empty = static_equilibrium_offset(parameters)
    np.testing.assert_array_equal(empty, np.zeros(6))
    centered = payload_static_offset(parameters, payload_mass_kg=0.349)
    assert centered[2] == pytest.approx(-0.349 * 9.81 / parameters.stiffness[2])
    assert centered[2] < 0.0
    offset = static_equilibrium_offset(parameters, payload_mass_kg=0.349, payload_com_m=(0.10, 0.20, 0.0))
    assert offset[3] != 0.0
    assert offset[4] != 0.0
    assert offset[2] == pytest.approx(centered[2])
    repeated = derive_isolator_parameters(parameters_to_config(parameters))
    np.testing.assert_array_equal(parameters.stiffness, repeated.stiffness)
    np.testing.assert_array_equal(parameters.damping, repeated.damping)


def parameters_to_config(parameters):
    """Test helper kept local so payload probes do not expose mutable state."""

    return IsolatorConfig(
        fn_hz=parameters.fn_hz,
        zeta=parameters.zeta,
        mass_kg=parameters.mass_kg,
        inertia_kg_m2=parameters.inertia_kg_m2,
        gravity_m_s2=parameters.gravity_m_s2,
    )


def test_transfer_function_matches_closed_form_and_supports_six_axis_grid():
    frequency = 5.0
    zeta = 0.1
    absolute = transfer_function(frequency, 5.0, zeta)
    relative = transfer_function(frequency, 5.0, zeta, output="relative")
    assert abs(absolute) == pytest.approx(math.sqrt(1.0 + 0.2**2) / 0.2)
    assert abs(relative) == pytest.approx(1.0 / 0.2)
    assert transmissibility(frequency, 5.0, zeta) == pytest.approx(abs(absolute))
    assert relative_transmissibility(frequency, 5.0, zeta) == pytest.approx(abs(relative))
    assert abs(single_axis_transfer("tz", frequency)) == pytest.approx(abs(absolute))

    parameters = derive_isolator_parameters()
    frequencies = np.asarray([0.0, 1.0, 5.0, 10.0])
    grid = transfer_function(frequencies, parameters.fn_hz, parameters.zeta)
    assert grid.shape == (4, 6)
    assert np.all(np.isfinite(grid))
    assert np.allclose(grid[0], 1.0 + 0.0j)
    metrics = transfer_metrics(frequencies, parameters, base_displacement_amplitude_m=0.001)
    assert set(
        ("T_accel", "R_relative", "D_relative", "T_peak", "travel_margin_m", "static_sag_uncompensated_m")
    ) <= set(metrics)
    assert metrics["T_peak"] > 1.0


def test_parameter_sweep_is_deterministic_and_does_not_select_a_final_point():
    candidates = parameter_sweep((3.0, 5.0), (0.05, 0.10))
    assert len(candidates) == 4
    assert [candidate.fn_hz[0] for candidate in candidates] == [3.0, 3.0, 5.0, 5.0]
    assert [candidate.zeta[0] for candidate in candidates] == [0.05, 0.10, 0.05, 0.10]
    assert candidates[0].stiffness != candidates[2].stiffness


def test_travel_and_angle_limits_fail_closed_at_and_above_boundary():
    config = IsolatorConfig(travel_limits_m=(0.01, 0.02, 0.03), angle_limits_rad=(0.1, 0.2, 0.3))
    assert check_isolator_safety(np.zeros(6), config).passed
    below = check_isolator_safety(np.asarray([0.009, 0.019, 0.029, 0.099, 0.199, 0.299]), config)
    assert below.passed
    at_boundary = check_isolator_safety(np.asarray([0.01, 0.0, 0.0, 0.0, 0.0, 0.0]), config)
    assert not at_boundary.passed
    assert "travel_tx" in [check.name for check in at_boundary.violations]
    above = check_isolator_safety(np.asarray([[0.0, 0.0, 0.0, 0.0, 0.0, 0.31]]), config)
    assert not above.passed
    assert "angle_rz" in [check.name for check in above.violations]
    invalid = check_isolator_safety(np.asarray([np.nan] * 6), config)
    assert not invalid.passed
    with pytest.raises(IsolatorSafetyError):
        validate_isolator_safety(np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.31]), config)


def test_phase02_timestep_and_equality_time_constant_reject_unsafe_pair():
    with pytest.raises(DeckDriverError, match=r"at least 2 \* physics_timestep_s"):
        DeckDriverConfig(physics_timestep_s=0.001, eq_solref=(0.0019, 0.5))


def _run_single_axis_mujoco_probe(axis_index: int) -> tuple[float, float]:
    arena = ShakeBenchArena()
    dt = 0.0002
    deck_config = DeckDriverConfig(physics_timestep_s=dt, eq_solref=(2.0 * dt, 0.5))
    xml = arena.process_deck_xml(config=deck_config)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    driver_id = model.body("deck_driver").id
    mocap_id = int(model.body_mocapid[driver_id])
    axis = AXES[axis_index]
    joint_id = model.joint(f"isolator_{axis}").id
    qpos_address = int(model.jnt_qposadr[joint_id])
    frequency_hz = 4.0
    omega = 2.0 * math.pi * frequency_hz
    amplitude = 1e-4 if axis_index < 3 else 1e-3
    samples_t = []
    samples_q = []
    mujoco.mj_forward(model, data)
    for _ in range(int(1.2 / dt)):
        time_s = float(data.time)
        command = amplitude * math.sin(omega * time_s)
        data.mocap_pos[mocap_id] = 0.0
        data.mocap_quat[mocap_id] = (1.0, 0.0, 0.0, 0.0)
        if axis_index < 3:
            data.mocap_pos[mocap_id, axis_index] = command
        else:
            rotation_axis = np.zeros(3)
            rotation_axis[axis_index - 3] = 1.0
            data.mocap_quat[mocap_id] = np.concatenate(
                ([math.cos(command / 2.0)], rotation_axis * math.sin(command / 2.0))
            )
        mujoco.mj_step(model, data)
        if time_s >= 0.6:
            samples_t.append(time_s)
            samples_q.append(float(data.qpos[qpos_address]))
    times = np.asarray(samples_t)
    values = np.asarray(samples_q)
    design = np.column_stack((np.sin(omega * times), np.cos(omega * times), np.ones_like(times)))
    coefficients = np.linalg.lstsq(design, values, rcond=None)[0]
    measured_relative_amplitude = float(np.linalg.norm(coefficients[:2]) / amplitude)
    expected_relative_amplitude = abs(relative_transmissibility(frequency_hz, 5.0, 0.1))
    assert np.all(data.warning.number == 0)
    return measured_relative_amplitude, expected_relative_amplitude


@pytest.mark.parametrize("axis_index", range(6), ids=AXES)
def test_six_single_axis_mujoco_transfers_match_analytic_envelope(axis_index):
    measured, expected = _run_single_axis_mujoco_probe(axis_index)
    # The analytic model is the isolator contract. The small extra error is
    # the intentionally measured Phase 02 equality-weld / discrete-step
    # layer, so this physics-only gate is explicit rather than hidden.
    assert measured == pytest.approx(expected, rel=0.05, abs=1e-4)
