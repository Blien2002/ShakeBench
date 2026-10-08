"""Tests for the scenario_lines_v1 vibration mode (explicit six-axis line-table scenarios)."""

import json
import os

import numpy as np
import pytest

from shakebench.physics import excitation as ex
from shakebench.physics.calibration import (
    CalibrationError,
    build_vibration_program,
    scenario_run_metadata,
    vibration_record,
)
from shakebench.physics.vibration_scenarios import RECOMMENDED, REGISTRY, build_scenario_program

G = 9.80665
TOP = np.array([0.11, 0.0, 0.299])  # tabletop centre relative to the deck origin (rotation pivot)

# multisine_v1, Gamma 0.5, seed 0 at commit 3ab08b7 *before* this integration. If your branch changed
# multisine_v1 on purpose, recompute this on the pre-integration commit and update it.
MULTISINE_BASELINE_HASH = os.environ.get("SB_MULTISINE_BASELINE_HASH", "2d918ef4329bb5d2")


@pytest.mark.parametrize("key", sorted(REGISTRY))
def test_every_scenario_respects_the_line_representation(key):
    for seed in range(5):
        prog = build_scenario_program(key, seed=seed)
        assert prog.mode == "scenario_lines_v1"
        assert prog.line_mask.sum(axis=1).max() <= ex.DEFAULT_MAX_LINES
        freqs = prog.line_frequency_hz[prog.line_mask]
        assert freqs.min() > 0.0 and freqs.max() < ex.CONSERVATIVE_MAX_LINE_FREQUENCY_HZ
        assert np.isfinite(prog.line_accel_amplitude).all()


def test_seed_determinism_and_variation():
    for key in RECOMMENDED:
        a = build_vibration_program(
            {"mode": "scenario_lines_v1", "gamma": 1.0, "seed": 3, "mode_params": {"scenario": key}}
        )
        b = build_vibration_program(
            {"mode": "scenario_lines_v1", "gamma": 1.0, "seed": 3, "mode_params": {"scenario": key}}
        )
        c = build_vibration_program(
            {"mode": "scenario_lines_v1", "gamma": 1.0, "seed": 4, "mode_params": {"scenario": key}}
        )
        assert vibration_record(a)["program_hash"] == vibration_record(b)["program_hash"]
        assert vibration_record(a)["program_hash"] != vibration_record(c)["program_hash"]
        assert a.mode_params["latents"] != c.mode_params["latents"]


def test_t0_is_a_pure_time_shift():
    t = np.linspace(1.0, 4.0, 301)
    for key in RECOMMENDED:
        p0 = build_scenario_program(key, seed=2, t0=0.0)
        p1 = build_scenario_program(key, seed=2, t0=0.7)
        np.testing.assert_allclose(p1.evaluate(t).q, p0.evaluate(t + 0.7).q, atol=1e-9)


def test_gamma_is_an_amplitude_multiplier():
    base = {"mode": "scenario_lines_v1", "seed": 1, "mode_params": {"scenario": "S4"}}
    p0 = build_vibration_program({**base, "gamma": 0.0})
    p1 = build_vibration_program({**base, "gamma": 1.0})
    ph = build_vibration_program({**base, "gamma": 0.5})
    assert np.abs(p0.line_accel_amplitude).max() == 0.0
    np.testing.assert_allclose(ph.line_accel_amplitude, 0.5 * p1.line_accel_amplitude)
    lv = build_vibration_program({**base, "gamma": 1.0, "mode_params": {"scenario": "S4", "level": 0.8}})
    np.testing.assert_allclose(lv.line_accel_amplitude, 0.8 * p1.line_accel_amplitude)
    assert p1.gamma_definition == "scenario_level_v1"


@pytest.mark.parametrize(
    "bad",
    [
        {"mode_params": {"scenario": "nope"}},
        {"mode_params": {"scenario": "S2", "params": {"bogus": 1.0}}},
        {"mode_params": {"scenario": "S2", "level": -1.0}},
        {"mode_params": {"scenario": "S2", "extra": 1}},
        {"mode_params": {"scenario": "S2"}, "gamma_definition": "normal_peak_v1"},
    ],
)
def test_invalid_configurations_are_rejected(bad):
    with pytest.raises(CalibrationError):
        build_vibration_program({"mode": "scenario_lines_v1", "gamma": 1.0, "seed": 0, **bad})


def test_replay_from_recorded_mode_params():
    p = build_vibration_program(
        {
            "mode": "scenario_lines_v1",
            "gamma": 1.0,
            "seed": 5,
            "t0_s": 0.3,
            "mode_params": {"scenario": "S3", "params": {"radius_m": 0.008}},
        }
    )
    rec = vibration_record(p)
    q = build_vibration_program(
        {
            "mode": rec["mode"],
            "gamma": rec["gamma_requested"],
            "seed": rec["excitation_seed"],
            "t0_s": rec["t0_s"],
            "mode_params": rec["mode_params"],
        }
    )
    assert vibration_record(q)["program_hash"] == rec["program_hash"]
    json.dumps(rec, allow_nan=False)


def test_existing_modes_are_unchanged():
    prog = build_vibration_program({"mode": "multisine_v1", "gamma": 0.5, "seed": 0})
    assert vibration_record(prog)["program_hash"].startswith(MULTISINE_BASELINE_HASH)
    assert "scenario_catalog_version" not in vibration_record(prog)
    assert scenario_run_metadata({"mode": "multisine_v1", "gamma": 0.5}) == {}


def test_canonical_identity_is_recorded_alongside_replay_recipe():
    config = {"mode": "scenario_lines_v1", "gamma": 1.0, "seed": 2, "mode_params": {"scenario": "S1"}}
    program = build_vibration_program(config)
    record = vibration_record(program)
    assert record["scenario_catalog_version"] == "canonical_v2"
    assert scenario_run_metadata(config)["vibration_semantics"]["scenario_catalog_version"] == "canonical_v2"
    assert record["mode_params"]["scenario"] == "S1"
    assert record["excitation_seed"] == 2
    assert "scenario_catalog_version" not in record["mode_params"]


def test_recommended_set_stays_within_documented_bounds():
    t = np.arange(0.0, 60.0, 0.002)
    for key in RECOMMENDED:
        for seed in range(10):
            m = build_scenario_program(key, seed=seed).evaluate(t)
            disp = m.q[:, :3] + np.cross(m.q[:, 3:], TOP)
            acc_z = m.qdd[:, 2] + np.cross(m.qdd[:, 3:], TOP)[:, 2]
            assert np.abs(disp).max() < 0.030, (key, seed)  # tabletop centre travel
            assert np.abs(acc_z).max() < 1.0 * G, (key, seed)  # no ballistic lift of the tabletop


@pytest.mark.skipif(os.environ.get("SHAKEBENCH_SLOW_TESTS") != "1", reason="builds the full robosuite environment")
def test_environment_smoke():
    from shakebench import models
    from shakebench.tasks.runtime import make_environment

    states = json.load(open(os.path.join(models.assets_root, "shakebench_task_states_official.json")))["states"]
    state = next(s for s in states if s.get("task", {}).get("object_id") == "apple")
    for key in RECOMMENDED:
        env, program = make_environment(
            state, gamma=1.0, horizon=40, mode="scenario_lines_v1", mode_params={"scenario": key}
        )
        assert program.mode == "scenario_lines_v1" and program.mode_params["scenario"] == key
        action = np.zeros(env.action_dim)
        for _ in range(4):
            env.step(action)
        env.close()


@pytest.mark.parametrize(
    "key,params",
    [
        ("S1", {"amp_deg": -1}),
        ("S3", {"radius_m": -0.1}),
        ("S2", {"h_rms": -1}),
        ("S2", {"v_rms": -1}),
        ("S2", {"n": 0}),
        ("S2", {"n": 13}),
        ("S2", {"n": 2.5}),
        ("S2", {"n": 2.0}),
        ("S2", {"n": True}),
        ("S2", {"f_lo": 0}),
        ("S2", {"f_lo": 6, "f_hi": 2}),
        ("S2", {"f_hi": ex.CONSERVATIVE_MAX_LINE_FREQUENCY_HZ}),
        ("S2", {"h_rms": float("inf")}),
        ("S2", {"f_hi": "6"}),
        ("S4", {"zeta": 0}),
        ("S4", {"gamma_peak": -1}),
        ("S4", {"rho_h": -1}),
    ],
)
def test_parameter_boundaries_fail_at_the_public_config(key, params):
    with pytest.raises(CalibrationError):
        build_vibration_program(
            {"mode": "scenario_lines_v1", "gamma": 1, "mode_params": {"scenario": key, "params": params}}
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"seed": -1},
        {"seed": 1.5},
        {"seed": True},
        {"t0": float("inf")},
        {"t0": True},
        {"level": -1},
        {"level": True},
        {"level_scale": -1},
        {"level_scale": float("nan")},
        {"level": -1, "level_scale": -1},
        {"level": 1e308, "level_scale": 1e308},
        {"params": []},
        {"params": {1: 2}},
    ],
)
def test_direct_api_validates_replay_and_scaling(overrides):
    with pytest.raises(ValueError):
        build_scenario_program("S3", **overrides)


def test_orbital_radius_frequency_direction_and_axes():
    for seed in range(8):
        p = build_scenario_program("S3", seed=seed)
        assert p.active_axes == ("tx", "ty", "tz")
        latent = p.mode_params["latents"]
        assert latent["radius_m"] == 0.010
        assert 2 <= latent["f0_hz"] <= 2.9
        w = p.line_omega_rad_s[0, 0]
        assert p.line_accel_amplitude[0, 0] / w**2 == pytest.approx(0.010)
        assert p.line_accel_amplitude[1, 0] / w**2 == pytest.approx(0.010)
        phase = p.line_phase_at_episode_zero[:2, 0]
        # Oriented area of the main circular orbit in deck x/y; positive means ccw from +z.
        q = -0.010 * np.sin(phase)
        v = -0.010 * w * np.cos(phase)
        area_rate = q[0] * v[1] - q[1] * v[0]
        assert (area_rate > 0) == (latent["direction"] == "ccw")


def test_road_line_rms_and_roll_axis():
    p = build_scenario_program("S2", seed=2)
    assert p.active_axes == ("tx", "ty", "tz")
    np.testing.assert_allclose(np.sqrt(np.sum(p.line_accel_amplitude**2, axis=1) / 2)[:3], [1.98, 1.98, 2.34])
    assert np.all((p.line_frequency_hz[p.line_mask] >= 2) & (p.line_frequency_hz[p.line_mask] <= 6))
    assert not np.array_equal(p.line_frequency_hz[0], p.line_frequency_hz[1])
    roll = build_scenario_program("S1", seed=2)
    assert roll.active_axes == ("rx", "ry", "rz")
    displacement = roll.line_accel_amplitude[3:5, 0] / roll.line_omega_rad_s[3:5, 0] ** 2
    assert np.linalg.norm(displacement) == pytest.approx(np.deg2rad(3.5))


def test_press_peak_and_alternation():
    p = build_scenario_program("S4", seed=3)
    fr = p.mode_params["latents"]["stroke_hz"]
    psi = p.mode_params["latents"]["psi_rad"]
    t = np.linspace(1, 1 + 2 / fr, 16001)
    a = p.evaluate(t).qdd
    later = p.evaluate(t + 1 / fr).qdd
    np.testing.assert_allclose(a[:, 2], later[:, 2], atol=1e-11)
    np.testing.assert_allclose(a[:, :2], -later[:, :2], atol=1e-11)
    assert np.max(np.abs(a[:, 2])) / G == pytest.approx(0.7, rel=3e-5)
    along = a[:, 0] * np.cos(psi) + a[:, 1] * np.sin(psi)
    assert np.max(np.abs(along)) / G == pytest.approx(1.05, rel=3e-5)
    np.testing.assert_allclose(a[:, 0] * np.sin(psi) - a[:, 1] * np.cos(psi), 0, atol=1e-11)


@pytest.mark.parametrize("key", sorted(REGISTRY))
def test_gamma_zero_is_static_for_every_scenario(key):
    p = build_vibration_program({"mode": "scenario_lines_v1", "gamma": 0, "mode_params": {"scenario": key}})
    sample = p.evaluate(np.array([-1, 0, 0.05, 0.3, 2, 60]))
    for array in (sample.q, sample.qdot, sample.qdd):
        np.testing.assert_array_equal(array, np.zeros_like(array))


def test_analytic_velocity_acceleration_and_ramp_boundary():
    for key in RECOMMENDED:
        p = build_scenario_program(key, seed=8, t0=0.7)
        zero = p.evaluate(0)
        assert not np.any(zero.q) and not np.any(zero.qdot) and not np.any(zero.qdd)
        t = np.array([0.03, 0.1, 0.24, 1.1, 2.3])
        h = 1e-5
        m, plus, minus = p.evaluate(t), p.evaluate(t + h), p.evaluate(t - h)
        np.testing.assert_allclose((plus.q - minus.q) / (2 * h), m.qdot, atol=1e-7)
        np.testing.assert_allclose((plus.qdot - minus.qdot) / (2 * h), m.qdd, atol=1e-5)
        shifted = build_scenario_program(key, seed=8, t0=0)
        # t0 shifts the carrier; the startup ramp remains tied to episode zero.
        for field in ("q", "qdot", "qdd"):
            np.testing.assert_allclose(
                getattr(p.evaluate([1, 2]), field), getattr(shifted.evaluate([1.7, 2.7]), field), atol=1e-10
            )


def test_run_metadata_and_packaged_suite():
    from pathlib import Path

    from shakebench import models
    from shakebench.physics.calibration import GAMMA_DEFINITIONS, scenario_run_metadata

    assert "scenario_level_v1" not in GAMMA_DEFINITIONS
    assert scenario_run_metadata({"mode": "multisine_v1", "gamma": 0.5}) == {}
    meta = scenario_run_metadata(
        {"mode": "scenario_lines_v1", "gamma": 0.5, "mode_params": {"scenario": "S3", "level": 0.8}}
    )["vibration_semantics"]
    assert meta["amplitude_multiplier"] == 0.4
    assert meta["gamma_definition"] == "scenario_level_v1"
    assert meta["comparable_to_peak_gamma_grid"] is False
    assert meta["axis_order"] == list(ex.AXES)
    assert meta["coordinate_units"] == ["m"] * 3 + ["rad"] * 3
    suite = json.loads(Path(models.assets_root, "shakebench_vibration_suite_v2.json").read_text())
    assert tuple(s["scenario"] for s in suite["scenarios"]) == RECOMMENDED


def test_scenario_validation_precedes_policy_import(tmp_path):
    from shakebench.evaluation.evaluate import main

    with pytest.raises(CalibrationError):
        main(
            [
                "--policy",
                "missing_policy:nope",
                "--mode",
                "scenario_lines_v1",
                "--mode-params",
                '{"scenario":"S2","params":{"n":0}}',
            ]
        )


def test_only_canonical_scenarios_are_public():
    assert tuple(REGISTRY) == ("S1", "S2", "S3", "S4")
    assert RECOMMENDED == ("S1", "S2", "S3", "S4")
    for retired in ("S1r", "S2w", "S2r", "S2d", "S3_v1", "S4r", "S4c", "S4b"):
        with pytest.raises(ValueError, match="unknown scenario"):
            build_scenario_program(retired)
