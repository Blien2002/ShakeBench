"""Phase 02 dynamic deck, timing, instrumentation, and negative-regression tests."""

from __future__ import annotations

import copy
import inspect
import json
import math
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

import robosuite.macros as macros
from robosuite.environments.base import MujocoEnv
from robosuite.environments.manipulation.lift import Lift
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.environments.robot_env import RobotEnv
from robosuite.scripts.shakebench_probe_deck_driver import (
    CANONICAL_LOAD_CASES,
    DeckProbeEnv,
    LOAD_CASE_PANDA_PLUS_WORKTABLE_REFERENCE_PROXY,
    Phase02R4Thresholds,
    _actual_coordinates,
    _sine_command,
    build_probe_xml,
    compute_gate_summary,
    gamma_conformance,
    main as probe_main,
    minimum_line_spacing_hz,
    select_candidate,
    synthetic_spectrum_estimator_validation,
    verify_artifact,
    run_trace,
    run_probe_suite,
    sine_conformance,
    spectrum_conformance,
)
from robosuite.utils.shakebench_deck import (
    DeckBodyHandles,
    DeckCommand,
    DeckDriver,
    DeckDriverConfig,
    DeckDriverError,
    DeckXMLProcessingError,
    ShakeBenchDeckXMLProcessor,
    audit_compiled_deck_model,
    audit_deck_xml,
)
from robosuite.controllers.parts.controller import get_model_timestep
from robosuite.controllers.parts.gripper.gripper_controller import get_model_timestep as get_gripper_model_timestep
from robosuite.controllers.parts.mobile_base.mobile_base_controller import (
    get_model_timestep as get_mobile_model_timestep,
)
from robosuite.utils.shakebench_excitation import build_excitation_program
from robosuite.utils.errors import SimulationError


def _empty_xml() -> str:
    return '<mujoco><option gravity="0 0 0"/><asset/><worldbody/></mujoco>'


def _compile_deck_xml(body_handles=None, *, config=None) -> tuple[str, DeckDriverConfig]:
    config = config if config is not None else DeckDriverConfig()
    source = """
      <body name="robot_mount" pos="0.2 0 0">
        <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
      </body>
      <body name="payload_root" pos="0 0 0.1">
        <inertial pos="0 0 0" mass="1" diaginertia="0.1 0.1 0.1"/>
      </body>
    """
    xml = '<mujoco><option gravity="0 0 0"/><asset/><worldbody>' + source + "</worldbody></mujoco>"
    processor = ShakeBenchDeckXMLProcessor(config=config, body_handles=body_handles or {})
    return processor(xml), config


def test_environment_owned_timestep_is_local_and_reaches_controller():
    original_macro = macros.SIMULATION_TIMESTEP
    env = Lift(
        robots="Panda",
        use_camera_obs=False,
        has_offscreen_renderer=False,
        renderer="mujoco",
        model_timestep=0.001,
        control_freq=20,
        initialization_noise=None,
    )
    try:
        assert macros.SIMULATION_TIMESTEP == original_macro
        assert env.sim.model.opt.timestep == pytest.approx(0.001)
        assert env.model_timestep == pytest.approx(env.sim.model.opt.timestep)
        assert env.control_timestep == pytest.approx(0.05)
        assert env._control_steps == 50
        assert env.robots[0].part_controllers["right"].model_timestep == pytest.approx(0.001)
    finally:
        env.close()

    default_env = Lift(
        robots="Panda",
        use_camera_obs=False,
        has_offscreen_renderer=False,
        renderer="mujoco",
        control_freq=20,
        initialization_noise=None,
    )
    try:
        assert default_env.sim.model.opt.timestep == pytest.approx(original_macro)
        assert default_env.model_timestep == pytest.approx(original_macro)
        assert macros.SIMULATION_TIMESTEP == original_macro
    finally:
        default_env.close()


def test_two_explicit_timestep_environments_do_not_cross_talk():
    common = {
        "robots": "Panda",
        "use_camera_obs": False,
        "has_offscreen_renderer": False,
        "renderer": "mujoco",
        "control_freq": 20,
        "initialization_noise": None,
    }
    first = Lift(model_timestep=0.001, **common)
    second = Lift(model_timestep=0.002, **common)
    try:
        assert first.model_timestep == pytest.approx(0.001)
        assert second.model_timestep == pytest.approx(0.002)
        assert first.sim.model.opt.timestep == pytest.approx(0.001)
        assert second.sim.model.opt.timestep == pytest.approx(0.002)
        assert first._control_steps == 50
        assert second._control_steps == 25
    finally:
        first.close()
        second.close()


def test_existing_lift_remains_deterministic_with_explicit_timestep():
    kwargs = {
        "robots": "Panda",
        "use_camera_obs": False,
        "has_offscreen_renderer": False,
        "renderer": "mujoco",
        "control_freq": 20,
        "model_timestep": 0.001,
        "initialization_noise": None,
        "seed": 23,
    }
    env1 = Lift(**kwargs)
    env2 = Lift(**kwargs)
    try:
        np.testing.assert_array_equal(env1.sim.get_state().flatten(), env2.sim.get_state().flatten())
        assert env1.sim.model.get_xml() == env2.sim.model.get_xml()
    finally:
        env1.close()
        env2.close()


def test_timestep_parameter_is_present_on_affected_parent_and_existing_env_signatures():
    for cls in (MujocoEnv, RobotEnv, ManipulationEnv, Lift):
        assert "model_timestep" in inspect.signature(cls.__init__).parameters


def test_control_timestep_must_be_representable_by_model_steps():
    driver = DeckDriver(config=DeckDriverConfig(physics_timestep_s=0.003, weld_solref=(0.006, 1.0)))
    with pytest.raises(SimulationError, match="integer multiple"):
        DeckProbeEnv(
            build_probe_xml(model_timestep=0.003),
            driver,
            model_timestep=0.003,
            control_freq=20,
        ).reset()


def test_late_install_fails_closed_and_preinit_driver_survives_both_reset_modes():
    dt = 0.0002
    config = DeckDriverConfig(physics_timestep_s=dt, weld_solref=(2.0 * dt, 1.0))
    installed = DeckDriver(config=config)
    env = DeckProbeEnv(
        build_probe_xml(model_timestep=dt),
        installed,
        model_timestep=dt,
        control_freq=20,
        horizon=20,
    )
    try:
        env.reset()
        assert installed._sim is env.sim
        assert installed._sim_token == id(env.sim)
        processor_count = len(env._xml_processors)
        pre_hook_count = len(env._pre_physics_step_hooks)
        late_driver = DeckDriver()
        with pytest.raises(DeckDriverError, match="before simulation initialization"):
            late_driver.install(env)
        assert len(env._xml_processors) == processor_count
        assert len(env._pre_physics_step_hooks) == pre_hook_count

        env.step(np.zeros(0))
        env.reset()  # hard-reset path recompiles and re-audits the installed deck
        env.step(np.zeros(0))
        env.hard_reset = False
        env.reset()  # non-hard-reset path reuses the already audited compiled deck
        env.step(np.zeros(0))
    finally:
        env.close()


@pytest.mark.parametrize(
    "sim",
    (
        SimpleNamespace(),
        SimpleNamespace(model=SimpleNamespace()),
        SimpleNamespace(model=SimpleNamespace(opt=SimpleNamespace())),
    ),
)
def test_controller_timestep_helper_falls_back_without_model_or_option(sim):
    assert get_model_timestep(sim) == pytest.approx(macros.SIMULATION_TIMESTEP)


def test_all_part_controller_families_use_the_shared_timestep_helper():
    assert get_gripper_model_timestep is get_model_timestep
    assert get_mobile_model_timestep is get_model_timestep


def test_world_spatial_rotation_state_respects_nonidentity_base_orientation():
    dt = 0.0002
    base_quat = (math.cos(math.pi / 4.0), 0.0, 0.0, math.sin(math.pi / 4.0))
    config = DeckDriverConfig(
        deck_quat_wxyz=base_quat,
        physics_timestep_s=dt,
        weld_solref=(2.0 * dt, 1.0),
    )
    trace, _ = run_trace(
        dt=dt,
        duration_s=0.4,
        trajectory=_sine_command(3, 5.0, 0.0005),
        config=config,
    )
    mask = trace.sample_timestamps_s >= 0.2
    command_angular = np.max(np.abs(trace.command_twist[mask, 3:]), axis=0)
    actual_angular = np.max(np.abs(trace.actual_twist[mask, 3:]), axis=0)
    command_alpha = np.max(np.abs(trace.command_acceleration[mask, 3:]), axis=0)
    actual_alpha = np.max(np.abs(trace.actual_acceleration[mask, 3:]), axis=0)

    # A local +x rotation under a +90-degree world-z base orientation is a
    # world +y spatial angular motion, for both velocity and acceleration.
    assert command_angular[1] > 10.0 * command_angular[0]
    assert actual_angular[1] > 10.0 * actual_angular[0]
    assert command_alpha[1] > 10.0 * command_alpha[0]
    assert actual_alpha[1] > 10.0 * actual_alpha[0]


def test_trace_separates_tracking_pose_error_from_raw_weld_diagnostics():
    trace, _ = run_trace(dt=0.0002, duration_s=0.12, trajectory=_sine_command(0, 5.0, 0.001))
    assert trace.deck_tracking_pose_error.shape == (trace.sample_timestamps_s.size, 6)
    assert trace.weld_constraint_residual_raw.shape == (trace.sample_timestamps_s.size, 6)
    assert trace.weld_constraint_force_raw.shape == (trace.sample_timestamps_s.size, 6)
    assert np.isfinite(trace.deck_tracking_pose_error).all()
    assert np.isfinite(trace.weld_constraint_residual_raw).all()
    assert np.isfinite(trace.weld_constraint_force_raw).all()
    assert not np.allclose(trace.deck_tracking_pose_error, trace.weld_constraint_residual_raw)
    assert "deck_tracking_pose_error" in trace.to_dict()
    assert "weld_constraint_residual_raw" in trace.to_dict()
    assert "weld_constraint_force_raw" in trace.to_dict()
    assert "weld_wrench" not in trace.to_dict()
    assert trace.to_dict()["field_contract"]["actual_twist"].startswith("world spatial")
    assert trace.to_dict()["field_contract"]["weld_constraint_force_raw"].startswith("MuJoCo efc_force")
    with pytest.raises(TypeError):
        trace.field_contract["actual_twist"] = "mutated"


def test_probe_matrix_records_each_required_coverage_axis_and_line():
    result = run_probe_suite(dt=0.0005, duration_s=0.1, spectrum_duration_s=2.0)
    families = {row["case_family"]: row for row in result["coverage_matrix"]}
    assert set(families) >= {
        "zero",
        "single_axis",
        "authored_spectrum",
        "target_gamma",
        "load",
        "dt_convergence",
        "solver_sensitivity",
    }
    assert families["single_axis"]["axes"] == ["tx", "ty", "tz", "rx", "ry", "rz"]
    assert len(result["provenance"]["candidate_grid"]) == 3
    assert all(candidate["status"] == "measured" for candidate in result["provenance"]["candidate_grid"])
    assert all("screening" in candidate and "physics_metrics" in candidate for candidate in result["provenance"]["candidate_grid"])
    assert all(set(candidate["screening"]["single_axis"]) == set(families["single_axis"]["axes"]) for candidate in result["provenance"]["candidate_grid"])
    assert all(len(candidate["screening"]["authored_spectrum"]["conformance"]["axes"]["tx"]["line_fits"]) == 12 for candidate in result["provenance"]["candidate_grid"])
    assert result["provenance"]["candidate_selection"]["selection_status"] == "blocked_no_candidate"
    assert result["confirmatory"]["status"] == "not_run_no_candidate"
    assert result["phase03_handoff"] == "BLOCKED"
    assert result["load_cases"] == list(CANONICAL_LOAD_CASES)
    assert result["load_matrix"] == []


def test_probe_matrix_never_labels_level_scale_as_gamma():
    result = run_probe_suite(dt=0.0005, duration_s=0.1, spectrum_duration_s=2.0)
    for candidate in result["provenance"]["candidate_grid"]:
        assert "commanded_level_scale" not in candidate
        assert "dt_s" in candidate
        screen = candidate["screening"]
        assert "level_scale" in screen["authored_spectrum"]
        assert screen["target_gamma"]["Gamma_commanded"] == pytest.approx(0.30, rel=0.0, abs=0.01)
        assert "Gamma_deck_actual" in screen["target_gamma"]


def test_spectrum_resolution_and_synthetic_estimator_gate_are_pre_registered():
    program = build_excitation_program(seed=17, level_scale=0.3)
    spacing = minimum_line_spacing_hz(program)
    assert spacing == pytest.approx(0.05887579362532458, rel=0.0, abs=1e-12)
    synthetic = synthetic_spectrum_estimator_validation(program, sample_dt_s=0.0005)
    assert synthetic["required_fit_window_s"] >= 2.0 / spacing
    assert synthetic["passed"]
    assert synthetic["conditioning_rule"]["max_condition_number"] == pytest.approx(10000.0)


def test_committed_artifact_has_a_read_only_verification_path():
    before = Path("tests/shakebench_phase_02r_probe.json").read_bytes()
    summary = verify_artifact("tests/shakebench_phase_02r_probe.json")
    assert summary["passed"]
    assert summary["integrity_valid"] is True
    assert summary["physics_gates_passed"] is True
    assert summary["phase03_handoff"] == "PASS"
    assert summary["schema_id"] == "shakebench.phase02r.conformance_matrix"
    assert summary["schema_version"] == 3
    assert Path("tests/shakebench_phase_02r_probe.json").read_bytes() == before


def test_controlled_artifact_requires_explicit_update_reason():
    assert probe_main(["--output", "tests/shakebench_phase_02r_probe.json"]) == 2
    assert probe_main(["--output", "tests/shakebench_phase_02r_probe.json", "--update"]) == 2
    assert probe_main(["--verify-artifact", "tests/shakebench_phase_02r_probe.json"]) == 0


def test_role_processor_builds_auditable_topology_without_task_names():
    handles = {"robot_base": "robot_mount", "probe": "payload_root"}
    config = DeckDriverConfig(physics_timestep_s=0.0001, weld_solref=(0.0002, 1.0))
    xml, config = _compile_deck_xml(handles, config=config)
    audit = audit_deck_xml(xml, config, handles)
    assert audit.parent_graph[config.driver_body_name] is None
    assert audit.parent_graph[config.deck_body_name] is None
    assert audit.parent_graph["robot_mount"] == config.deck_body_name
    assert audit.parent_graph["payload_root"] == config.deck_body_name
    assert audit.driver_geom_names == ()
    assert audit.weld_body1 == config.driver_body_name
    assert audit.weld_body2 == config.deck_body_name

    model = mujoco.MjModel.from_xml_string(xml)
    compiled = audit_compiled_deck_model(model, config, handles)
    assert compiled["parent_graph"]["robot_mount"] == config.deck_body_name
    assert compiled["deck_mass_kg"] == pytest.approx(config.deck_mass_kg)
    assert compiled["deck_inertia_kg_m2"] == pytest.approx(list(config.deck_inertia_kg_m2))
    assert compiled["driver_geom_names"] == []


def test_public_audit_mappings_are_immutable_but_to_dict_returns_copies():
    handles = {"robot_base": "robot_mount"}
    body_handles = DeckBodyHandles(handles)
    with pytest.raises(TypeError):
        body_handles.roles["robot_base"] = "other"
    assert body_handles.to_dict() == handles

    xml, config = _compile_deck_xml(handles)
    audit = audit_deck_xml(xml, config, handles)
    with pytest.raises(TypeError):
        audit.parent_graph[config.deck_body_name] = "other"
    with pytest.raises(TypeError):
        audit.role_body_names["robot_base"] = "other"
    copy = audit.to_dict()
    copy["parent_graph"][config.deck_body_name] = "other"
    copy["role_body_names"]["robot_base"] = "other"
    assert audit.parent_graph[config.deck_body_name] is None
    assert audit.role_body_names["robot_base"] == "robot_mount"


def test_role_processor_fails_closed_for_missing_and_duplicate_application():
    config = DeckDriverConfig()
    with pytest.raises(DeckXMLProcessingError, match="missing body"):
        ShakeBenchDeckXMLProcessor(config, {"robot_base": "not_in_xml"})(_empty_xml())

    xml = ShakeBenchDeckXMLProcessor(config)(_empty_xml())
    with pytest.raises(DeckXMLProcessingError, match="applied more than once"):
        ShakeBenchDeckXMLProcessor(config)(xml)
    with pytest.raises(DeckXMLProcessingError, match="applied more than once"):
        ShakeBenchDeckXMLProcessor(DeckDriverConfig(driver_body_name="other_driver", deck_body_name="other_deck"))(xml)


def test_equality_parameters_have_one_canonical_name_with_boundary_compatibility():
    canonical = DeckDriverConfig(eq_solref=(0.001, 1.0), eq_solimp=(0.85, 0.95, 0.001, 0.5, 2.0))
    legacy = DeckDriverConfig(weld_solref=(0.001, 1.0), weld_solimp=(0.85, 0.95, 0.001, 0.5, 2.0))
    assert canonical.eq_solref == legacy.eq_solref
    assert canonical.eq_solimp == legacy.eq_solimp
    assert canonical.to_dict()["eq_solref"] == [0.001, 1.0]
    assert canonical.to_dict()["eq_solimp"] == [0.85, 0.95, 0.001, 0.5, 2.0]
    assert "weld_solref" not in canonical.to_dict()
    with pytest.raises(DeckDriverError, match="different values"):
        DeckDriverConfig(eq_solref=(0.001, 1.0), weld_solref=(0.002, 1.0))


def test_role_processor_rejects_nested_free_joint_that_cannot_compile():
    xml = """<mujoco><asset/><worldbody>
      <body name="invalid_probe"><freejoint name="probe_free"/>
        <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
      </body>
    </worldbody></mujoco>"""
    with pytest.raises(DeckXMLProcessingError, match="freejoint"):
        ShakeBenchDeckXMLProcessor(DeckDriverConfig(), {"probe": "invalid_probe"})(xml)


def test_pre_step_timing_has_no_hidden_one_step_lead():
    dt = 0.0001
    trace, _ = run_trace(dt=dt, duration_s=0.1)
    np.testing.assert_allclose(
        trace.application_timestamps_s,
        trace.command_timestamps_s,
        rtol=0.0,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        trace.sample_timestamps_s,
        trace.command_timestamps_s + dt,
        rtol=0.0,
        atol=2e-14,
    )
    assert trace.command_pose.shape == (trace.sample_timestamps_s.size, 7)
    assert trace.actual_twist.shape == (trace.sample_timestamps_s.size, 6)
    assert trace.actual_acceleration.shape == (trace.sample_timestamps_s.size, 6)
    assert trace.deck_tracking_pose_error.shape == (trace.sample_timestamps_s.size, 6)
    assert trace.weld_constraint_force_raw.shape == (trace.sample_timestamps_s.size, 6)
    assert trace.warning_number_delta.shape[0] == trace.sample_timestamps_s.size


def test_tracking_error_uses_the_same_post_step_sample_time_as_actual_state():
    dt = 0.0002
    frequency_hz = 5.0
    amplitude = 0.001
    trace, _ = run_trace(dt=dt, duration_s=0.12, trajectory=_sine_command(0, frequency_hz, amplitude))
    expected_command_at_sample = amplitude * np.sin(2.0 * math.pi * frequency_hz * trace.sample_timestamps_s)
    actual_at_sample = _actual_coordinates(trace)[:, 0]
    np.testing.assert_allclose(
        trace.deck_tracking_pose_error[:, 0],
        actual_at_sample - expected_command_at_sample,
        rtol=0.0,
        atol=1e-12,
    )


@pytest.mark.parametrize("lite_physics", (True, False))
def test_driver_post_integration_refresh_keeps_state_and_constraint_buffers_current(lite_physics):
    dt = 0.0002
    config = DeckDriverConfig(eq_solref=(2.0 * dt, 1.0), physics_timestep_s=dt)
    driver = DeckDriver(trajectory=_sine_command(0, 5.0, 0.001), config=config)
    env = DeckProbeEnv(
        build_probe_xml(model_timestep=dt),
        driver,
        model_timestep=dt,
        control_freq=20,
        horizon=20,
        lite_physics=lite_physics,
    )
    try:
        env.reset()
        driver.reset_trace()
        env.step(np.zeros(0))
        trace = driver.trace
        raw_model = env.sim.model._model
        raw_data = env.sim.data._data
        deck_id = mujoco.mj_name2id(raw_model, mujoco.mjtObj.mjOBJ_BODY, config.deck_body_name)
        joint_id = mujoco.mj_name2id(raw_model, mujoco.mjtObj.mjOBJ_JOINT, config.deck_freejoint_name)
        qpos_adr = int(raw_model.jnt_qposadr[joint_id])
        weld_id = mujoco.mj_name2id(raw_model, mujoco.mjtObj.mjOBJ_EQUALITY, config.weld_name)
        rows = np.flatnonzero(
            (raw_data.efc_type == int(mujoco.mjtConstraint.mjCNSTR_EQUALITY))
            & (raw_data.efc_id == int(weld_id))
        )
        assert trace.sample_timestamps_s[-1] == pytest.approx(float(raw_data.time), abs=1e-14)
        np.testing.assert_allclose(trace.actual_pose[-1, :3], raw_data.xpos[deck_id], rtol=0.0, atol=1e-14)
        np.testing.assert_allclose(
            trace.actual_pose[-1, 3:], raw_data.xquat[deck_id], rtol=0.0, atol=1e-14
        )
        np.testing.assert_allclose(
            raw_data.qpos[qpos_adr : qpos_adr + 3], raw_data.xpos[deck_id], rtol=0.0, atol=1e-14
        )
        np.testing.assert_allclose(trace.weld_constraint_residual_raw[-1], raw_data.efc_pos[rows])
        np.testing.assert_allclose(trace.weld_constraint_force_raw[-1], raw_data.efc_force[rows])
    finally:
        env.close()


def test_disabling_the_refresh_fixture_reproduces_stale_qpos_and_derived_pose():
    dt = 0.0002
    config = DeckDriverConfig(eq_solref=(2.0 * dt, 1.0), physics_timestep_s=dt)
    driver = DeckDriver(trajectory=_sine_command(0, 5.0, 0.001), config=config)
    env = DeckProbeEnv(
        build_probe_xml(model_timestep=dt),
        driver,
        model_timestep=dt,
        control_freq=20,
        horizon=20,
        lite_physics=True,
    )
    try:
        env.reset()
        env._post_integration_refresh_requested = False
        env._post_physics_step_hooks = []
        driver.reset_trace()
        env.step(np.zeros(0))
        raw_model = env.sim.model._model
        raw_data = env.sim.data._data
        deck_id = mujoco.mj_name2id(raw_model, mujoco.mjtObj.mjOBJ_BODY, config.deck_body_name)
        joint_id = mujoco.mj_name2id(raw_model, mujoco.mjtObj.mjOBJ_JOINT, config.deck_freejoint_name)
        qpos_adr = int(raw_model.jnt_qposadr[joint_id])
        assert np.max(np.abs(raw_data.qpos[qpos_adr : qpos_adr + 3] - raw_data.xpos[deck_id])) > 1e-12
    finally:
        env.close()


def _fit_translation_acceleration(trace):
    time_s = trace.sample_time_s
    values = trace.actual_acceleration[:, 0]
    return _fit_fixture_sine(time_s[time_s >= 0.1], values[time_s >= 0.1], 5.0)[0]


def test_old_mocap_target_refresh_is_a_right_limit_negative_case():
    dt = 0.0002
    config = DeckDriverConfig(eq_solref=(2.0 * dt, 1.0), physics_timestep_s=dt)
    driver = DeckDriver(trajectory=_sine_command(0, 5.0, 0.001), config=config)
    env = DeckProbeEnv(
        build_probe_xml(model_timestep=dt),
        driver,
        model_timestep=dt,
        control_freq=20,
        horizon=20,
        lite_physics=True,
    )
    try:
        env.reset()
        # Keep the full forward refresh but remove the driver's right-limit
        # mocap write, reproducing the old q(t) input at state t+dt.
        env._post_integration_refresh_hooks = []
        accelerations = []
        sample_times = []

        def sample_old_target_state(sample_time_s, policy_step=False):
            _, _, _, acceleration = driver._actual_state(env.sim.model._model, env.sim.data._data)
            sample_times.append(float(sample_time_s))
            accelerations.append(float(acceleration[0]))

        env._post_physics_step_hooks = [sample_old_target_state]
        driver.reset_trace()
        for _ in range(10):
            env.step(np.zeros(0))
        sample_times = np.asarray(sample_times)
        accelerations = np.asarray(accelerations)
        mask = sample_times >= 0.1
        old_target_amplitude = _fit_fixture_sine(sample_times[mask], accelerations[mask], 5.0)[0]
        assert old_target_amplitude > 10.0
    finally:
        env.close()


def test_right_limit_target_refresh_matches_authored_qdd_at_sample_time():
    dt = 0.0002
    config = DeckDriverConfig(eq_solref=(2.0 * dt, 1.0), physics_timestep_s=dt)
    driver = DeckDriver(trajectory=_sine_command(0, 5.0, 0.001), config=config)
    env = DeckProbeEnv(
        build_probe_xml(model_timestep=dt),
        driver,
        model_timestep=dt,
        control_freq=20,
        horizon=20,
        lite_physics=True,
    )
    try:
        env.reset()
        driver.reset_trace()
        for _ in range(10):
            env.step(np.zeros(0))
        trace = driver.trace
        expected_amplitude = 0.001 * (2.0 * math.pi * 5.0) ** 2
        actual_amplitude = _fit_translation_acceleration(trace)
        assert actual_amplitude == pytest.approx(expected_amplitude, rel=0.01)
        np.testing.assert_allclose(
            trace.sample_target_acceleration[:, 0],
            -0.001 * (2.0 * math.pi * 5.0) ** 2 * np.sin(2.0 * math.pi * 5.0 * trace.sample_time_s),
            rtol=0.0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            env.sim.data.mocap_pos[driver._driver_mocap_id],
            trace.sample_target_pose[-1, :3],
            rtol=0.0,
            atol=1e-14,
        )
    finally:
        env.close()


@pytest.mark.parametrize("lite_physics", (True, False))
def test_refresh_does_not_integrate_a_second_time(lite_physics):
    dt = 0.0002
    config = DeckDriverConfig(eq_solref=(2.0 * dt, 1.0), physics_timestep_s=dt)
    driver = DeckDriver(trajectory=_sine_command(0, 5.0, 0.001), config=config)
    env = DeckProbeEnv(
        build_probe_xml(model_timestep=dt),
        driver,
        model_timestep=dt,
        control_freq=20,
        horizon=20,
        lite_physics=lite_physics,
    )
    state_before_refresh = {}
    state_after_refresh = {}

    def capture_before_forward(sample_time_s, policy_step=False):
        state_before_refresh["qpos"] = np.array(env.sim.data.qpos, copy=True)
        state_before_refresh["qvel"] = np.array(env.sim.data.qvel, copy=True)
        state_before_refresh["time"] = float(env.sim.data.time)

    def capture_after_forward(sample_time_s, policy_step=False):
        state_after_refresh["qpos"] = np.array(env.sim.data.qpos, copy=True)
        state_after_refresh["qvel"] = np.array(env.sim.data.qvel, copy=True)
        state_after_refresh["time"] = float(env.sim.data.time)

    try:
        env.reset()
        env.add_post_integration_refresh_hook(capture_before_forward)
        env.add_post_physics_step_hook(capture_after_forward)
        env.step(np.zeros(0))
        np.testing.assert_array_equal(state_before_refresh["qpos"], state_after_refresh["qpos"])
        np.testing.assert_array_equal(state_before_refresh["qvel"], state_after_refresh["qvel"])
        assert state_before_refresh["time"] == state_after_refresh["time"]
    finally:
        env.close()


def test_trace_records_both_mocap_limits_and_explicit_sample_time_contract():
    dt = 0.0002
    config = DeckDriverConfig(eq_solref=(2.0 * dt, 1.0), physics_timestep_s=dt)
    driver = DeckDriver(trajectory=_sine_command(0, 5.0, 0.001), config=config)
    env = DeckProbeEnv(
        build_probe_xml(model_timestep=dt),
        driver,
        model_timestep=dt,
        control_freq=20,
        horizon=20,
    )
    try:
        env.reset()
        driver.reset_trace()
        env.step(np.zeros(0))
        trace = driver.trace
        np.testing.assert_array_equal(trace.integration_target_time_s, trace.command_timestamps_s)
        np.testing.assert_array_equal(trace.integration_application_time_s, trace.application_timestamps_s)
        np.testing.assert_array_equal(trace.sample_time_s, trace.sample_timestamps_s)
        np.testing.assert_allclose(trace.sample_target_time_s, trace.sample_time_s, rtol=0.0, atol=1e-14)
        np.testing.assert_allclose(trace.sample_application_time_s, trace.sample_time_s, rtol=0.0, atol=1e-14)
        np.testing.assert_allclose(
            trace.integration_target_time_s[1:],
            trace.sample_target_time_s[:-1],
            rtol=0.0,
            atol=1e-14,
        )
        np.testing.assert_allclose(trace.command_pose[1:], trace.sample_target_pose[:-1], rtol=0.0, atol=1e-14)
        assert trace.sample_target_pose.shape == trace.actual_pose.shape
        assert trace.sample_target_twist.shape == trace.actual_twist.shape
        assert trace.sample_target_acceleration.shape == trace.actual_acceleration.shape
        contract = trace.to_dict()["field_contract"]
        assert "integration_target_time_s" in contract
        assert "sample_target_time_s" in contract
        assert "sample_application_time_s" in contract
    finally:
        env.close()


def test_upstream_environment_without_driver_does_not_request_post_integration_refresh():
    env = Lift(
        robots="Panda",
        use_camera_obs=False,
        has_offscreen_renderer=False,
        renderer="mujoco",
        initialization_noise=None,
    )
    try:
        assert env._post_integration_refresh_requested is False
        assert env._post_integration_refresh_hooks == []
    finally:
        env.close()


def test_post_refresh_provisional_gamma_uses_the_right_limit_acceleration():
    # The right-limit target is written before refresh, so current-state
    # acceleration is compared to qdd at the same sample target time.
    dt = 0.00005
    frequency = 8.87
    amplitude = 0.001
    trace, _ = run_trace(dt=dt, duration_s=0.5, trajectory=_sine_command(0, frequency, amplitude))
    conformance = sine_conformance(
        trace,
        axis_index=0,
        frequency_hz=frequency,
        amplitude=amplitude,
        discard_s=0.2,
    )
    assert conformance["amplitude_relative_error"] <= 0.01
    assert abs(conformance["phase_error_deg"]) <= 1.0
    gamma = gamma_conformance(
        trace,
        axis_index=0,
        frequency_hz=frequency,
        displacement_amplitude=amplitude,
        discard_s=0.2,
    )
    assert gamma["actual_gamma"] == pytest.approx(gamma["command_gamma"], rel=0.01)
    assert gamma["relative_error_abs"] <= 0.01
    assert abs(gamma["phase_error_deg"]) <= 1.0


def test_six_axis_spectrum_records_actual_state_and_weld_diagnostics():
    dt = 0.00005
    program = build_excitation_program(seed=17, level_scale=0.15)
    trace, _ = run_trace(dt=dt, duration_s=0.6, trajectory=program)
    conformance = spectrum_conformance(trace, program, discard_s=0.5)
    assert set(conformance["axes"]) == set(("tx", "ty", "tz", "rx", "ry", "rz"))
    assert np.isfinite(_actual_coordinates(trace)).all()
    assert np.isfinite(trace.actual_acceleration).all()
    assert np.isfinite(trace.weld_constraint_residual_raw).all()
    assert np.isfinite(trace.weld_constraint_force_raw).all()
    assert conformance["warning_count"] == 0


@pytest.mark.parametrize("axis_index", range(6))
def test_zero_and_each_single_axis_are_instrumented(axis_index):
    amplitude = 0.001 if axis_index < 3 else 0.0005
    trace, _ = run_trace(
        dt=0.0002,
        duration_s=0.12,
        trajectory=_sine_command(axis_index, 5.0, amplitude),
    )
    assert trace.sample_timestamps_s.size > 0
    assert np.max(np.abs(trace.command_twist[:, axis_index])) > 0.0
    assert np.allclose(trace.command_twist[:, [index for index in range(6) if index != axis_index]], 0.0)


@pytest.mark.parametrize("level_scale", (0.0, 0.15, 0.30, 0.50))
def test_multiple_gamma_scales_keep_command_and_actual_layers(level_scale):
    program = build_excitation_program(seed=4, level_scale=level_scale)
    trace, _ = run_trace(dt=0.0002, duration_s=0.12, trajectory=program)
    expected = program.evaluate(trace.sample_timestamps_s).q
    actual = _actual_coordinates(trace)
    assert expected.shape == actual.shape
    assert trace.command_timestamps_s.size == trace.sample_timestamps_s.size
    if level_scale == 0.0:
        assert np.max(np.abs(actual)) < 1e-10


@pytest.mark.parametrize("representative_load", (False, True))
def test_empty_and_representative_load_paths_compile_and_run(representative_load):
    from robosuite.scripts.shakebench_probe_deck_driver import run_trace as run_probe_trace

    trace, audit = run_probe_trace(
        dt=0.0002,
        duration_s=0.12,
        trajectory=_sine_command(0, 5.0, 0.001),
        representative_load=representative_load,
    )
    assert trace.sample_timestamps_s.size > 0
    if representative_load:
        assert audit["parent_graph"]["worktable_reference_proxy"] == "deck"
    else:
        assert "worktable_reference_proxy" not in audit["parent_graph"]


def test_load_proxy_covers_canonical_worktable_mass_and_inertia_without_collision():
    dt = 0.0002
    config = DeckDriverConfig(eq_solref=(2.0 * dt, 1.0), physics_timestep_s=dt)
    driver = DeckDriver(config=config, body_handles={"worktable_reference": "worktable_reference_proxy"})
    xml = driver.processor(build_probe_xml(model_timestep=dt, representative_load=True))
    model = mujoco.MjModel.from_xml_string(xml)
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "worktable_reference_proxy")
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "worktable_reference_proxy_geom")
    assert model.body_mass[body_id] == pytest.approx(32.0)
    assert model.body_inertia[body_id] == pytest.approx([0.9696, 1.1363, 2.0867])
    assert model.body_ipos[body_id] == pytest.approx([0.0, 0.0, 0.0])
    assert model.geom_contype[geom_id] == 0
    assert model.geom_conaffinity[geom_id] == 0


@pytest.mark.parametrize(
    "dt,tau,solimp",
    (
        (0.0001, 0.0002, (0.9, 0.95, 0.001, 0.5, 2.0)),
        (0.0002, 0.0004, (0.85, 0.95, 0.001, 0.5, 2.0)),
        (0.00005, 0.0001, (0.9, 0.98, 0.001, 0.5, 2.0)),
    ),
)
def test_dt_solref_solimp_sensitivity_is_explicit(dt, tau, solimp):
    trace, audit = run_trace(
        dt=dt,
        duration_s=0.1,
        trajectory=_sine_command(0, 5.0, 0.001),
        tau=tau,
        solimp=solimp,
    )
    assert audit["eq_solref"] == pytest.approx([tau, 1.0])
    assert audit["eq_solimp"] == pytest.approx(list(solimp))
    assert trace.solver_niter.shape[0] == trace.sample_timestamps_s.size


def _negative_mocap_parent_fixture() -> str:
    return """<mujoco model="phase02_negative_mocap_parent">
  <option timestep="0.0002" gravity="0 0 0" iterations="100" tolerance="1e-12"/>
  <worldbody>
    <body name="mocap_parent" mocap="true" pos="0 0 0">
      <site name="parent_site" pos="0 0 0"/>
      <body name="compliant_child" pos="0 0 0">
        <joint name="child_slide" type="slide" axis="1 0 0"
               stiffness="986.9604401089358" damping="6.283185307179586"/>
        <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
        <site name="child_site" pos="0 0 0"/>
      </body>
    </body>
  </worldbody>
</mujoco>"""


def test_negative_mocap_parent_fixture_cannot_supply_dynamic_base_excitation():
    model = mujoco.MjModel.from_xml_string(_negative_mocap_parent_fixture())
    data = mujoco.MjData(model)
    parent_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "mocap_parent")
    child_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "compliant_child")
    dt = model.opt.timestep
    frequency_hz = 5.0
    omega = 2.0 * math.pi * frequency_hz
    amplitude = 0.001
    parent = []
    child = []
    times = []
    for _ in range(int(1.2 / dt)):
        time_s = float(data.time)
        command = amplitude * math.sin(omega * time_s)
        data.mocap_pos[0, 0] = command
        mujoco.mj_step(model, data)
        times.append(float(data.time))
        parent.append(float(data.xpos[parent_id, 0]))
        child.append(float(data.xpos[child_id, 0]))

    times = np.asarray(times)
    parent = np.asarray(parent)
    child = np.asarray(child)
    mask = times >= 0.6
    parent_amplitude, _ = _fit_fixture_sine(times[mask], parent[mask], frequency_hz)
    child_amplitude, _ = _fit_fixture_sine(times[mask], child[mask], frequency_hz)
    relative_amplitude, _ = _fit_fixture_sine(times[mask], (child - parent)[mask], frequency_hz)
    transmissibility = child_amplitude / parent_amplitude
    analytic = math.sqrt(1.0 + (2.0 * 0.1) ** 2) / (2.0 * 0.1)

    assert relative_amplitude == pytest.approx(0.0, abs=1e-12)
    assert transmissibility == pytest.approx(1.0, abs=1e-12)
    assert analytic == pytest.approx(5.0990195135927845)
    assert abs(transmissibility - analytic) > 4.0


def _screening_candidate(candidate_id, dt_s, phase_error_deg, *, passed=True):
    line_fits = [
        {
            "frequency_hz": 5.0 + 0.01 * index,
            "amplitude_relative_error": 0.001,
            "phase_error_deg": phase_error_deg,
            "fit_residual_rms": 1e-8,
            "condition_number": 10.0,
        }
        for index in range(64)
    ]
    return {
        "candidate_id": candidate_id,
        "dt_s": dt_s,
        "frequency_sampling_passed": True,
        "solref_relation_passed": True,
        "screening": {
            "zero": {"passed": True},
            "single_axis": {axis: {"passed": True} for axis in ("tx", "ty", "tz", "rx", "ry", "rz")},
            "authored_spectrum": {
                "passed": passed,
                "fit_window_meets_resolution": True,
                "line_fits": line_fits,
            },
        },
    }


def test_r4_candidate_selection_is_metric_driven_and_deterministic():
    candidates = [
        _screening_candidate("candidate_b", 0.0002, 0.4),
        _screening_candidate("candidate_a", 0.0001, 0.9),
        _screening_candidate("candidate_c", 0.0002, 0.8),
    ]
    decision = select_candidate(candidates, thresholds=Phase02R4Thresholds())
    assert decision["selected_candidate_id"] == "candidate_b"
    assert decision["selection_rule"]["primary"] == "maximum_dt_among_screening_passes"

    candidates[0]["screening"]["authored_spectrum"]["line_fits"][0]["phase_error_deg"] = 1.1
    decision_after_metric_change = select_candidate(candidates, thresholds=Phase02R4Thresholds())
    assert decision_after_metric_change["selected_candidate_id"] == "candidate_c"

    candidates[2]["screening"]["authored_spectrum"]["line_fits"][0]["phase_error_deg"] = 1.1
    decision_after_rejection = select_candidate(candidates, thresholds=Phase02R4Thresholds())
    assert decision_after_rejection["selected_candidate_id"] == "candidate_a"
    assert decision_after_rejection["rejections"]["candidate_b"]
    assert decision_after_rejection["rejections"]["candidate_c"]


def test_r4_short_screening_stops_before_confirmatory_matrix():
    result = run_probe_suite(dt=0.0005, duration_s=0.1, spectrum_duration_s=2.0)
    selection = result["provenance"]["candidate_selection"]
    assert selection["selection_status"] == "blocked_no_candidate"
    assert selection["confirmatory_started"] is False
    assert result["confirmatory"]["status"] == "not_run_no_candidate"
    assert result["load_matrix"] == []
    assert result["phase03_handoff"] == "BLOCKED"
    assert all(candidate["screening"]["status"] == "measured" for candidate in result["provenance"]["candidate_grid"])


def test_r4_panda_base_inclusive_fixture_uses_compiled_robot_subtree():
    load_case = LOAD_CASE_PANDA_PLUS_WORKTABLE_REFERENCE_PROXY
    dt = 0.0002
    config = DeckDriverConfig(eq_solref=(2.0 * dt, 1.0), physics_timestep_s=dt)
    driver = DeckDriver(
        config=config,
        body_handles={
            "panda_base": "robot0_base",
            "worktable_reference": "worktable_reference_proxy",
        },
    )
    compiled_xml = driver.processor(build_probe_xml(model_timestep=dt, load_case=load_case))
    model = mujoco.MjModel.from_xml_string(compiled_xml)
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot0_base")
    deck_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, config.deck_body_name)
    assert model.body_parentid[base_id] == deck_id

    trace, audit = run_trace(
        dt=dt,
        duration_s=0.02,
        trajectory=_sine_command(0, 5.0, 0.001),
        load_case=load_case,
    )
    assert trace.sample_timestamps_s.size > 0
    provenance = audit["load_provenance"]
    assert provenance["panda_base_included"] is True
    assert provenance["compiled_model_assertions"]["passed"] is True
    assert len(provenance["attached_body_names"]) >= 10
    assert provenance["panda_subtree_mass_kg"] > 0.0
    assert provenance["panda_subtree_body_masses_kg"]
    assert provenance["panda_subtree_body_inertia_kg_m2"]
    assert provenance["initial_joint_state"]
    assert provenance["controller_started"] is False
    assert provenance["task_environment"] is False


def test_r4_gate_summary_requires_selection_and_panda_confirmatory_evidence():
    result = run_probe_suite(dt=0.0005, duration_s=0.1, spectrum_duration_s=2.0)
    gates = compute_gate_summary(result)
    assert gates["candidate_screening_gate"]["passed"] is True
    assert gates["candidate_selection_gate"]["passed"] is False
    assert gates["panda_base_compiled_audit_gate"]["passed"] is False
    assert gates["overall_phase02r_status"] == "partial_or_blocked"


def test_r4_metric_mutation_changes_recomputed_selection_and_handoff():
    payload = json.loads(Path("tests/shakebench_phase_02r_probe.json").read_text(encoding="utf-8"))
    mutated = copy.deepcopy(payload)
    nominal = next(
        candidate
        for candidate in mutated["provenance"]["candidate_grid"]
        if candidate["candidate_id"] == "nominal"
    )
    nominal["screening"]["authored_spectrum"]["conformance"]["axes"]["tx"]["line_fits"][0]["phase_error_deg"] = 1.1
    decision = select_candidate(mutated["provenance"]["candidate_grid"], thresholds=Phase02R4Thresholds())
    assert decision["selected_candidate_id"] == "fine"
    gates = compute_gate_summary(mutated)
    assert gates["candidate_selection_gate"]["passed"] is False
    assert gates["overall_phase02r_status"] == "partial_or_blocked"


def test_r4_controlled_artifact_matches_independent_manifest():
    manifest = json.loads(Path("tests/shakebench_phase_02r_probe_manifest.json").read_text(encoding="utf-8"))
    summary = verify_artifact(manifest["artifact_path"])
    assert summary["file_sha256"] == manifest["file_sha256"]
    assert summary["payload_sha256"] == manifest["payload_sha256"]
    assert summary["schema_id"] == manifest["schema_id"]
    assert summary["schema_version"] == manifest["schema_version"]
    assert summary["physics_gates_passed"] is manifest["physics_gates_passed"]
    assert summary["phase03_handoff"] == manifest["phase03_handoff"]
    payload = json.loads(Path(manifest["artifact_path"]).read_text(encoding="utf-8"))
    assert payload["provenance"]["selected_provisional_candidate_id"] == manifest["selected_candidate_id"]
    assert len(payload["load_matrix"]) == manifest["confirmatory_matrix_records"]
    assert all(
        sum(len(axis["line_fits"]) for axis in record["spectrum"]["conformance"]["axes"].values())
        == manifest["spectrum_lines_per_record"]
        for record in payload["load_matrix"]
    )


def test_r4_artifact_update_reason_and_previous_hash_are_authenticated(tmp_path):
    payload = json.loads(Path("tests/shakebench_phase_02r_probe.json").read_text(encoding="utf-8"))
    payload["artifact_update"]["reason"] = "tampered"
    candidate = tmp_path / "tampered_probe.json"
    candidate.write_text(json.dumps(payload), encoding="utf-8")
    summary = verify_artifact(candidate)
    assert summary["integrity_valid"] is False
    assert "artifact payload integrity hash is missing or does not verify" in summary["errors"]
    assert Path("tests/shakebench_phase_02r_probe.json").read_bytes() != candidate.read_bytes()


def _fit_fixture_sine(time_s, values, frequency_hz):
    omega = 2.0 * math.pi * frequency_hz
    design = np.column_stack((np.sin(omega * time_s), np.cos(omega * time_s), np.ones(time_s.size)))
    coefficient = np.linalg.lstsq(design, values, rcond=None)[0]
    return float(np.hypot(coefficient[0], coefficient[1])), float(math.atan2(coefficient[1], coefficient[0]))
