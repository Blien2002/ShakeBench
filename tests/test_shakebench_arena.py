"""Physics and XML contract tests for the Phase 03 ShakeBench arena."""

from __future__ import annotations

import numpy as np
import pytest
import mujoco

from robosuite.models.arenas import ShakeBenchArena, TableArena
from robosuite.scripts.shakebench_probe_deck_driver import DeckProbeEnv
from robosuite.utils.shakebench_deck import (
    DeckDriver,
    DeckDriverConfig,
    audit_compiled_deck_model,
    audit_deck_xml,
)
from robosuite.utils.mjcf_utils import xml_path_completion
from robosuite.utils.shakebench_isolator import AXES, DEFAULT_ISOLATOR_CONFIG, derive_isolator_parameters


def _deck_config() -> DeckDriverConfig:
    dt = 0.0002
    return DeckDriverConfig(physics_timestep_s=dt, eq_solref=(2.0 * dt, 0.5))


def _compile_deck_arena(arena: ShakeBenchArena | None = None):
    arena = arena if arena is not None else ShakeBenchArena()
    config = _deck_config()
    xml = arena.process_deck_xml(config=config)
    model = mujoco.MjModel.from_xml_string(xml)
    return arena, config, xml, model


def test_shakebench_arena_is_exported_and_compiles_with_canonical_inertial():
    arena = ShakeBenchArena()
    assert isinstance(arena, ShakeBenchArena)
    assert isinstance(TableArena(), TableArena)  # stock TableArena remains available
    assert arena.table_top_abs.tolist() == pytest.approx([0.0, 0.0, 0.8])
    assert "shakebench_phenolic_bench_dark_1k.png" in arena.get_xml()

    model = mujoco.MjModel.from_xml_string(arena.get_xml())
    audit = arena.audit_compiled_model(model)
    body_id = model.body(arena.worktable_body_name).id
    assert audit["mass_kg"] == pytest.approx(32.0, rel=0.0, abs=1e-12)
    assert audit["com_m"] == pytest.approx([0.0, 0.0, 0.0], rel=0.0, abs=1e-12)
    assert audit["inertia_kg_m2"] == pytest.approx([0.9696, 1.1363, 2.0867], rel=0.0, abs=1e-12)
    assert model.body_subtreemass[body_id] == pytest.approx(32.0, rel=0.0, abs=1e-12)
    assert set(audit["joints"]) == {"tx", "ty", "tz", "rx", "ry", "rz"}
    assert set(audit["visual_geoms"]) == set(arena.visual_geom_names)
    assert all(item["contype"] == item["conaffinity"] == 0 for item in audit["visual_geoms"].values())
    assert audit["collision_geom"]["contype"] != 0
    assert audit["collision_geom"]["conaffinity"] != 0


def test_raw_arena_xml_defaults_equal_python_defaults_without_configurator():
    """The asset itself must be valid before Arena mutates its XML tree."""

    model = mujoco.MjModel.from_xml_path(xml_path_completion("arenas/shakebench_arena.xml"))
    expected = derive_isolator_parameters(DEFAULT_ISOLATOR_CONFIG)
    body_id = model.body("worktable").id
    assert model.body_mass[body_id] == pytest.approx(DEFAULT_ISOLATOR_CONFIG.mass_kg, rel=0.0, abs=1e-12)
    np.testing.assert_array_equal(model.body_ipos[body_id], np.zeros(3))
    np.testing.assert_array_equal(model.body_inertia[body_id], DEFAULT_ISOLATOR_CONFIG.inertia_kg_m2)
    for index, axis in enumerate(AXES):
        joint_id = model.joint(f"isolator_{axis}").id
        dof_id = model.jnt_dofadr[joint_id]
        qpos_id = model.jnt_qposadr[joint_id]
        np.testing.assert_array_equal(model.jnt_pos[joint_id], np.zeros(3))
        np.testing.assert_array_equal(model.jnt_axis[joint_id], np.eye(3)[index % 3])
        assert model.jnt_stiffness[joint_id] == pytest.approx(expected.stiffness[index], rel=0.0, abs=1e-12)
        assert model.dof_damping[dof_id] == pytest.approx(expected.damping[index], rel=0.0, abs=1e-12)
        assert model.qpos_spring[qpos_id] == pytest.approx(expected.springref[index], rel=0.0, abs=1e-12)
        limit = DEFAULT_ISOLATOR_CONFIG.limits[index]
        assert model.jnt_limited[joint_id]
        np.testing.assert_array_equal(model.jnt_range[joint_id], (-limit, limit))


def test_phase02_processor_reparents_worktable_by_explicit_role():
    arena, config, xml, model = _compile_deck_arena()
    xml_audit = audit_deck_xml(xml, config, arena.deck_body_handles)
    compiled_audit = audit_compiled_deck_model(model, config, arena.deck_body_handles)

    assert xml_audit.parent_graph[arena.worktable_body_name] == config.deck_body_name
    assert compiled_audit["parent_graph"][arena.worktable_body_name] == config.deck_body_name
    assert compiled_audit["deck_mass_kg"] == pytest.approx(config.deck_mass_kg)
    assert compiled_audit["driver_geom_names"] == []
    assert arena.audit_compiled_model(model)["mass_kg"] == pytest.approx(32.0)


def test_visual_layer_toggle_preserves_compiled_physics_and_deck_trace():
    visible = ShakeBenchArena(visual=True)
    hidden = ShakeBenchArena(visual=False)
    visible_model = mujoco.MjModel.from_xml_string(visible.process_deck_xml(config=_deck_config()))
    hidden_model = mujoco.MjModel.from_xml_string(hidden.process_deck_xml(config=_deck_config()))

    for field in (
        "body_mass",
        "body_ipos",
        "body_inertia",
        "jnt_type",
        "jnt_axis",
        "jnt_stiffness",
        "dof_damping",
        "qpos_spring",
        "jnt_range",
        "jnt_limited",
        "geom_contype",
        "geom_conaffinity",
    ):
        np.testing.assert_array_equal(getattr(visible_model, field), getattr(hidden_model, field))
    assert not np.array_equal(visible_model.geom_rgba, hidden_model.geom_rgba)

    def trace_for(arena):
        driver = DeckDriver(config=_deck_config(), body_handles=arena.deck_body_handles)
        env = DeckProbeEnv(
            arena.get_xml(),
            driver,
            model_timestep=0.0002,
            control_freq=20.0,
            horizon=2,
        )
        try:
            env.reset()
            env.step(np.zeros(0))
            return driver.trace
        finally:
            env.close()

    visible_trace = trace_for(visible)
    hidden_trace = trace_for(hidden)
    for field in (
        "integration_target_time_s",
        "sample_time_s",
        "actual_pose",
        "actual_twist",
        "actual_acceleration",
        "weld_constraint_residual_raw",
        "weld_constraint_force_raw",
    ):
        np.testing.assert_array_equal(getattr(visible_trace, field), getattr(hidden_trace, field))


def test_optional_target_container_is_rigid_without_hidden_mass_or_dof():
    arena = ShakeBenchArena(include_target_container=True)
    model = mujoco.MjModel.from_xml_string(arena.get_xml())
    audit = arena.audit_compiled_model(model)
    assert arena.target_container_added
    assert len([name for name in arena.target_container_geom_names.values() if not name.endswith("_visual")]) == 5
    assert model.body(arena.worktable_body_name).id >= 0
    assert audit["mass_kg"] == pytest.approx(32.0, rel=0.0, abs=1e-12)
    assert model.njnt == 6
    for key, name in arena.target_container_geom_names.items():
        geom_id = model.geom(name).id
        if key.endswith("_visual"):
            assert model.geom_contype[geom_id] == 0
            assert model.geom_conaffinity[geom_id] == 0
        else:
            assert model.geom_contype[geom_id] != 0
            assert model.geom_conaffinity[geom_id] != 0
    assert arena.target_container_spec["inner_xy_m"] == pytest.approx([0.164, 0.144])


def test_canonical_dimensions_cannot_be_silently_rescaled():
    with pytest.raises(ValueError, match="canonical table_full_size"):
        ShakeBenchArena(table_full_size=(0.8, 0.6, 0.06))


def test_empty_preload_and_payload_offset_are_observable_in_mujoco():
    arena, config, xml, model = _compile_deck_arena()
    data = mujoco.MjData(model)
    table_joint = model.joint("isolator_tz").id
    table_qpos = model.jnt_qposadr[table_joint]
    for _ in range(2500):
        mujoco.mj_step(model, data)
    assert data.qpos[table_qpos] == pytest.approx(0.0, rel=0.0, abs=1e-7)

    # Add a free payload as a world child after the arena/deck contract has
    # been processed. This mirrors the future Can topology without creating a
    # task or asking the arena to infer object names.
    import xml.etree.ElementTree as ET

    root = ET.fromstring(xml)
    worldbody = root.find("worldbody")
    payload = ET.SubElement(worldbody, "body", {"name": "phase03_payload", "pos": "0 0 0.81"})
    ET.SubElement(payload, "freejoint", {"name": "phase03_payload_free"})
    ET.SubElement(
        payload,
        "inertial",
        {"pos": "0 0 0", "mass": "0.349", "diaginertia": "0.0001 0.0001 0.0001"},
    )
    ET.SubElement(payload, "geom", {"name": "phase03_payload_geom", "type": "sphere", "size": "0.01"})
    payload_model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    payload_data = mujoco.MjData(payload_model)
    payload_qpos = payload_model.jnt_qposadr[payload_model.joint("isolator_tz").id]
    for _ in range(10000):
        mujoco.mj_step(payload_model, payload_data)
    expected = -0.349 * 9.81 / arena.isolator_parameters.stiffness[2]
    assert payload_data.qpos[payload_qpos] < -5e-5
    assert payload_data.qpos[payload_qpos] == pytest.approx(expected, rel=0.0, abs=3e-5)
    assert np.all(payload_data.warning.number == 0)
