"""Phase 7.5A scene authority, frame, and clearance gates."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

from robosuite.environments.manipulation.vibration_pick_place_can import VibrationPickPlaceCan
from robosuite.models.arenas import ShakeBenchArena
from robosuite.utils.shakebench_deck import DeckDriverConfig
from robosuite.utils.shakebench_geometry import geometry_scene_path, load_geometry_profile
from robosuite.utils.shakebench_scene import (
    SceneConfigError,
    audit_compiled_scene,
    compiled_physics_signature,
    load_scene_visual_config,
    scene_clearance_report,
    scene_config_hash,
)


def _write_scene_config(tmp_path: Path, mutate) -> Path:
    payload = load_scene_visual_config().to_dict()
    mutate(payload)
    payload["payload_sha256"] = scene_config_hash(payload)
    path = tmp_path / "scene.json"
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def _arena_deck_model(arena: ShakeBenchArena):
    config = DeckDriverConfig(physics_timestep_s=0.0002, eq_solref=(0.0004, 0.5))
    xml = arena.process_deck_xml(config=config)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def _assert_nested_equal(first, second):
    if isinstance(first, dict):
        assert isinstance(second, dict)
        assert set(first) == set(second)
        for key in first:
            _assert_nested_equal(first[key], second[key])
    elif isinstance(first, np.ndarray):
        np.testing.assert_array_equal(first, second)
    elif isinstance(first, (list, tuple)):
        assert isinstance(second, type(first))
        assert len(first) == len(second)
        for first_item, second_item in zip(first, second):
            _assert_nested_equal(first_item, second_item)
    else:
        assert first == second


def test_scene_config_authenticates_source_and_package_assets():
    config = load_scene_visual_config()
    assert config.schema_id == "shakebench.scene_visual"
    assert config.geometry_variant == "A"
    assert config.physics_effect is False
    assert config.config_sha256 == scene_config_hash(config.to_dict())
    assert all(not Path(item["path"]).is_absolute() for item in config.section("source")["reference_files"])


@pytest.mark.parametrize(
    "mutate, pattern",
    (
        (lambda payload: payload["platen"].update(size_m=[0.0, 1.1, 0.08]), "platen.size_m"),
        (lambda payload: payload["expected_roles"].append("shakebench_platen_visual"), "duplicate"),
        (lambda payload: payload["physics_effect"].__class__ and payload.update(physics_effect=True), "physics_effect"),
        (lambda payload: payload["room"]["equipment"][0]["geoms"][0].update(contype=1), "unsupported fields"),
        (lambda payload: payload["render"]["lights"][0].update(dir=[0, 0, 0]), "non-zero"),
        (lambda payload: payload["materials"].update(wall_texture="../outside.png"), "package-relative"),
    ),
)
def test_scene_config_fails_closed_for_invalid_payloads(tmp_path: Path, mutate, pattern: str):
    path = _write_scene_config(tmp_path, mutate)
    with pytest.raises(SceneConfigError, match=pattern):
        load_scene_visual_config(path)


def test_compiled_inventory_binds_platen_and_supports_to_their_frames():
    arena = ShakeBenchArena()
    model, data = _arena_deck_model(arena)
    sim = type("Sim", (), {"model": model, "data": data})()
    audit = audit_compiled_scene(sim, arena.scene_config)
    assert audit.passed
    assert audit.role_audit["deck_visual"]["frame"] == "dynamic_deck"
    assert audit.role_audit["isolated_worktable"]["frame"] == "isolated_worktable"
    assert audit.visual_geoms["shakebench_table_upper_leg_0"]["frame"] == "isolated_worktable"
    assert audit.visual_geoms["shakebench_table_lower_mount_plate_0"]["frame"] == "dynamic_deck"
    assert audit.visual_geoms["shakebench_stewart_outer_0"]["frame"] == "world"
    assert audit.visual_geoms["shakebench_stewart_rod_0"]["frame"] == "dynamic_deck"


@pytest.mark.parametrize("geometry_profile", ["canonical", "direct_mount_v1"])
def test_table_feet_clear_deck_through_isolator_travel(geometry_profile):
    if geometry_profile == "canonical":
        arena = ShakeBenchArena()
    else:
        geometry = load_geometry_profile(geometry_profile)
        arena = ShakeBenchArena(
            table_offset=geometry["table_top_pos_m"], scene_config=geometry_scene_path(geometry_profile)
        )
    model, data = _arena_deck_model(arena)
    joints = [model.joint(f"isolator_{axis}") for axis in ("tx", "ty", "tz", "rx", "ry", "rz")]
    poses = [np.zeros(6), [0, 0, -0.003, 0, 0, 0], *itertools.product(*(joint.range for joint in joints))]
    for pose in poses:
        for joint, value in zip(joints, pose):
            data.qpos[joint.qposadr] = value
        mujoco.mj_forward(model, data)
        deck = model.geom("shakebench_platen_surface")
        deck_top = data.geom(deck.id).xpos[2] + deck.size[2]
        for index in range(4):
            for part in ("foot", "leg"):
                geom = model.geom(f"shakebench_table_upper_{part}_{index}")
                world = data.geom(geom.id)
                bottom = world.xpos[2] - np.abs(world.xmat.reshape(3, 3)[2]) @ geom.size
                assert bottom > deck_top, (geometry_profile, geom.name, pose, bottom - deck_top)


def test_visual_switch_preserves_name_based_physics_signature():
    visible = ShakeBenchArena(visual=True)
    hidden = ShakeBenchArena(visual=False)
    visible_model, _ = _arena_deck_model(visible)
    hidden_model, _ = _arena_deck_model(hidden)
    assert compiled_physics_signature(visible_model)["sha256"] == compiled_physics_signature(hidden_model)["sha256"]
    np.testing.assert_array_equal(visible_model.geom_contype, hidden_model.geom_contype)
    np.testing.assert_array_equal(visible_model.geom_conaffinity, hidden_model.geom_conaffinity)
    assert not np.array_equal(visible_model.geom_rgba, hidden_model.geom_rgba)


def test_visual_switch_preserves_named_state_and_metric_trace():
    kwargs = {
        "robots": "Panda",
        "observation_tier": None,
        "use_camera_obs": False,
        "has_renderer": False,
        "has_offscreen_renderer": False,
        "renderer": "mujoco",
        "initialization_noise": None,
        "seed": 0,
        "horizon": 2,
    }
    visible = VibrationPickPlaceCan(scene_visual=True, **kwargs)
    hidden = VibrationPickPlaceCan(scene_visual=False, **kwargs)
    try:
        named_bodies = ("deck", "worktable", "robot0_base", "can_main")
        visible_obs = visible.reset()
        hidden_obs = hidden.reset()
        for key in visible_obs:
            np.testing.assert_array_equal(visible_obs[key], hidden_obs[key])
        np.testing.assert_array_equal(visible.sim.data.qpos, hidden.sim.data.qpos)
        np.testing.assert_array_equal(visible.sim.data.qvel, hidden.sim.data.qvel)
        for name in named_bodies:
            visible_id = visible.sim.model.body_name2id(name)
            hidden_id = hidden.sim.model.body_name2id(name)
            np.testing.assert_array_equal(visible.sim.data.xpos[visible_id], hidden.sim.data.xpos[hidden_id])
            np.testing.assert_array_equal(visible.sim.data.xquat[visible_id], hidden.sim.data.xquat[hidden_id])
        action = np.zeros(visible.action_dim)
        visible.step(action)
        hidden.step(action)
        np.testing.assert_array_equal(visible.sim.data.qpos, hidden.sim.data.qpos)
        np.testing.assert_array_equal(visible.sim.data.qvel, hidden.sim.data.qvel)
        _assert_nested_equal(visible.get_metrics(), hidden.get_metrics())
    finally:
        visible.close()
        hidden.close()


def test_visual_bodies_add_no_mass_joint_actuator_or_explicit_contact_pair():
    arena = ShakeBenchArena()
    model, _ = _arena_deck_model(arena)
    audit = audit_compiled_scene(type("Sim", (), {"model": model})(), arena.scene_config)
    assert all(item["subtree_mass_kg"] == 0.0 for item in audit.visual_bodies.values())
    assert all(not item["joint_names"] for item in audit.visual_bodies.values())
    scene_names = set(audit.visual_geoms)
    for pair_id in range(model.npair):
        names = {
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(model.pair_geom1[pair_id])),
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(model.pair_geom2[pair_id])),
        }
        assert not names.intersection(scene_names)


def test_lab_wheels_touch_floor_and_buried_wheel_is_rejected():
    """Floor support confirmation must not whitelist a wheel penetrating the slab."""
    from robosuite.utils.shakebench_scene import _distance_record

    arena = ShakeBenchArena()
    model, data = _arena_deck_model(arena)
    floor = model.geom("shakebench_floor_slab_left").id
    wheel = model.geom("shakebench_tool_cart_wheel_0").id

    def measure():
        return _distance_record(
            model,
            data,
            "visual_visual",
            floor,
            model.geom(floor).name,
            wheel,
            model.geom(wheel).name,
            0,
            np.array([0, 0, 0, 1, 0, 0, 0]),
        )

    tangent = measure()
    assert tangent["method"] == "analytic_cylinder_support_to_containing_floor_top"
    assert tangent["signed_distance_m"] == pytest.approx(0.0, abs=1e-12)
    model.geom_pos[wheel, 2] -= 0.01
    mujoco.mj_forward(model, data)
    buried = measure()
    assert buried["signed_distance_m"] == pytest.approx(-0.01, abs=1e-12)
    assert buried["whitelist_reason"] is None


def test_nominal_and_registered_safe_scene_clearance_passes():
    env = VibrationPickPlaceCan(
        robots="Panda",
        observation_tier=None,
        use_camera_obs=False,
        has_renderer=False,
        has_offscreen_renderer=False,
        renderer="mujoco",
        initialization_noise=None,
        seed=0,
        horizon=1,
    )
    try:
        report = scene_clearance_report(env.sim, env.scene_config)
        assert report.passed
        assert report.clearance_passed
        assert report.safe_envelope["sample_count"] == 21
        assert report.support_interfaces["worktable_assembly_error_m"] == pytest.approx(0.0, abs=1.0e-9)
        assert report.support_interfaces["robot_mount_assembly_error_m"] == pytest.approx(0.0, abs=1.0e-9)
        assert report.stewart["passed"]
        assert not any(record["include_in_scene_gate"] for record in report.active_physical_contacts)
    finally:
        env.close()


def test_negative_clearance_fixture_rejects_table_leg_inside_mount_proxy():
    env = VibrationPickPlaceCan(
        robots="Panda",
        observation_tier=None,
        use_camera_obs=False,
        has_renderer=False,
        has_offscreen_renderer=False,
        renderer="mujoco",
        initialization_noise=None,
        seed=0,
        horizon=1,
    )
    try:
        model = env.sim.model._model
        leg_id = model.geom("shakebench_table_upper_leg_0").id
        model.geom_pos[leg_id, 0] = -0.50
        mujoco.mj_forward(model, env.sim.data._data)
        report = scene_clearance_report(env.sim, env.scene_config)
        assert not report.passed
        assert any(
            "shakebench_table_upper_leg_0" in {item["geom1"], item["geom2"]} and item["signed_distance_m"] < 0.0
            for item in report.unexpected_penetration_pairs
        )
    finally:
        env.close()


def test_floor_visual_switch_does_not_change_physical_floor_contact_semantics():
    visible = ShakeBenchArena(visual=True)
    hidden = ShakeBenchArena(visual=False)
    visible_model, _ = _arena_deck_model(visible)
    hidden_model, _ = _arena_deck_model(hidden)
    floor_visible = visible_model.geom("floor").id
    floor_hidden = hidden_model.geom("floor").id
    assert visible_model.geom_contype[floor_visible] == hidden_model.geom_contype[floor_hidden]
    assert visible_model.geom_conaffinity[floor_visible] == hidden_model.geom_conaffinity[floor_hidden]
    assert visible_model.geom_condim[floor_visible] == hidden_model.geom_condim[floor_hidden]


def test_scene_cameras_are_named_and_present_in_the_compiled_task():
    env = VibrationPickPlaceCan(
        robots="Panda",
        observation_tier=None,
        use_camera_obs=False,
        has_renderer=False,
        has_offscreen_renderer=False,
        renderer="mujoco",
        initialization_noise=None,
        seed=0,
        horizon=1,
    )
    try:
        model = env.sim.model._model
        names = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, index) for index in range(model.ncam)}
        assert {"shakebench_camera_overview", "shakebench_camera_assembly", "shakebench_camera_side"}.issubset(names)
    finally:
        env.close()
