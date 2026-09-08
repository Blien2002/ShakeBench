"""Opt-in direct mounting must change real assembly and preserve physics isolation."""

import math

import mujoco
import numpy as np
import pytest

from robosuite.environments.manipulation.vibration_pick_place_can import VibrationPickPlaceCan
from robosuite.models.arenas import ShakeBenchArena
from robosuite.utils.shakebench_deck import DeckDriverConfig
from robosuite.utils.shakebench_geometry import geometry_scene_path, load_geometry_profile
from robosuite.utils.shakebench_isolator import AXES, relative_transmissibility
from robosuite.utils.shakebench_scene import compiled_physics_signature, scene_clearance_report


def make_env(**kwargs):
    return VibrationPickPlaceCan(
        robots="Panda",
        geometry_profile="direct_mount_v1",
        observation_tier=None,
        use_camera_obs=False,
        has_renderer=False,
        has_offscreen_renderer=False,
        renderer="mujoco",
        initialization_noise=None,
        seed=0,
        horizon=5,
        **kwargs,
    )


def new_arena():
    geometry = load_geometry_profile("direct_mount_v1")
    return ShakeBenchArena(
        table_offset=geometry["table_top_pos_m"], scene_config=geometry_scene_path("direct_mount_v1")
    )


def test_direct_mount_removes_pedestal_physics_and_places_four_supports():
    env = make_env()
    try:
        model, data = env.sim.model._model, env.sim.data._data
        assert env.robot_mount_type == "NullMount"
        assert not any("pedestal" in (model.geom(i).name or "") for i in range(model.ngeom))
        assert sum(model.body_mass[i] for i in range(model.nbody) if model.body(i).name.startswith("fixed_mount")) == 0
        np.testing.assert_allclose(data.body("robot0_link0").xpos, [-0.47, 0, 0.009], atol=1e-12)
        np.testing.assert_allclose(env.arena.table_top_abs, [0.18, 0, 0.299], atol=1e-12)
        assert model.body("worktable").mass == 32
        assert model.body("robot0_link0").mass == 4
        expected_initial_qpos = np.asarray(env.geometry_profile["initial_joint_qpos_rad"], dtype=float)
        np.testing.assert_allclose(env.robots[0].get_robot_joint_positions(), expected_initial_qpos, atol=1e-12)
        expected_eef = np.asarray(env.geometry_profile["initial_eef_pos_robot_base_m"], dtype=float)
        np.testing.assert_allclose(
            env.robots[0].pose_in_base_from_name(env.gripper_body_name)[:3, 3], expected_eef, atol=1e-6
        )
        table_contacts = [
            contact
            for contact in env.sim.data.contact[: env.sim.data.ncon]
            if "table_collision"
            in {
                model.geom(contact.geom1).name,
                model.geom(contact.geom2).name,
            }
        ]
        assert not table_contacts
        for i in range(4):
            np.testing.assert_allclose(
                data.geom(f"shakebench_table_upper_foot_{i}").xpos[:2],
                data.geom(f"shakebench_table_lower_mount_plate_{i}").xpos[:2],
                atol=1e-12,
            )
        assert env._scene_clearance.passed
        # Geometry does not self-authorize. The separate frozen science
        # authority permits evidence collection, while final Phase 8 handoff
        # remains independently pending.
        assert env.get_policy_task_context()["physics_profile"]["scoreable"] is True
    finally:
        env.close()


def test_direct_mount_visual_toggle_is_physically_invariant():
    visible, hidden = make_env(), make_env(scene_visual=False)
    try:
        assert compiled_physics_signature(visible.sim)["sha256"] == compiled_physics_signature(hidden.sim)["sha256"]
        action = np.zeros(visible.action_dim)
        visible.step(action)
        hidden.step(action)
        np.testing.assert_array_equal(visible.sim.data.qpos, hidden.sim.data.qpos)
        np.testing.assert_array_equal(visible.sim.data.qvel, hidden.sim.data.qvel)
    finally:
        visible.close()
        hidden.close()


def test_direct_mount_seating_exception_does_not_allow_buried_robot():
    env = make_env()
    try:
        model, data = env.sim.model._model, env.sim.data._data
        model.body_pos[model.body("robot0_base").id, 2] -= 0.01
        mujoco.mj_forward(model, data)
        report = scene_clearance_report(env.sim, env.scene_config)
        assert not report.passed
        assert any(x["geom2"] == "robot0_link0_collision" for x in report.unexpected_penetration_pairs)
    finally:
        env.close()


def test_profile_rejects_conflicting_mount_and_unbound_scene():
    with pytest.raises(ValueError, match="base_types conflicts"):
        make_env(base_types="RethinkMount")
    with pytest.raises(ValueError, match="explicit geometry_profile"):
        VibrationPickPlaceCan(robots="Panda", scene_config=geometry_scene_path("direct_mount_v1"))


def test_lower_worktable_preload_equilibrium():
    arena = new_arena()
    model = mujoco.MjModel.from_xml_string(
        arena.process_deck_xml(config=DeckDriverConfig(physics_timestep_s=0.0002, eq_solref=(0.0004, 0.5)))
    )
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data, nstep=2500)
    offset = data.qpos[model.jnt_qposadr[model.joint("isolator_tz").id]]
    assert offset == pytest.approx(0, abs=1e-7)
    assert not data.warning.number.any()


@pytest.mark.parametrize("axis", range(6), ids=AXES)
def test_lower_worktable_six_axis_transfer(axis):
    arena = new_arena()
    model = mujoco.MjModel.from_xml_string(
        arena.process_deck_xml(config=DeckDriverConfig(physics_timestep_s=0.0002, eq_solref=(0.0004, 0.5)))
    )
    data = mujoco.MjData(model)
    mocap = model.body_mocapid[model.body("deck_driver").id]
    address = model.jnt_qposadr[model.joint("isolator_" + AXES[axis]).id]
    amplitude, frequency = 0.0001, 2.0
    omega = 2 * math.pi * frequency
    times, values = [], []
    for _ in range(10000):
        t = data.time
        command = amplitude * math.sin(omega * t)
        if axis < 3:
            data.mocap_pos[mocap, axis] = command
        else:
            data.mocap_quat[mocap] = [math.cos(command / 2), 0, 0, 0]
            data.mocap_quat[mocap, axis - 2] = math.sin(command / 2)
        mujoco.mj_step(model, data)
        if t >= 0.6:
            times.append(t)
            values.append(data.qpos[address])
    times = np.asarray(times)
    design = np.column_stack([np.sin(omega * times), np.cos(omega * times), np.ones_like(times)])
    fit = np.linalg.lstsq(design, values, rcond=None)[0]
    measured = np.linalg.norm(fit[:2]) / amplitude
    assert measured == pytest.approx(abs(relative_transmissibility(frequency, 5, 0.1)), rel=0.05, abs=1e-4)
    assert not data.warning.number.any()
