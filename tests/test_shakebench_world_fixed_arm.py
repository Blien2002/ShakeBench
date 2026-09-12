"""World-fixed installation, motion, clearance and task-point reachability."""

import math

import mujoco
import numpy as np
import pytest
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from robosuite.environments.manipulation.vibration_pick_place import VibrationPickPlace
from robosuite.models.arenas import ShakeBenchArena
from robosuite.utils.shakebench_calibration import level_scale_for_gamma
from robosuite.utils.shakebench_deck import DeckDriverConfig
from robosuite.utils.shakebench_excitation import build_excitation_program
from robosuite.utils.shakebench_geometry import geometry_scene_path, load_geometry_profile
from robosuite.utils.shakebench_isolator import AXES, relative_transmissibility
from robosuite.utils.shakebench_scene import compiled_physics_signature, scene_clearance_report
from robosuite.utils.shakebench_tasks import task_variants


def make_env(**kwargs):
    return VibrationPickPlace(
        robots="Panda",
        use_camera_obs=False,
        has_renderer=False,
        has_offscreen_renderer=False,
        initialization_noise=None,
        seed=0,
        horizon=100,
        hard_reset=False,
        **kwargs,
    )


def new_arena():
    geometry = load_geometry_profile()
    return ShakeBenchArena(
        table_offset=geometry["table_top_pos_m"], scene_config=geometry_scene_path("world_fixed_arm_v1")
    )


def check_reachability(env):
    model = env.sim.model._model
    data = mujoco.MjData(model)
    data.qpos[:] = env.sim.data.qpos
    robot = env.robots[0]
    indices = robot._ref_joint_pos_indexes
    joint_ids = robot._ref_joint_indexes
    site = robot.eef_site_id["right"]
    rotation = Rotation.from_euler("x", np.pi).as_matrix()
    obj = env.sim.data.body_xpos[env.can_body_id].copy()
    target = env.sim.data.get_body_xpos("worktable") + env.target_frame_local_origin_m
    seed = robot.init_qpos
    for point in (obj, target + [0, 0, 0.08]):

        def residual(q):
            data.qpos[indices] = q
            mujoco.mj_forward(model, data)
            return np.r_[
                data.site_xpos[site] - point,
                0.2 * Rotation.from_matrix(rotation @ data.site_xmat[site].reshape(3, 3).T).as_rotvec(),
            ]

        fit = least_squares(
            residual,
            seed,
            bounds=(model.jnt_range[joint_ids, 0] + 1e-6, model.jnt_range[joint_ids, 1] - 1e-6),
            max_nfev=150,
        )
        assert np.linalg.norm(residual(fit.x)[:3]) < 0.002, (point, fit.fun)
        assert np.linalg.norm(fit.fun[3:]) < 0.002
        seed = fit.x


@pytest.mark.parametrize("task", task_variants(), ids=lambda t: t.object_id + "_" + t.surface_id)
def test_world_fixed_scene(task):
    program = build_excitation_program(seed=42, level_scale=level_scale_for_gamma(0.15, seed=42))
    env = make_env(task=task, deck_trajectory=program)
    try:
        model, data = env.sim.model._model, env.sim.data._data
        assert env._scene_clearance.passed
        assert model.body("worktable").mass == 32
        for name in ("robot0_base", "robot_support"):
            assert model.body(name).parentid == 0
            assert model.body(name).jntnum == 0
            assert model.body(name).mocapid == -1
        assert model.body("worktable").parentid == model.body("deck").id
        assert "robot_base" not in env.deck_driver.role_handles
        foundation = model.geom("robot_support_foundation")
        np.testing.assert_allclose(data.geom(foundation.id).xpos - foundation.size, [-1.225, -0.42, -0.765])
        np.testing.assert_allclose(data.geom(foundation.id).xpos + foundation.size, [-0.47, 0.42, 0])
        assert foundation.matid == model.geom("shakebench_floor_slab_left").matid
        assert env.robot_mount_type == "NullMount"
        assert not any((model.geom(i).name or "").startswith("fixed_mount") for i in range(model.ngeom))
        assert {
            int(model.geom_contype[model.geom(f"robot_support_trim_{side}").id])
            for side in ("front", "south", "north", "front_wall", "south_wall", "north_wall")
        } == {0}
        for side in ("front_wall", "south_wall", "north_wall"):
            trim = model.geom(f"robot_support_trim_{side}")
            np.testing.assert_allclose(data.geom(trim.id).xpos[2] - trim.size[2], -0.765)
            np.testing.assert_allclose(data.geom(trim.id).xpos[2] + trim.size[2], 0)
            np.testing.assert_allclose(trim.rgba, [0.04, 0.043, 0.045, 1])
        for side in ("south", "north"):
            old = model.geom(f"shakebench_pit_border_xneg_{side}")
            new = model.geom(f"robot_support_trim_{side}")
            np.testing.assert_allclose(data.geom(old.id).xpos[0] + old.size[0], data.geom(new.id).xpos[0] - new.size[0])
            assert max(data.geom(old.id).xpos[1] - old.size[1], data.geom(new.id).xpos[1] - new.size[1]) < min(
                data.geom(old.id).xpos[1] + old.size[1], data.geom(new.id).xpos[1] + new.size[1]
            )
            np.testing.assert_allclose(old.rgba, new.rgba)
        deck_geom = model.geom("shakebench_platen_surface")
        deck_left = data.geom(deck_geom.id).xpos[0] - deck_geom.size[0]
        assert deck_left > -0.47 + 0.05
        assert 0.05 < env.arena.table_top_abs[0] - env.table_full_size[0] / 2 - deck_left < 0.07
        assert not any(model.geom(i).name == "shakebench_guardrail_post_1" for i in range(model.ngeom))
        check_reachability(env)
        base = np.r_[data.body("robot0_base").xpos, data.body("robot0_base").xquat].copy()
        table_positions = []
        deck_errors = []
        for _ in range(20):
            env.step(np.zeros(env.action_dim))
            np.testing.assert_array_equal(np.r_[data.body("robot0_base").xpos, data.body("robot0_base").xquat], base)
            table_positions.append(data.body("worktable").xpos.copy())
            deck_errors.append(np.linalg.norm(data.body("deck").xpos - program.evaluate(data.time).q[:3]))
        assert np.ptp(table_positions, axis=0).max() > 1e-6
        assert max(deck_errors) < 0.002
        assert not data.warning.number.any()
        context = env.get_task_context()
        assert context["support_topology_id"] == "world_fixed_arm_v1"
        np.testing.assert_allclose(
            context["task_context"]["world_to_robot_base_position_m"], data.body("robot0_base").xpos
        )
        assert context["physics_profile"]["scoreable"] is False
        assert "deck_to_robot_base" not in str(context)
        env.deck_driver.trajectory = None
        env.reset()
        data = env.sim.data._data
        for _ in range(5):
            env.step(np.zeros(env.action_dim))
            np.testing.assert_array_equal(data.body("robot0_base").xpos, base[:3])
            assert np.linalg.norm(data.body("deck").xpos) < 0.002
        assert not data.warning.number.any()
    finally:
        env.close()


def test_support_collision_and_fixed_topology_regressions():
    env = make_env()
    try:
        model, data = env.sim.model._model, env.sim.data._data
        support = model.geom("robot_support_foundation").id
        model.geom_pos[support, 2] -= 0.11
        mujoco.mj_forward(model, data)
        report = scene_clearance_report(env.sim, env.scene_config)
        assert not report.passed
        assert any(row["geom2"] == "robot_support_foundation" for row in report.unexpected_penetration_pairs)
        model.body_parentid[model.body("robot0_base").id] = model.body("deck").id
        with pytest.raises(ValueError, match="direct world child"):
            env.audit_compiled_model()
    finally:
        env.close()


def test_visual_toggle_preserves_physics():
    visible, hidden = make_env(), make_env(scene_visual=False)
    try:
        assert compiled_physics_signature(visible.sim)["sha256"] == compiled_physics_signature(hidden.sim)["sha256"]
    finally:
        visible.close()
        hidden.close()


def test_reject_superseded_mounts():
    for profile in ("canonical", "direct_mount_v1"):
        with pytest.raises(ValueError, match="world_fixed_arm_v1"):
            load_geometry_profile(profile)
    with pytest.raises(ValueError, match="base_types conflicts"):
        make_env(base_types="RethinkMount")


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


def test_support_gap_uses_conservative_bound_without_masking_penetration():
    from robosuite.models.arenas.scene_audit import _distance_record

    model = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody>"
        '<geom name="fixed_mount0_pedestal_feet_col" type="box" size=".1 .1 .1"/>'
        '<geom name="moving_box" type="box" size=".1 .1 .1" pos="0 0 .3"/>'
        "</worldbody></mujoco>"
    )
    data = mujoco.MjData(model)

    def measure():
        mujoco.mj_forward(model, data)
        return _distance_record(
            model,
            data,
            "support_motion_clearance",
            0,
            model.geom(0).name,
            1,
            "moving_box",
            0,
            np.array([0, 0, 0, 1, 0, 0, 0]),
        )

    record = measure()
    assert record["signed_distance_m"] == pytest.approx(0.1)
    assert record["method"] == "conservative_aabb_separation_lower_bound"
    model.geom_pos[1, 2] = 0.15
    assert measure()["signed_distance_m"] < 0
