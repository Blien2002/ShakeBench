"""Appearance acceptance: cabinet orientation and continuous safety rail joints."""

import mujoco
import numpy as np
import pytest

from robosuite.environments.manipulation.vibration_pick_place_can import VibrationPickPlaceCan


@pytest.fixture(scope="module")
def env():
    env = VibrationPickPlaceCan(
        robots="Panda",
        geometry_profile="direct_mount_v1",
        observation_tier=None,
        use_camera_obs=False,
        has_renderer=False,
        has_offscreen_renderer=False,
        renderer="mujoco",
        initialization_noise=None,
        seed=0,
        horizon=1,
    )
    yield env
    env.close()


def test_cabinet_operator_face_points_towards_shaker_and_monitor_is_blank(env):
    model, data = env.sim.model._model, env.sim.data._data
    cabinet = data.body("shakebench_control_cabinet")
    inward = -cabinet.xpos.copy()
    inward[2] = 0
    inward /= np.linalg.norm(inward)
    normal = cabinet.xmat.reshape(3, 3) @ np.array([0, -1, 0])
    assert np.dot(inward, normal) > 0.99999
    names = [model.geom(i).name or "" for i in range(model.ngeom)]
    assert not any(name.startswith("shakebench_instrument_bench_display_bar_") for name in names)
    screen = model.geom("shakebench_instrument_bench_monitor_screen")
    assert model.mat_emission[screen.matid] == 0


def test_curved_elbow_centres_meet_straight_tube_ends(env):
    model, data = env.sim.model._model, env.sim.data._data
    ends = []
    for i in range(4):
        geom = model.geom(f"shakebench_guardrail_top_{i}")
        pose = data.geom(geom.id)
        axis = pose.xmat.reshape(3, 3)[:, 2]
        ends.extend([pose.xpos - axis * geom.size[1], pose.xpos + axis * geom.size[1]])
    elbow_ends = []
    ring_size = env.scene_config.section("pit")["guardrail_details"]["elbow_ring_segments"]
    for i in range(4):
        geom = model.geom(f"shakebench_guardrail_elbow_{i}")
        mesh_id = model.geom_dataid[geom.id]
        start, count = model.mesh_vertadr[mesh_id], model.mesh_vertnum[mesh_id]
        verts = model.mesh_vert[start : start + count]
        pose = data.geom(geom.id)
        world = verts @ pose.xmat.reshape(3, 3).T + pose.xpos
        elbow_ends.extend([world[:ring_size].mean(axis=0), world[-ring_size:].mean(axis=0)])
    distances = np.linalg.norm(np.asarray(ends)[:, None, :] - np.asarray(elbow_ends)[None, :, :], axis=2)
    assert np.all(distances.min(axis=1) < 1e-6)
    assert len(set(distances.argmin(axis=1))) == 8
    assert env._scene_clearance.passed


def test_only_one_explicit_light_casts_shadows(env):
    model = env.sim.model._model
    assert np.count_nonzero(model.light_castshadow) == 1
    assert model.light("shakebench_key_light").castshadow
    assert not model.light("shakebench_fill_light").castshadow
