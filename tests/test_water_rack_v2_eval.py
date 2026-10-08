"""Eval-only water v2: real support/containment/release in all three slots."""

import mujoco
import numpy as np
import pytest

from shakebench.environments.wine_rack import BOTTLE_AXIS_HEIGHT_M, SLOT_Y_M, WineRack, default_state


@pytest.mark.parametrize("variant", ["wine_3", "bottled_water_16"])
@pytest.mark.parametrize("location", ["middle", "left", "right"])
def test_unchanged_slot_rule_accepts_released_supported_bottle(variant, location):
    env = WineRack(task_state=default_state(location, bottle_variant=variant), physics_profile="official")
    try:
        model, data = env.sim.model._model, env.sim.data._data
        rotation = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
        rack = data.xmat[env.rack_body_id].reshape(3, 3)
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, (rack @ rotation).ravel())
        # Water's broad shoulder requires the validated elevated drop pose.
        # The old wine-only test pose starts water deeply interpenetrating the cradles.
        x, rise = (0.015, 0.053) if variant == "bottled_water_16" else (0.0, 0.013)
        position = data.xpos[env.rack_body_id] + rack @ [x, SLOT_Y_M[location], BOTTLE_AXIS_HEIGHT_M + rise]
        env.sim.data.set_joint_qpos(env.bottle.joints[0], [*position, *quat])
        env.sim.data.set_joint_qvel(env.bottle.joints[0], np.zeros(6))
        env.sim.forward()
        env._settle()
        env.sim.forward()
        env._record_post_physics_metrics(0.0)
        assert all(env._conditions.values()), env._conditions
        # Check the unchanged hold window against sustained real contacts, with the arm fixed.
        robot = env.robots[0]
        qids = [*robot._ref_joint_pos_indexes, *robot._ref_gripper_joint_pos_indexes["right"]]
        vids = [*robot._ref_joint_vel_indexes, *robot._ref_gripper_joint_vel_indexes["right"]]
        fixed = data.qpos[qids].copy()
        for _ in range(round(0.55 / model.opt.timestep)):
            mujoco.mj_step(model, data)
            data.qpos[qids], data.qvel[vids] = fixed, 0
            env._record_post_physics_metrics(float(data.time))
        assert env._check_success() and all(env._conditions.values())
        assert np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
        physics = env.get_policy_task_context()["bottle_physics"]
        assert physics["revision"] == ("water_physics_v2" if variant == "bottled_water_16" else "mesh_inertia_v1")
        assert physics["mass_kg"] == pytest.approx(0.7 if variant == "bottled_water_16" else 1.1)
        if variant == "bottled_water_16":
            assert physics["inertia_profile"] == "bottom_weighted_water_v1"
            assert physics["support_profile"] == "flat_sole_mesh_v1"
            assert model.body_mass[env.bottle_body_id] == pytest.approx(0.7)
            assert "wine_bottle_flat_sole" in env.bottle.contact_geoms
    finally:
        env.close()
