"""Plain response-rule regression; native scorer and a small real MuJoCo contact fixture."""

import unittest
from collections import deque
from types import SimpleNamespace

import mujoco
import numpy as np

from shakebench.environments.stack_blocks import StackBlocks, plain_footprints_overlap


def scorer(joint="plain", offsets=(0.020, -0.020), yaw=0.6):
    env = StackBlocks.__new__(StackBlocks)
    env.joint, env.stack_state = joint, {"order": ["blue", "green", "yellow"]}
    env.block_body_ids = {"blue": 1, "green": 2, "yellow": 3}
    env.block_geom_ids = {name: {body} for name, body in env.block_body_ids.items()}
    env.table_body_id, env.table_geom_id, env.robot_geom_ids = 0, 0, {4}
    rotations = np.tile(np.eye(3), (5, 1, 1))
    for body in (2, 3):
        angle = yaw * (body - 1)
        rotations[body] = [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
    positions = np.array(
        [[0, 0, 0], [0, 0, 0.025], [offsets[0], 0, 0.075], [offsets[0] + offsets[1], 0, 0.125], [0, 0, 1.0]]
    )
    env.sim = SimpleNamespace(data=SimpleNamespace(_data=SimpleNamespace(xmat=rotations.reshape(5, 9), xpos=positions)))
    env._history = {name: deque() for name in ("green", "yellow")}
    env._conditions, env._success, env._stage, env._candidate_since = {}, False, 0, None
    env._last_metric_time = -1.0
    env._sample_diagnostics = lambda _t: None
    env._contacts = lambda name, geoms, support_up_world=None: (1.0, True)
    return env


def advance(env, seconds=1.7):
    for t in np.arange(0.0, seconds, 0.01):
        env._record_post_physics_metrics(float(t))
    return env._success


class PlainRules(unittest.TestCase):
    def test_staggered_yawed_supported_order_succeeds(self):
        env = scorer()
        self.assertTrue(advance(env))
        self.assertTrue(env._conditions["green"]["aligned"])

    def test_no_geometric_overlap_fails_even_with_spurious_force(self):
        self.assertFalse(advance(scorer(offsets=(0.1, 0))))

    def test_edge_only_touch_is_not_overlap(self):
        self.assertFalse(plain_footprints_overlap(np.array([0.05, 0, 0.05]), np.eye(3)))

    def test_reverse_vertical_order_and_missing_block_fail(self):
        env = scorer()
        env.sim.data._data.xpos[[2, 3]] = env.sim.data._data.xpos[[3, 2]]
        self.assertFalse(advance(env))
        env = scorer()
        env._contacts = lambda name, geoms, support_up_world=None: (0.0 if name == "yellow" else 1.0, True)
        self.assertFalse(advance(env))

    def test_floating_side_by_side_and_robot_supported_fail(self):
        env = scorer()
        env.sim.data._data.xpos[2, 2] = 0.025
        self.assertFalse(advance(env))
        env = scorer()
        env._contacts = lambda name, geoms, support_up_world=None: (1.0, name != "yellow")
        self.assertFalse(advance(env))
        env = scorer()
        env._contacts = lambda name, geoms, support_up_world=None: (0.0, True)
        self.assertFalse(advance(env))

    def test_transient_contact_and_moving_tower_fail(self):
        self.assertFalse(advance(scorer(), seconds=0.3))
        env = scorer()
        for i in range(170):
            env.sim.data._data.xpos[2, 0] = 0.020 + 0.005 * np.sin(i)
            env._record_post_physics_metrics(i * 0.01)
        self.assertFalse(env._success)
        env = scorer()
        for i in range(170):
            angle = 0.10 * np.sin(i)
            env.sim.data._data.xmat[3] = np.array(
                [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
            ).ravel()
            env._record_post_physics_metrics(i * 0.01)
        self.assertFalse(env._success)

    def test_tenon_native_condition_dictionary_unchanged(self):
        # Load the exact pre-patch method source into a standalone function, without importing/registering tasks twice.
        import textwrap
        from pathlib import Path

        namespace = {"np": np}
        import shakebench.environments.stack_blocks as current

        namespace.update(
            {
                k: getattr(current, k)
                for k in (
                    "DRIFT_WINDOW_S",
                    "DRIFT_POSITION_M",
                    "DRIFT_ANGLE_RAD",
                    "BLOCK_HALF_M",
                    "TILT_MAX_RAD",
                    "ALIGN_TOLERANCE_M",
                )
            }
        )
        before = Path(__file__).resolve().parent / "fixtures/stack_plain_v2_before.txt"
        text = before.read_text()
        method = text[text.index("    def _block_conditions") : text.index("    def _new_diagnostics")]
        exec(textwrap.dedent(method), namespace)
        for joint in ("tenon", "tenon_tight", "tenon_loose"):
            for offset in (0.0, 0.02, 0.1):
                old, new = scorer(joint=joint, offsets=(offset, 0.0)), scorer(joint=joint, offsets=(offset, 0.0))
                for t in np.arange(0, 0.8, 0.01):
                    self.assertEqual(
                        namespace["_block_conditions"](old, 1, "green", float(t)),
                        new._block_conditions(1, "green", float(t)),
                    )

    def test_real_mujoco_offset_stack_load_bearing_contacts(self):
        xml = '<mujoco><option timestep=".001"/><worldbody><geom name="table" type="plane" size="1 1 .1"/>'
        for name, x, z in (("blue", 0, 0.025), ("green", 0.018, 0.075), ("yellow", 0, 0.125)):
            xml += f'<body name="{name}" pos="{x} 0 {z}"><freejoint/><geom name="{name}" type="box" size=".025 .025 .025" mass=".08125" friction=".45 .005 .0001"/></body>'
        xml += "</worldbody></mujoco>"
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        env = scorer(offsets=(0.018, -0.018), yaw=0.0)
        env.sim = SimpleNamespace(model=SimpleNamespace(_model=model), data=SimpleNamespace(_data=data))
        env.block_body_ids = {name: model.body(name).id for name in ("blue", "green", "yellow")}
        env.block_geom_ids = {name: {model.geom(name).id} for name in env.block_body_ids}
        env.table_body_id, env.table_geom_id, env.robot_geom_ids = 0, model.geom("table").id, set()
        env._contacts = StackBlocks._contacts.__get__(env)
        for step in range(2000):
            mujoco.mj_step(model, data)
            env._record_post_physics_metrics(data.time)
        self.assertTrue(env._success)
        self.assertTrue(env._conditions["green"]["supported"])
        relative = data.xpos[env.block_body_ids["green"]] - data.xpos[env.block_body_ids["blue"]]
        self.assertGreater(np.linalg.norm(relative[:2]), 0.0125)


if __name__ == "__main__":
    unittest.main()
