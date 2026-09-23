"""Privileged, static-table ring stacking oracle using normalized OSC actions."""

import numpy as np

from robosuite.utils.control_utils import orientation_error


class RingStackOracle:
    """Grasp each ring wall, lift over the peg cap, align and release in order."""

    def __init__(self, env):
        self.env = env
        self.abort_requested = False
        self.site = env.robots[0].eef_site_id["right"]
        self.orientation = np.diag([-1.0, 1.0, -1.0])
        self._actions = self._run()

    def action(self):
        return next(self._actions, np.zeros(7))

    def _move(self, target, grip, steps, *, min_steps=0):
        settled = 0
        for step in range(steps):
            data = self.env.sim.data
            controller = self.env.robots[0].part_controllers["right"]
            delta = target - data.site_xpos[self.site]
            rotation = orientation_error(self.orientation, data.site_xmat[self.site].reshape(3, 3))
            settled = (
                settled + 1
                if step >= min_steps and np.linalg.norm(delta) < 0.002 and np.linalg.norm(rotation) < 0.02
                else 0
            )
            if settled >= 2:
                return
            if controller.input_ref_frame == "base":
                delta = controller.origin_ori.T @ delta
                rotation = controller.origin_ori.T @ rotation
            yield np.r_[np.clip(delta / 0.05, -1, 1), np.clip(rotation / 0.5, -1, 1), grip]

    def _run(self):
        env = self.env
        for index, (name, spec) in enumerate(env.get_policy_task_context()["rings"].items()):
            body = env.ring_body_ids[name]
            start = env.sim.data.xpos[body].copy()
            offset = self.orientation[:, 0] * (spec["outer_radius"] + spec["inner_radius"]) / 2
            offset[2] = 0
            grasp = start + offset
            peg = env.sim.data.xpos[env.peg_body_id].copy()
            lift = 0.21 if np.linalg.norm(start[:2] - peg[:2]) < 0.1 else 0.12
            for height, grip, steps, min_steps in (
                (0.06, -1, 60, 0),
                (0.003, -1, 55, 0),
                (0.003, 1, 20, 8),
                (lift, 1, 65, 0),
            ):
                yield from self._move(grasp + [0, 0, height], grip, steps, min_steps=min_steps)
            if env.sim.data.xpos[body][2] < start[2] + 0.08:
                self.abort_requested = True
                return
            offset = env.sim.data.xpos[body].copy() - env.sim.data.site_xpos[self.site]
            peg = env.sim.data.xpos[env.peg_body_id].copy()
            for height, grip, steps, min_steps in (
                (0.22, 1, 70, 0),
                (0.177, 1, 40, 0),
                (0.177, -1, 12, 4),
                (0.32, -1, 60, 0),
            ):
                yield from self._move(peg + [0, 0, height] - offset, grip, steps, min_steps=min_steps)
            if env.get_metrics()["success"]["stage"] != index + 1:
                self.abort_requested = True
                return
