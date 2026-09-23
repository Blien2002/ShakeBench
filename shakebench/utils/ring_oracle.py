"""Privileged, static-table ring stacking oracle using normalized OSC actions."""

import numpy as np

from robosuite.utils.control_utils import orientation_error


class RingStackOracle:
    """Grasp each ring wall, lift over the peg cap, align and release in order."""

    def __init__(self, env):
        self.env = env
        self.abort_requested = False
        self.site = env.robots[0].eef_site_id["right"]
        wrist_x = env.sim.data.site_xmat[self.site].reshape(3, 3)[:, 0].copy()
        wrist_x[2] = 0
        wrist_x /= np.linalg.norm(wrist_x)
        tool_z = np.array([0.0, 0.0, -1.0])
        self.orientation = np.column_stack((wrist_x, np.cross(tool_z, wrist_x), tool_z))
        self._actions = self._run()

    def action(self):
        return next(self._actions, np.zeros(7))

    def _move(self, target, grip, steps, *, min_steps=0, translation_limit=0.4, position_tolerance=0.002):
        settled = 0
        for step in range(steps):
            data = self.env.sim.data
            controller = self.env.robots[0].part_controllers["right"]
            delta = target - data.site_xpos[self.site]
            rotation = orientation_error(self.orientation, data.site_xmat[self.site].reshape(3, 3))
            settled = (
                settled + 1
                if step >= min_steps and np.linalg.norm(delta) < position_tolerance and np.linalg.norm(rotation) < 0.02
                else 0
            )
            if settled >= 2:
                return
            if controller.input_ref_frame == "base":
                delta = controller.origin_ori.T @ delta
                rotation = controller.origin_ori.T @ rotation
            yield np.r_[
                np.clip(delta / 0.08, -translation_limit, translation_limit), np.clip(rotation / 0.5, -1, 1), grip
            ]

    def _run(self):
        env = self.env
        for index, (name, spec) in enumerate(env.get_policy_task_context()["rings"].items()):
            body = env.ring_body_ids[name]
            start = env.sim.data.xpos[body].copy()
            peg = env.sim.data.xpos[env.peg_body_id].copy()
            ring_from_peg = start[:2] - peg[:2]
            side = 1.0 if ring_from_peg[np.argmax(np.abs(ring_from_peg))] >= 0 else -1.0
            offset = side * self.orientation[:, 0] * (spec["outer_radius"] + spec["inner_radius"]) / 2
            offset[2] = 0
            grasp = start + offset
            lift = 0.21 if np.linalg.norm(start[:2] - peg[:2]) < 0.1 else 0.12
            # Limit speed rather than feedback gain; allow finger closure before
            # millimetric contact offsets turn descent into a long stationary wait.
            for height, grip, steps, min_steps, translation_limit, position_tolerance in (
                (0.06, -1, 100, 0, 0.4, 0.002),
                (0.003, -1, 100, 0, 0.12, 0.005),
                (0.003, 1, 20, 8, 0.12, 0.005),
                (lift, 1, 100, 0, 0.4, 0.002),
            ):
                yield from self._move(
                    grasp + [0, 0, height],
                    grip,
                    steps,
                    min_steps=min_steps,
                    translation_limit=translation_limit,
                    position_tolerance=position_tolerance,
                )
            if env.sim.data.xpos[body][2] < start[2] + 0.08:
                self.abort_requested = True
                return
            offset = env.sim.data.xpos[body].copy() - env.sim.data.site_xpos[self.site]
            peg = env.sim.data.xpos[env.peg_body_id].copy()
            for height, grip, steps, min_steps in (
                (0.22, 1, 120, 0),
                (0.177, 1, 80, 0),
                (0.177, -1, 12, 4),
                (0.32, -1, 100, 0),
            ):
                yield from self._move(peg + [0, 0, height] - offset, grip, steps, min_steps=min_steps)
            if env.get_metrics()["success"]["stage"] != index + 1:
                self.abort_requested = True
                return
