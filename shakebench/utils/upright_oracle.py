"""Experimental upright grasp probe; it does not yet stand or place objects.

This intentionally stays outside dataset collection until a physical rollout
passes the released-and-stable upright success condition for all three objects.
"""

import numpy as np

from shakebench.utils.ring_oracle import RingStackOracle


class UprightOracle(RingStackOracle):
    """Reuse the ring oracle's OSC path follower to qualify a bottle grasp."""

    def __init__(self, env):
        if env.task_state["task"]["task_type"] != "upright":
            raise ValueError("UprightOracle requires an upright task state")
        self.failure_reason = None
        self.verified = False
        super().__init__(env)

    def _stop(self, reason):
        self.failure_reason = reason
        self.abort_requested = True
        self.phase = "aborted"

    def _run(self):
        env = self.env
        object_id = env.task_state["task"]["object_id"]
        if object_id != "wine_bottle":
            self._stop(f"{object_id}_grasp_unqualified")
            return

        body = env.object_body_id
        start = env.sim.data.xpos[body].copy()
        object_up = env.sim.data.xmat[body].reshape(3, 3)[:, 2]
        table_up = env.sim.data.xmat[env.table_body_id].reshape(3, 3)[:, 2]
        grasp = start - 0.06 * object_up + 0.012 * table_up

        self.phase = "approach"
        yield from self._follow([grasp + 0.09 * table_up], -1, speed=0.25, acceleration=0.5, response_scale=0.2)
        if self.abort_requested:
            self._stop("approach_stalled")
            return
        self.phase = "descend"
        yield from self._follow([grasp], -1, speed=0.08, acceleration=0.3, response_scale=0.2)
        if self.abort_requested:
            self._stop("descend_stalled")
            return

        self.phase = "grasp"
        yield from self._hold(1, 16)
        if not env.get_metrics()["touching_gripper"]:
            self._stop("no_gripper_contact")
            return
        offset = env.sim.data.xpos[body].copy() - env.sim.data.site_xpos[self.site].copy()
        self.phase = "test_lift"
        yield from self._follow(
            [env.sim.data.site_xpos[self.site].copy() + 0.02 * table_up],
            1,
            speed=0.08,
            acceleration=0.3,
            response_scale=0.2,
        )
        if self.abort_requested:
            self._stop("test_lift_stalled")
            return
        new_offset = env.sim.data.xpos[body] - env.sim.data.site_xpos[self.site]
        if (
            env.sim.data.xpos[body][2] < start[2] + 0.01
            or np.linalg.norm(new_offset - offset) > 0.015
            or env.get_metrics()["touching_table"]
        ):
            self._stop("test_lift_slipped")
            return

        # A qualified grasp must not be mistaken for task success: rotation,
        # placement and released stability are still unimplemented.
        self._stop("rotation_unimplemented")
