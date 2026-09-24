"""Privileged, static-table ring stacking oracle using smooth OSC actions."""

import numpy as np

from robosuite.utils.control_utils import orientation_error


# Open Panda finger mesh reaches about 0.055 m from the EEF in XY; keep a small clearance.
OPEN_FINGER_SWEEP_RADIUS_M = 0.06


class RingStackOracle:
    """Grasp each ring wall, lift over the peg cap, align and release in order."""

    def __init__(self, env):
        self.env = env
        self.abort_requested = False
        self.phase = "idle"
        self.site = env.robots[0].eef_site_id["right"]
        # A shared 7.3 degree downward correction is the smallest fixed pose that
        # reaches the full spawn region; keep it fixed after the approach.
        wrist_x = env.sim.data.site_xmat[self.site].reshape(3, 3)[:, 0].copy()
        wrist_x[2] = 0
        wrist_x /= np.linalg.norm(wrist_x)
        tool_z = np.array([0.0, 0.0, -1.0])
        self.orientation = np.column_stack((wrist_x, np.cross(tool_z, wrist_x), tool_z))
        self._actions = self._run()

    def action(self):
        return next(self._actions, np.zeros(7))

    @staticmethod
    def _rounded_path(points, rounds=2):
        """Round polyline corners with convex corner cutting."""
        path = np.asarray(points, dtype=float)
        for _ in range(rounds):
            rounded = [path[0]]
            for start, end in zip(path[:-1], path[1:]):
                rounded.extend((0.75 * start + 0.25 * end, 0.25 * start + 0.75 * end))
            rounded.append(path[-1])
            path = np.asarray(rounded)
        return path

    def _osc_action(self, velocity_world, grip):
        controller = self.env.robots[0].part_controllers["right"]
        rotation = orientation_error(
            self.orientation,
            self.env.sim.data.site_xmat[self.site].reshape(3, 3),
        )
        if np.linalg.norm(rotation) < 0.004:
            rotation[:] = 0
        if controller.input_ref_frame == "base":
            velocity_world = controller.origin_ori.T @ velocity_world
            rotation = controller.origin_ori.T @ rotation
        return np.r_[np.clip(velocity_world, -1, 1), np.clip(rotation / 0.5, -0.08, 0.08), grip]

    def _follow(
        self,
        waypoints,
        grip,
        *,
        speed=0.14,
        acceleration=0.5,
        position_tolerance=0.002,
        response_scale=1.0,
    ):
        """Track one rounded path without stopping at its internal waypoints."""
        control_freq = self.env.control_freq
        dt = 1.0 / control_freq
        start = self.env.sim.data.site_xpos[self.site].copy()
        path = self._rounded_path([start, *waypoints])
        vectors = np.diff(path, axis=0)
        lengths = np.linalg.norm(vectors, axis=1)
        keep = lengths > 1e-9
        vectors, lengths = vectors[keep], lengths[keep]
        if not len(lengths):
            return
        starts = path[:-1][keep]
        cumulative = np.r_[0.0, np.cumsum(lengths)]
        total = cumulative[-1]
        final = np.asarray(waypoints[-1], dtype=float)

        def sample(distance):
            index = min(np.searchsorted(cumulative, distance, side="right") - 1, len(lengths) - 1)
            fraction = (distance - cumulative[index]) / lengths[index]
            return starts[index] + fraction * vectors[index]

        progress = command_speed = 0.0
        final_correction_sent = False
        stalled = 0
        previous = start
        max_steps = int(np.ceil(total / (speed * dt * response_scale))) * 2 + 2 * control_freq
        for _ in range(max_steps):
            current = self.env.sim.data.site_xpos[self.site].copy()
            fractions = np.clip(np.einsum("ij,ij->i", current - starts, vectors) / lengths**2, 0, 1)
            candidates = cumulative[:-1] + fractions * lengths
            valid = candidates >= progress - 1e-9
            distances = np.linalg.norm(starts + fractions[:, None] * vectors - current, axis=1)
            distances[~valid] = np.inf
            progress = max(progress, candidates[np.argmin(distances)])
            remaining = total - progress
            lookahead = min(total, progress + max(0.012, 0.15 * speed))
            reference = sample(lookahead)
            direction = reference - current
            distance = np.linalg.norm(direction)
            if distance > 0:
                direction /= distance
            final_error = np.linalg.norm(final - current)
            braking_distance = max(remaining, final_error)
            command_speed = min(
                speed,
                command_speed + acceleration * dt,
                np.sqrt(2 * acceleration * braking_distance / response_scale),
            )
            command = direction * min(command_speed, distance / dt)
            rotation = orientation_error(
                self.orientation,
                self.env.sim.data.site_xmat[self.site].reshape(3, 3),
            )
            if remaining < position_tolerance and final_error < position_tolerance and np.linalg.norm(rotation) < 0.02:
                if final_error < 1e-6 or final_correction_sent:
                    return
                command = (final - current) / dt
                final_correction_sent = True
            moved = np.linalg.norm(current - previous)
            stalled = (
                stalled + 1
                if moved < 2e-5
                and final_error > position_tolerance
                and distance > 0.002
                and np.linalg.norm(rotation) < 0.02
                else 0
            )
            if stalled >= control_freq:
                self.abort_requested = True
                return
            previous = current
            yield self._osc_action(command, grip)
        self.abort_requested = True

    def _hold(self, grip, steps):
        """Operate the fingers while holding the current Cartesian pose."""
        target = self.env.sim.data.site_xpos[self.site].copy()
        for _ in range(steps):
            error = target - self.env.sim.data.site_xpos[self.site]
            norm = np.linalg.norm(error)
            if norm > 0.04:
                error *= 0.04 / norm
            yield self._osc_action(error, grip)

    def _run(self):
        env = self.env
        rings = env.get_policy_task_context()["rings"]
        for index, (name, spec) in enumerate(rings.items()):
            body = env.ring_body_ids[name]
            start = env.sim.data.xpos[body].copy()
            peg = env.sim.data.xpos[env.peg_body_id].copy()
            radial = self.orientation[:, 0].copy()
            radial[2] = 0
            eef = env.sim.data.site_xpos[self.site]
            if name == "small":
                # Center the entering pad in the hole; keep the blue ring grasp unchanged.
                pad_names = env.robots[0].gripper["right"].important_geoms
                candidates = [
                    np.r_[eef[:2] - env.sim.data.geom_xpos[env.sim.model.geom_name2id(pad_names[key][0]), :2], 0]
                    for key in ("left_fingerpad", "right_fingerpad")
                ]
            else:
                radius = (spec["outer_radius"] + spec["inner_radius"]) / 2
                candidates = [side * radial * radius for side in (-1, 1)]
                neighbor = env.sim.data.xpos[env.ring_body_ids["small"]]
                clear = [
                    candidate
                    for candidate in candidates
                    if np.linalg.norm(start[:2] + candidate[:2] - neighbor[:2])
                    >= rings["small"]["outer_radius"] + OPEN_FINGER_SWEEP_RADIUS_M
                ]
                candidates = clear or candidates
            offset = min(candidates, key=lambda candidate: np.linalg.norm(start[:2] + candidate[:2] - eef[:2]))
            grasp = start + offset
            lift = 0.21 if np.linalg.norm(start[:2] - peg[:2]) < 0.1 else 0.12

            self.phase = "approach"
            yield from self._follow([grasp + [0, 0, 0.06]], -1, speed=0.41, acceleration=1.02, response_scale=0.2)
            if self.abort_requested:
                return
            self.phase = "descend"
            yield from self._follow(
                [grasp + [0, 0, 0.003]],
                -1,
                speed=0.11,
                acceleration=0.3,
                position_tolerance=0.005,
                response_scale=0.2,
            )
            if self.abort_requested:
                return
            self.phase = "grasp"
            yield from self._hold(1, 5)

            offset = env.sim.data.xpos[body].copy() - env.sim.data.site_xpos[self.site]
            peg = env.sim.data.xpos[env.peg_body_id].copy()
            self.phase = "carry"
            yield from self._follow(
                [
                    grasp + [0, 0, lift],
                    peg + [0, 0, 0.22] - offset,
                    peg + [0, 0, 0.177] - offset,
                ],
                1,
                speed=0.39,
                acceleration=0.93,
                response_scale=0.2,
            )
            if self.abort_requested:
                return
            if env.sim.data.xpos[body][2] < start[2] + 0.08:
                self.abort_requested = True
                return

            self.phase = "release"
            yield from self._hold(-1, 4)
            self.phase = "retreat"
            yield from self._follow(
                [peg + [0, 0, 0.27] - offset], -1, speed=0.39, acceleration=0.93, response_scale=0.2
            )
            if self.abort_requested:
                return

            self.phase = "settle"
            for _ in range(20):
                if env.get_metrics()["success"]["stage"] == index + 1:
                    break
                yield from self._hold(-1, 1)
            if env.get_metrics()["success"]["stage"] != index + 1:
                self.abort_requested = True
                return
        self.phase = "done"
