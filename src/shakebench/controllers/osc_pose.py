"""Operational-space delta-pose controller interface."""

from __future__ import annotations

import numpy as np

from .base import Controller, ControllerOutput


class OSCPoseController(Controller):
    action_dim = 7

    def pre_action(self, action, policy_step, *, alpha=1.0, phase=0.0) -> ControllerOutput:
        output = super().pre_action(action, policy_step, alpha=alpha, phase=phase)
        command = output.command.copy()
        output_max = np.asarray(self.config["output_max"], dtype=np.float32)
        command[:6] *= output_max
        return ControllerOutput(command, output.clipped)
