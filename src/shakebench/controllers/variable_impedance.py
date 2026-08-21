"""Equilibrium pose, diagonal stiffness, and grip-force controller."""

from __future__ import annotations

import numpy as np

from .base import Controller, ControllerOutput


class VariableImpedanceController(Controller):
    action_dim = 13

    def pre_action(self, action, policy_step, *, alpha=1.0, phase=0.0) -> ControllerOutput:
        output = super().pre_action(action, policy_step, alpha=alpha, phase=phase)
        normalized = output.command
        command = normalized.copy()
        command[:6] *= np.asarray(self.config["equilibrium_output_max"], dtype=np.float32)
        lower = np.asarray(self.config["stiffness_min"], dtype=np.float32)
        upper = np.asarray(self.config["stiffness_max"], dtype=np.float32)
        command[6:12] = lower + 0.5 * (normalized[6:12] + 1.0) * (upper - lower)
        command[12] = 35.0 * (normalized[12] + 1.0)
        if self.config.get("intra_step_mode") == "feedforward":
            command[12] = np.clip(command[12] * (1.0 + 0.1 * np.sin(phase)), 0.0, 70.0)
        return ControllerOutput(command.astype(np.float32), output.clipped)
