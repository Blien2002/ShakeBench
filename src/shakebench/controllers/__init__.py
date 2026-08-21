"""Public controller factory."""

from __future__ import annotations

from .base import Controller
from .joint_position import JointPositionController
from .osc_pose import OSCPoseController
from .variable_impedance import VariableImpedanceController

_CONTROLLERS = {
    "JOINT_POSITION": JointPositionController,
    "OSC_POSE": OSCPoseController,
    "VARIABLE_IMPEDANCE": VariableImpedanceController,
}


def make_controller(config: dict) -> Controller:
    name = str(config.get("name", "JOINT_POSITION")).upper()
    try:
        return _CONTROLLERS[name](config)
    except KeyError as exc:
        raise ValueError(f"unknown controller {name!r}; available={sorted(_CONTROLLERS)}") from exc


__all__ = [
    "Controller",
    "JointPositionController",
    "OSCPoseController",
    "VariableImpedanceController",
    "make_controller",
]
