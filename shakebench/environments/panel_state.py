"""Versioned panel starts and ordered control actions."""

from collections.abc import Mapping
from copy import deepcopy
from numbers import Real

import numpy as np

from shakebench.models.arenas.shakebench_arena import CANONICAL_WORKTABLE_DIMENSIONS_M
from shakebench.scene.geometry import load_geometry_profile

TASK_TYPE = "panel_operation"
TASK_VERSION = 1
STATE_SCHEMA = "shakebench.panel_operation.states"
SCHEMA_VERSION = 1
STEPS = ("press", "toggle_up", "toggle_down", "turn_on", "turn_off")
STEP_CONTROL = {
    "press": "button",
    "toggle_up": "lever",
    "toggle_down": "lever",
    "turn_on": "knob",
    "turn_off": "knob",
}
INSTRUCTIONS = {
    "press": "Press the push button.",
    "toggle_up": "Flip the toggle switch up.",
    "toggle_down": "Flip the toggle switch down.",
    "turn_on": "Turn the knob on.",
    "turn_off": "Turn the knob off.",
}
# Keep previously generated states valid when narrowing the training sampler.
PANEL_STATE_XY_MIN_M = (-0.17, -0.145)
PANEL_STATE_XY_MAX_M = (0.10, 0.145)
PANEL_EDGE_CLEARANCE_M = 0.06
PANEL_YAW_LIMIT_RAD = np.deg2rad(20)
PANEL_HALF_FOOTPRINT_M = np.array([0.0663, 0.0952])
SUCCESS_HOLD_S = 0.5
ARM_LOWER_RAD = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
ARM_UPPER_RAD = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
ARM_START_LABELS = ("home", "near_knob", "near_lever", "near_button", "near_panel")


def default_state(step="press"):
    """Return a deterministic valid start for one panel action."""
    if step not in STEPS:
        raise ValueError(f"step must be one of {STEPS}")
    controls = {"knob_on": False, "lever_up": False, "button_on": False}
    if step == "turn_off":
        controls["knob_on"] = True
    if step == "toggle_down":
        controls["lever_up"] = True
    return {
        "state_id": f"panel-{step}-000",
        "task": {"task_type": TASK_TYPE, "version": TASK_VERSION},
        "panel_xy_m": [0.0, 0.0],
        "panel_yaw_rad": 0.0,
        "controls": controls,
        "steps": [step],
        "arm_qpos_rad": load_geometry_profile()["initial_joint_qpos_rad"],
        "arm_start_label": "home",
        "gripper_open": True,
        "excitation_seed": 0,
        "imu_seed": 0,
        "t0_s": 0.0,
    }


def validate_state(state):
    """Validate panel pose, arm pose and an ordered nonrepeating action chain."""
    required = set(default_state())
    if not isinstance(state, Mapping) or not required <= set(state) or set(state) - required - {"split"}:
        raise ValueError("panel state fields must match version 1")
    task = state["task"]
    if (
        not isinstance(task, Mapping)
        or dict(task) != {"task_type": TASK_TYPE, "version": TASK_VERSION}
        or type(task["version"]) is not int
    ):
        raise ValueError("expected panel_operation task version 1")
    if not isinstance(state["state_id"], str) or not state["state_id"].strip():
        raise ValueError("state_id must be nonempty")
    if "split" in state and state["split"] not in ("train", "eval"):
        raise ValueError("split must be train or eval")
    steps = state["steps"]
    if (
        not isinstance(steps, list)
        or not 1 <= len(steps) <= 3
        or any(not isinstance(step, str) or step not in STEPS for step in steps)
    ):
        raise ValueError("steps must contain one to three known actions")
    if len({STEP_CONTROL[step] for step in steps}) != len(steps):
        raise ValueError("each control may appear only once")
    controls = state["controls"]
    if not isinstance(controls, Mapping) or set(controls) != {"knob_on", "lever_up", "button_on"}:
        raise ValueError("controls must contain knob_on, lever_up and button_on")
    if any(type(value) is not bool for value in controls.values()):
        raise ValueError("control states must be boolean")
    future = dict(controls)
    for step in steps:
        key = {"button": "button_on", "lever": "lever_up", "knob": "knob_on"}[STEP_CONTROL[step]]
        expected = {"toggle_up": False, "toggle_down": True, "turn_on": False, "turn_off": True}.get(step)
        if expected is not None and future[key] != expected:
            raise ValueError("step requires the opposite initial control state")
        future[key] = not future[key]
    for key, size in (("panel_xy_m", 2), ("arm_qpos_rad", 7)):
        values = state[key]
        if (
            not isinstance(values, (list, tuple, np.ndarray))
            or np.shape(values) != (size,)
            or any(type(value) is bool or not isinstance(value, Real) or not np.isfinite(value) for value in values)
        ):
            raise ValueError(f"{key} must contain {size} finite numbers")
    xy = np.asarray(state["panel_xy_m"], dtype=float)
    arm = np.asarray(state["arm_qpos_rad"], dtype=float)
    if np.any(xy < PANEL_STATE_XY_MIN_M) or np.any(xy > PANEL_STATE_XY_MAX_M):
        raise ValueError("panel_xy_m outside sampling bounds")
    if np.any(arm < ARM_LOWER_RAD) or np.any(arm > ARM_UPPER_RAD):
        raise ValueError("arm_qpos_rad outside Panda joint limits")
    for key in ("panel_yaw_rad", "t0_s"):
        value = state[key]
        if type(value) is bool or not isinstance(value, Real) or not np.isfinite(value):
            raise ValueError(f"{key} must be finite")
    yaw = float(state["panel_yaw_rad"])
    if abs(yaw) > PANEL_YAW_LIMIT_RAD or state["t0_s"] < 0:
        raise ValueError("panel_yaw_rad or t0_s outside bounds")
    c, s = np.cos(yaw), np.sin(yaw)
    half_extent = np.abs([[c, -s], [s, c]]) @ PANEL_HALF_FOOTPRINT_M
    if np.any(
        np.abs(xy) + half_extent > np.asarray(CANONICAL_WORKTABLE_DIMENSIONS_M[:2]) / 2 - PANEL_EDGE_CLEARANCE_M + 1e-12
    ):
        raise ValueError("panel footprint must keep 6 cm clearance from the worktable edge")
    if state["arm_start_label"] not in ARM_START_LABELS:
        raise ValueError("invalid arm_start_label")
    if type(state["gripper_open"]) is not bool:
        raise ValueError("gripper_open must be boolean")
    for key in ("excitation_seed", "imu_seed"):
        if type(state[key]) is not int or not 0 <= state[key] < 2**32:
            raise ValueError(f"{key} must be an integer in [0, 2**32)")
    result = deepcopy(dict(state))
    result.update(panel_xy_m=xy.tolist(), panel_yaw_rad=yaw, arm_qpos_rad=arm.tolist(), t0_s=float(state["t0_s"]))
    return result


def load_states(payload):
    """Load an authenticated panel state asset."""
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema_id") != STATE_SCHEMA
        or type(payload.get("schema_version")) is not int
        or payload["schema_version"] != SCHEMA_VERSION
    ):
        raise ValueError("unsupported panel state schema/version")
    if not isinstance(payload.get("states"), list) or not payload["states"]:
        raise ValueError("states must be a nonempty list")
    states = [validate_state(state) for state in payload["states"]]
    if len({state["state_id"] for state in states}) != len(states):
        raise ValueError("duplicate state_id")
    return {"states": states}
