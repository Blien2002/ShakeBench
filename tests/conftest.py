"""Shared test setup: headless MuJoCo, task registration and per-task initial states."""

import importlib
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "disable")

import pytest  # noqa: E402

TASK_MODULES = {
    "pick_place": "shakebench.environments.vibration_pick_place",
    "panel_operation": "shakebench.environments.panel_operation",
    "push_t": "shakebench.environments.push_t",
    "ring_on_peg": "shakebench.environments.ring_on_peg",
    "ring_on_rail": "shakebench.environments.ring_on_rail",
    "stack_blocks": "shakebench.environments.stack_blocks",
    "upright": "shakebench.environments.upright",
    "wine_rack": "shakebench.environments.wine_rack",
}


def _initial_state(task):
    """The task's default state, or the first pick-place development state."""
    importlib.import_module(TASK_MODULES[task])
    if task == "pick_place":
        from shakebench import models
        from shakebench.tasks.states.assets import load_state_asset

        return load_state_asset(Path(models.assets_root) / "shakebench_states_dev.json")["states"][0]
    module = "shakebench.environments.panel_state" if task == "panel_operation" else TASK_MODULES[task]
    return importlib.import_module(module).default_state()


@pytest.fixture(scope="session", autouse=True)
def register_all_tasks():
    for module in TASK_MODULES.values():
        importlib.import_module(module)


@pytest.fixture(scope="session")
def initial_state():
    return _initial_state
