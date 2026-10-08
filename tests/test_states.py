"""Every shipped state asset authenticates, and its first state validates through its task."""

from pathlib import Path

import pytest

from shakebench import models
from shakebench.tasks.registry import prepare_task_state
from shakebench.tasks.states.assets import load_state_asset

ASSETS = sorted(path.name for path in Path(models.assets_root).glob("shakebench_*states*.json"))


def test_state_assets_are_shipped():
    assert len(ASSETS) >= 13


@pytest.mark.parametrize("name", ASSETS)
def test_state_asset_loads(name):
    states = load_state_asset(Path(models.assets_root) / name)["states"]
    identifiers = [state["state_id"] for state in states]
    assert identifiers and len(set(identifiers)) == len(identifiers)
    prepare_task_state(states[0])
