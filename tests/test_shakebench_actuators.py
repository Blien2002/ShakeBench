"""Environment-backed OSC/gripper positive controls."""

from __future__ import annotations

from robosuite.scripts.shakebench_verify_actuators import run_actuator_positive_controls


def test_official_environment_has_all_fourteen_action_positive_controls(tmp_path):
    result = run_actuator_positive_controls(tmp_path / "actuator_controls.json")
    assert result["passed"]
    assert len(result["results"]) == 14
