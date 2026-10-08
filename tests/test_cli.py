"""Command-line entry points start and print their help."""

import subprocess
import sys

import pytest

ENTRY_POINTS = [
    "shakebench.scripts.evaluate",
    "shakebench.scripts.evaluate_async",
    "shakebench.scripts.eval_websocket_policy",
    "shakebench.demos.demo_policy_video",
]


@pytest.mark.parametrize("module", ENTRY_POINTS)
def test_help(module):
    result = subprocess.run([sys.executable, "-m", module, "--help"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-2000:]
    assert "usage:" in result.stdout
