"""Declared interpreter compatibility smoke for the official loader."""

from __future__ import annotations

import sys


def test_declared_python_minimum_and_official_imports():
    assert sys.version_info >= (3, 10)
    from robosuite.scripts import shakebench_handoff_remediation  # noqa: F401
    from robosuite.utils import shakebench_physics  # noqa: F401
