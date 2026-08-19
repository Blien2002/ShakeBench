#!/usr/bin/env python3
"""Thin launcher for the ViBench demo CLI.

The full command-line implementation lives in ``vibench.cli`` so the same
entry point is available as a console script (``vibench``), as
``python -m vibench``, and through this repository-local script.
"""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vibench.cli import main  # noqa: E402  (intentional after src path bootstrap)


if __name__ == "__main__":
    raise SystemExit(main())
