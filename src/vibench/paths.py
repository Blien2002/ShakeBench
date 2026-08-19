"""Repository and installed-resource paths for ViBench.

ViBench is self-contained: configs, textures and generated outputs are
resolved relative to the project itself, never relative to neighbouring
directories on the machine.

Resolution order:
1. ``VIBENCH_ROOT`` environment variable, when set.
2. The source checkout that contains this module (normal ``./run.sh`` and
   editable ``pip install -e .`` usage).
3. ``<prefix>/share/vibench`` for a regular wheel installation.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _is_project_root(path: Path) -> bool:
    return (path / "configs" / "scenarios.yaml").is_file() and (path / "assets" / "textures").is_dir()


def _resolve_project_root() -> Path:
    override = os.environ.get("VIBENCH_ROOT")
    if override:
        candidate = Path(override).expanduser().resolve()
        if _is_project_root(candidate):
            return candidate
        raise RuntimeError(
            f"VIBENCH_ROOT={override!r} does not contain ViBench's configs/ and assets/ directories"
        )

    source_checkout = Path(__file__).resolve().parents[2]
    if _is_project_root(source_checkout):
        return source_checkout

    installed_data = Path(sys.prefix) / "share" / "vibench"
    if _is_project_root(installed_data):
        return installed_data

    raise RuntimeError(
        "ViBench resources were not found. Run the benchmark from its own project "
        "root (./run.sh), install it as editable, or set VIBENCH_ROOT to the checkout."
    )


PROJECT_ROOT = _resolve_project_root()


def project_path(*parts: str) -> Path:
    """Resolve a path relative to the ViBench project root."""

    return PROJECT_ROOT.joinpath(*parts)
