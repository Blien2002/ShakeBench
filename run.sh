#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAACLAB_ROOT="${ISAACLAB_ROOT:-/home/miracle04/IsaacLab-3.0}"

# OpenUSD's parallel scene construction intermittently corrupts memory in the
# usd_core build bundled with this IsaacLab environment.  Keep scene authoring
# deterministic by default while preserving an explicit user override.
export PXR_WORK_THREAD_LIMIT="${PXR_WORK_THREAD_LIMIT:-1}"
export PYTHONPATH="$PROJECT_ROOT/src:$ISAACLAB_ROOT/source/isaaclab_assets${PYTHONPATH:+:$PYTHONPATH}"
exec "$ISAACLAB_ROOT/.venv/bin/python" "$PROJECT_ROOT/scripts/run_demo.py" "$@"
