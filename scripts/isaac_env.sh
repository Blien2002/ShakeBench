#!/usr/bin/env bash
# Sourced by ViBench launcher scripts.  Locates the external Isaac Lab
# environment without depending on any other folder on this Desktop or on a
# hardcoded user home path.
#
# The repository itself remains self-contained; Isaac Lab is an optional
# runtime backend that must be installed separately.  Override its location
# with ISAACLAB_ROOT when it is not in the default location:
#   ISAACLAB_ROOT=/path/to/IsaacLab-3.0 ./run.sh

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -z "${ISAACLAB_ROOT:-}" ]]; then
    if [[ -x "$HOME/IsaacLab-3.0/.venv/bin/python" ]]; then
        ISAACLAB_ROOT="$HOME/IsaacLab-3.0"
    else
        echo "ViBench: Isaac Lab environment not found." >&2
        echo "Set ISAACLAB_ROOT to an Isaac Lab checkout whose .venv/bin/python exists, for example:" >&2
        echo "  export ISAACLAB_ROOT=/path/to/IsaacLab-3.0" >&2
        exit 1
    fi
fi

if [[ ! -x "$ISAACLAB_ROOT/.venv/bin/python" ]]; then
    echo "ViBench: invalid ISAACLAB_ROOT=$ISAACLAB_ROOT (expected $ISAACLAB_ROOT/.venv/bin/python)" >&2
    exit 1
fi

VIBENCH_PYTHON="$ISAACLAB_ROOT/.venv/bin/python"
export PYTHONPATH="$PROJECT_ROOT/src:$ISAACLAB_ROOT/source/isaaclab_assets:${PYTHONPATH:-}"
