#!/usr/bin/env bash
set -euo pipefail

# shellcheck source=scripts/isaac_env.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/scripts/isaac_env.sh"

# OpenUSD's parallel scene construction intermittently corrupts memory in the
# usd_core build bundled with this IsaacLab environment.  Keep scene authoring
# deterministic by default while preserving an explicit user override.
export PXR_WORK_THREAD_LIMIT="${PXR_WORK_THREAD_LIMIT:-1}"
exec "$SHAKEBENCH_PYTHON" "$PROJECT_ROOT/scripts/run_demo.py" "$@"
