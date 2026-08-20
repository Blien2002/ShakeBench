#!/usr/bin/env bash
set -euo pipefail

# Generic runner for repository tools/probes that need the Isaac Lab venv.
# Example: ./run_python.sh tools/visual_audit.py out/frame.png
#
# shellcheck source=scripts/isaac_env.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/scripts/isaac_env.sh"

# Match run.sh: the bundled OpenUSD build can corrupt memory while authoring
# large meshes in parallel.  Keep tool/probe scene construction deterministic
# unless the caller explicitly supplies a different limit.
export PXR_WORK_THREAD_LIMIT="${PXR_WORK_THREAD_LIMIT:-1}"
exec "$VIBENCH_PYTHON" "$@"
