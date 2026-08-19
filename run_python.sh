#!/usr/bin/env bash
set -euo pipefail

# Generic runner for repository tools/probes that need the Isaac Lab venv.
# Example: ./run_python.sh tools/visual_audit.py out/frame.png
#
# shellcheck source=scripts/isaac_env.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/scripts/isaac_env.sh"

exec "$VIBENCH_PYTHON" "$@"
