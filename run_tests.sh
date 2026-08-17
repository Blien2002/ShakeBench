#!/usr/bin/env bash
set -euo pipefail

# shellcheck source=scripts/isaac_env.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/scripts/isaac_env.sh"

# Isaac Lab's venv breaks when unrelated pytest plugins are autoloaded.
: "${PYTEST_DISABLE_PLUGIN_AUTOLOAD:=1}"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD
exec "$VIBENCH_PYTHON" -m pytest -q "$@"
