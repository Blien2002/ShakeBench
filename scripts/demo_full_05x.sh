#!/usr/bin/env bash
set -euo pipefail

# Complete-task demo with the 6-DOF spectrum scaled to 0.5x.
#
# This uses the 240 Hz training profile with 8 solver substeps and enables
# the disclosed grasp-assist stabilizer, so it is a visual demonstration and
# NOT an official score run.  Metrics record grasp_assist_used=true.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

exec "${PROJECT_ROOT}/run.sh" --record \
  --physics-profile training \
  --solver-substeps 8 \
  --episode-s 16 \
  --vibration spectral \
  --spectral-scale 0.5 \
  --seed 17 \
  --workpiece sugar_box \
  --workpiece-scale 0.75 \
  --grasp-assist \
  --output "${PROJECT_ROOT}/out/vibench_demo_full_05x_assist.mp4" \
  --metrics-output "${PROJECT_ROOT}/out/vibench_demo_full_05x_assist.json" \
  "$@"
