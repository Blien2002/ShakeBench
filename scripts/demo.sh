#!/usr/bin/env bash
set -euo pipefail

# Safe, quick ShakeBench demo: Gamma=0.15 contact-preserving 6-DOF spectral
# vibration on the 240 Hz training profile.  This is a visual demo, not an
# official score run.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

exec "${PROJECT_ROOT}/run.sh" --record \
  --physics-profile training \
  --episode-s 6 \
  --vibration spectral \
  --gamma 0.15 \
  --seed 17 \
  --workpiece sugar_box \
  --workpiece-scale 0.75 \
  --output "${PROJECT_ROOT}/out/shakebench_demo_safe.mp4" \
  --metrics-output "${PROJECT_ROOT}/out/shakebench_demo_safe.json" \
  "$@"
