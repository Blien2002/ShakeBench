#!/usr/bin/env bash
# ShakeBench gamma=0 oracle SFT on StarVLA (QwenOFT + Qwen3-VL-2B-Instruct).
#
# Run from the starVLA repo root (or set STARVLA_ROOT):
#   bash examples/ShakeBench/train_files/run_shakebench_train.sh
#
# Prereqs (already prepared on this host):
#   - data:     playground/Datasets/ShakeBench/oracle-gamma-zero (success-only SFT subset)
#   - model:    playground/Pretrained_models/Qwen3-VL-2B-Instruct
#   - registry: examples/ShakeBench (links ShakeBench/integrations/starvla)
set -euo pipefail
cd "${STARVLA_ROOT:-$PWD}"
if [[ ! -f starVLA/training/train_starvla.py ]]; then
  echo "ERROR: run from the starVLA repo root or set STARVLA_ROOT" >&2
  exit 1
fi

# Activate the starvla conda env if it is not already active.
if [[ "${CONDA_DEFAULT_ENV:-}" != "starvla" ]]; then
  source /home/cyx/miniforge3/etc/profile.d/conda.sh
  conda activate starvla
fi

# 4090 boxes need P2P/IB disabled or NCCL dies during DeepSpeed init.
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export WANDB_MODE=disabled

run_root_dir=playground/Checkpoints
run_id=shakebench_qwenoft_2b_100eps
config_yaml=examples/ShakeBench/train_files/starvla_qwenoft_shakebench.yaml
num_processes=${NUM_PROCESSES:-8}

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes "${num_processes}" \
  --main_process_port "${MASTER_PORT:-29500}" \
  starVLA/training/train_starvla.py \
  --config_yaml "${config_yaml}" \
  --run_root_dir "${run_root_dir}" \
  --run_id "${run_id}"
