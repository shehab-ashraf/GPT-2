#!/bin/bash
# Usage: bash scripts/run.sh [extra flags]
# Data must be downloaded first: python -m src.data.download --shards 50
set -euo pipefail

DATA_DIR="${DATA_DIR:-cache/fineweb-10B}"
SAVE_DIR="${SAVE_DIR:-cache/nanogpt-3000}"

python -m src.train \
    --train_data       "${DATA_DIR}/fineweb_train_*.bin" \
    --val_data         "${DATA_DIR}/fineweb_val_*.bin" \
    --save_dir         "${SAVE_DIR}" \
    --micro_batch      64 \
    --seq_len          2048 \
    --total_batch_size 524288 \
    --total_steps      3000 \
    --warmup_steps     200 \
    --cooldown_steps   1800 \
    --target_loss      0.0 \
    --muon_lr          0.02 \
    --adam_lr           0.006 \
    --muon_wd          0.01 \
    --adam_wd           0.1 \
    --log_interval     1 \
    --val_interval     250 \
    --compile \
    --wandb \
    --wandb_project    "${WANDB_PROJECT:-nanogpt}" \
    "$@"
