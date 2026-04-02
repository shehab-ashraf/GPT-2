set -euo pipefail

DATA_DIR="${DATA_DIR:-cache/fineweb-10B}"
TRAIN_SHARDS="${TRAIN_SHARDS:-50}"
TRAIN_DATA_GLOB="${TRAIN_DATA_GLOB:-${DATA_DIR}/fineweb_train_*.bin}"
VAL_DATA_GLOB="${VAL_DATA_GLOB:-${DATA_DIR}/fineweb_val_*.bin}"
SAVE_DIR="${SAVE_DIR:-checkpoints}"
WANDB_PROJECT="${WANDB_PROJECT:-nanogpt}"

ensure_data() {
    shopt -s nullglob
    local shards=(${TRAIN_DATA_GLOB})
    shopt -u nullglob
    if (( ${#shards[@]} > 0 )); then
        return
    fi

    echo "no training shards found under ${DATA_DIR}"
    echo "downloading fineweb subset (${TRAIN_SHARDS} shards) ..."
    python src/data/download.py --shards "${TRAIN_SHARDS}" --output-dir "${DATA_DIR}"
}

run_training() {
    python src/train.py \
        --train_data_dir   "${TRAIN_DATA_GLOB}" \
        --val_data_dir     "${VAL_DATA_GLOB}" \
        --save_dir         "${SAVE_DIR}" \
        --micro_batch      64 \
        --seq_len          2048 \
        --total_batch_size 524288 \
        --total_steps      1500 \
        --warmup_steps     100 \
        --cooldown_steps   900 \
        --target_loss      3.28 \
        --muon_lr          0.02 \
        --adam_lr          0.006 \
        --muon_wd          0.01 \
        --adam_wd          0.1 \
        --log_interval     1 \
        --val_interval     125 \
        --compile \
        --wandb \
        --wandb_project    "${WANDB_PROJECT}" \
        "$@"
}

ensure_data
run_training "$@"
