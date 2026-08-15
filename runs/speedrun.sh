set -euo pipefail
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

DATA_DIR="${DATA_DIR:-cache/fineweb-10B}"
OUT_DIR="${OUT_DIR:-out}"
NPROC="${NPROC:-1}"
TOTAL_STEPS="${TOTAL_STEPS:-3000}"
MICRO_BATCH="${MICRO_BATCH:-32}"
RUN_NAME="${RUN_NAME:-nanogpt-${TOTAL_STEPS}}"
RESUME="${RESUME:-0}"
DOWNLOAD_SHARDS="${DOWNLOAD_SHARDS:-50}"
SAVE_DIR="${OUT_DIR}/${RUN_NAME}"

if [[ "${NPROC}" == "2" ]]; then
    WARMUP_STEPS_DEF=$(( TOTAL_STEPS / 40 < 100 ? 100 : TOTAL_STEPS / 40 ))
else
    WARMUP_STEPS_DEF=$(( TOTAL_STEPS / 30 < 100 ? 100 : TOTAL_STEPS / 30 ))
fi
WARMUP_STEPS="${WARMUP_STEPS:-${WARMUP_STEPS_DEF}}"
COOLDOWN_STEPS="${COOLDOWN_STEPS:-$(( TOTAL_STEPS < 4000 ? TOTAL_STEPS * 3 / 10 : TOTAL_STEPS * 6 / 10 ))}"

if [[ "${SKIP_SYNC:-0}" != "1" ]]; then
    uv sync
fi
if [[ "${SKIP_DATA_DOWNLOAD:-0}" != "1" ]]; then
    uv run python -m src.data.download --shards "${DOWNLOAD_SHARDS}" --output-dir "${DATA_DIR}"
fi
mkdir -p "${SAVE_DIR}"

TRAIN_ARGS=(
    --run              "${WANDB_RUN:-dummy}"
    --wandb-project    "${WANDB_PROJECT:-nanogpt}"
    --train-data       "${DATA_DIR}/fineweb_train_*.bin"
    --val-data         "${DATA_DIR}/fineweb_val_*.bin"
    --save-dir         "${SAVE_DIR}"
    --device-batch-size "${MICRO_BATCH}"
    --max-seq-len      2048
    --total-batch-size 524288
    --num-iterations   "${TOTAL_STEPS}"
    --warmup-steps     "${WARMUP_STEPS}"
    --cooldown-steps   "${COOLDOWN_STEPS}"
    --final-lr-frac    0.0
    --matrix-lr        0.02
    --muon-momentum    0.95
    --embedding-lr     0.0036
    --adam-beta1       0.9
    --adam-beta2       0.95
    --weight-decay     0.01
    --adam-wd          0.1
    --dtype            bfloat16
    --seed             42
    --log-interval     1
    --eval-every       250
    --eval-tokens      10485760
    --save-every       1000
    --compile-mode     auto
    --bucket-cap-mb    50
)

if [[ "${RESUME}" == "1" ]]; then
    TRAIN_ARGS+=(--resume)
fi

if [[ "${NPROC}" == "1" ]]; then
    exec uv run python -m src.train "${TRAIN_ARGS[@]}" "$@"
else
    exec uv run torchrun --standalone --nproc_per_node="${NPROC}" -m src.train "${TRAIN_ARGS[@]}" "$@"
fi
