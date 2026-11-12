#!/usr/bin/env bash
# Single physical GPU, one worker. Dry-run prints the exact command without Python/CUDA.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
PYTHON="${PANOVLN_PYTHON:-/root/miniforge3/envs/panovln/bin/python}"
GPU="" MODEL="" OUTPUT="" LIMIT=""
CONFIG="config/vln_r2r_mem_dev.yaml"
SEED=42
DRY_RUN=0
while (( $# )); do
  case "$1" in
    --dry-run) DRY_RUN=1; shift; continue;;
    --gpu|--model-path|--output-dir|--config|--max-episodes|--seed)
      (( $# >= 2 )) || { echo "Missing value for $1" >&2; exit 2; };;
    *) echo "Unknown argument: $1" >&2; exit 2;;
  esac
  case "$1" in
    --gpu) GPU="$2";;
    --model-path) MODEL="$2";;
    --output-dir) OUTPUT="$2";;
    --config) CONFIG="$2";;
    --max-episodes) LIMIT="$2";;
    --seed) SEED="$2";;
  esac
  shift 2
done
[[ "$GPU" =~ ^[0-9]+$ ]] || { echo 'Supply --gpu with one physical GPU index' >&2; exit 2; }
[[ "$LIMIT" =~ ^[0-9]+$ ]] || { echo 'Supply --max-episodes explicitly (0 means all)' >&2; exit 2; }
[[ "$SEED" =~ ^[0-9]+$ ]] || { echo 'Invalid seed' >&2; exit 2; }
[[ -n "$MODEL" && -n "$OUTPUT" ]] || { echo 'Supply --model-path and a new --output-dir' >&2; exit 2; }
[[ -f "$CONFIG" ]] || { echo "Missing config: $CONFIG" >&2; exit 2; }
# A fresh directory prevents silently mixing checkpoints/conditions via episode skipping.
[[ ! -e "$OUTPUT" ]] || { echo "Output already exists: $OUTPUT; choose a fresh directory" >&2; exit 2; }
ARGS=(src/eval/eval.py --exp-config "$CONFIG" --split-num 1 --split-id 0
  --model-path "$MODEL" --result-path "$OUTPUT" --total-max-episodes "$LIMIT"
  --forward-distance 25 --turn-angle 15 --max-memory-images 10
  --memory-pool-window-frames 100 --actions-per-replan uncertainty --uncertainty-budget 1.2
  --replan-action-range 4 8 --stop-commit-max-actions 10 --collision-recovery-steps 2
  --save-topdown false --attn-implementation flash_attention_2 --seed "$SEED")
if (( DRY_RUN )); then
  printf 'CUDA_VISIBLE_DEVICES=%q WANDB_MODE=offline ' "$GPU"
  printf '%q ' "$PYTHON" "${ARGS[@]}"
  printf '\n'
  exit 0
fi
[[ -d "$MODEL" ]] || { echo "Missing model: $MODEL" >&2; exit 2; }
[[ -x "$PYTHON" ]] || { echo "Missing Python: $PYTHON" >&2; exit 2; }
export CUDA_VISIBLE_DEVICES="$GPU" WANDB_MODE=offline PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
export VLN_EVAL_CPU_THREADS_PER_WORKER=4
"$PYTHON" "${ARGS[@]}"
# Aggregation needs no GPU. Keep per-episode metrics for paired comparisons.
CUDA_VISIBLE_DEVICES= "$PYTHON" src/eval/analyze_results.py --path "$OUTPUT"
