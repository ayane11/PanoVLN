#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
GPU_DEVICES=""
CONFIG="src/train/config/memvln.yaml"
OUTPUT=""
PORT=29546
RESUME=""
INITIAL=""
PYTHON="${PANOVLN_PYTHON:-/root/miniforge3/envs/panovln/bin/python}"
DRY_RUN=0
while (( $# )); do
  case "$1" in
    --gpus) GPU_DEVICES="$2"; shift 2;;
    --config) CONFIG="$2"; shift 2;;
    --output-dir) OUTPUT="$2"; shift 2;;
    --master-port) PORT="$2"; shift 2;;
    --resume) RESUME="$2"; shift 2;;
    --init-from) INITIAL="$2"; shift 2;;
    --dry-run) DRY_RUN=1; shift;;
    *) echo "Unknown argument: $1" >&2; exit 2;;
  esac
done
[[ "$GPU_DEVICES" =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo 'Supply --gpus with explicit physical GPU indices' >&2; exit 2; }
[[ -n "$OUTPUT" ]] || { echo 'Supply a unique --output-dir' >&2; exit 2; }
IFS=',' read -ra GPU_ARRAY <<< "$GPU_DEVICES"
GPU_COUNT="${#GPU_ARRAY[@]}"
[[ "$(printf '%s\n' "${GPU_ARRAY[@]}" | sort -u | wc -l)" -eq "$GPU_COUNT" ]] || { echo 'Duplicate GPU index' >&2; exit 2; }
export CUDA_VISIBLE_DEVICES="$GPU_DEVICES"
export WANDB_MODE=offline
export WANDB_DIR="$OUTPUT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export NCCL_NVLS_ENABLE=0
export PYTHONUNBUFFERED=1
ARGS=(--config "$CONFIG" --output-dir "$OUTPUT")
[[ -z "$RESUME" ]] || ARGS+=(--resume "$RESUME")
[[ -z "$INITIAL" ]] || ARGS+=(--init-from "$INITIAL")
if (( DRY_RUN )); then
  # Metadata-only validation must never initialize CUDA.
  CUDA_VISIBLE_DEVICES= "$PYTHON" -m src.train.train_memvln "${ARGS[@]}" --validate-only --world-size "$GPU_COUNT"
  exit
fi
exec "$PYTHON" -m torch.distributed.run --nproc_per_node "$GPU_COUNT" --master_port "$PORT" \
  --module src.train.train_memvln "${ARGS[@]}"
