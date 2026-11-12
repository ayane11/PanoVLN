#!/usr/bin/env bash
# Run inside the container. Starts a NEW joint experiment from PanoVLN.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
OUTPUT="outputs/memvln_joint_4gpu"
PORT=29547
DRY_RUN=()
while (( $# )); do
  case "$1" in
    --dry-run) DRY_RUN=(--dry-run); shift;;
    --output-dir|--master-port)
      (( $# >= 2 )) || { echo "Missing value for $1" >&2; exit 2; }
      if [[ "$1" == --output-dir ]]; then OUTPUT="$2"; else PORT="$2"; fi
      shift 2;;
    *) echo "Unknown argument: $1" >&2; exit 2;;
  esac
done
[[ "$PORT" =~ ^[0-9]+$ ]] && (( PORT >= 1024 && PORT <= 65535 )) || { echo 'Invalid master port' >&2; exit 2; }
printf 'New joint experiment: GPUs=0,1,2,3 batch=24 accumulation=6 W&B=offline output=%s\n' "$OUTPUT"
export WANDB_MODE=offline PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
exec bash scripts/train_memvln.sh --gpus 0,1,2,3 \
  --config src/train/config/memvln.yaml --output-dir "$OUTPUT" \
  --master-port "$PORT" "${DRY_RUN[@]}"
