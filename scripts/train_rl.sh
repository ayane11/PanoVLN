#!/usr/bin/env bash
# 在 dyj_vln 容器内执行：bash /workspace/VLN/PanoVLN/scripts/train_rl.sh
set -euo pipefail

# ==================== 修改这里，无需命令行参数 ====================
EXECUTION_MODE=resident            # 常驻采集/训练，权重通过内存同步
SAVE_FREQ=5                       # 每5轮采集→PPO更新保存，不是每5个Adam minibatch
PARAM_OFFLOAD=true
MEMORY_LIMITS_GIB=(34 26)          # 按 GPUS 顺序：GPU4 / GPU3，GiB
ENV_POOL_SIZE=8
GPUS=(4 3)                         # 用户指定 GPU3、4；主采集进程放在 GPU4
PRECISION=bf16
ATTENTION_BACKEND=flash_attention_2
RUN_NAME=continuous_bf16_aligned_2gpu             # 相同名称自动续训；新名称从 INITIAL_CHECKPOINT 开始
INITIAL_CHECKPOINT=checkpoints/PanoVLN
MAX_UPDATES=0                       # 本实验累计更新轮数；0 表示跑到 EPOCHS 数据耗尽
EPOCHS=20                            # 数据遍历轮数，续训时不能修改
PORT=8027                  # Habitat 服务端口，必须空闲
KEEP_LAST=0                          # 0 保留全部正式 ckpt；正数仅保留最近几个
KEEP_EVERY=0                         # 0 不额外保留；如 5 则永久保留第 5/10/15...轮
KEEP_ROLLOUTS=false                  # false：清理本运行已消费的临时轨迹缓存，保留待导出batch
WANDB_PROJECT=panovln-dapo  # 始终使用 offline 模式
# 如从已有 RL checkpoint 开始，下列训练配置必须与其元数据一致。
MINI_BATCH_DECISIONS=1024            # SimpleVLN 有效值：ppo_mini_batch_size(128) × rollout_n(8)
LEARNING_RATE=5e-7
PPO_EPOCHS=1
ROLLOUT_N=8
TARGET_GROUPS=128
GENERATION_GROUPS=192               # 每次取出的任务数，非GPU并发数
FSDP_SIZE=-1                       # -1 使用全部指定卡；或指定可整除卡数的分片规模
USE_TORCH_COMPILE=true             # 与源一致，仅用于非零 entropy 的熵函数
ENABLE_GRAD_CKPT=true
FORWARD_PREFETCH=false
OPTIMIZER_OFFLOAD=true
OPTIMIZER_OFFLOAD_MODE=parameter    # 逐参数搬运Adam状态，降低共享卡峰值
PPO_MICRO_BATCH_SIZE=4             # 对齐 SimpleVLN Qwen3.5 4B
LOG_PROB_MICRO_BATCH_SIZE=8         # 对齐 SimpleVLN Qwen3.5 4B
INFERENCE_BATCH_SIZE=8             # 原生集中批量推理
INFERENCE_WAIT_MS=2
ROLLOUT_WINDOW=48                   # SimpleVLN rollout调度窗口；与环境池大小分开配置
ROLLOUT_SCHEDULER=sliding
SEED=42
# ================================================================

if (( $# != 0 )); then
    echo '请直接修改脚本顶部配置，不需要传入参数。' >&2
    exit 1
fi
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
PY="/root/miniforge3/envs/panovln-rl/bin/python"
if [[ ! -x "$PY" ]]; then
    printf 'Python 不存在或不可执行：%s\n项目根目录：%s\n' "$PY" "$PROJECT_DIR" >&2
    exit 1
fi
if [[ ! -d /data/vln/scene ]]; then
    echo '场景目录不存在：/data/vln/scene；请检查容器的数据挂载。' >&2
    exit 1
fi
if [[ ! "$RUN_NAME" =~ ^[a-zA-Z0-9_-]+$ ]]; then
    echo 'RUN_NAME 只能包含字母、数字、下划线和连字符。' >&2
    exit 1
fi
if (( ${#GPUS[@]} < 2 )); then
    echo '至少需要两张卡。' >&2
    exit 1
fi
if [[ "$INITIAL_CHECKPOINT" != /* ]]; then
    INITIAL_CHECKPOINT="$PROJECT_DIR/$INITIAL_CHECKPOINT"
fi
ROOT="$PROJECT_DIR/data/dapo"
RUN_DIR="$PROJECT_DIR/outputs/dapo/$RUN_NAME"
ACTOR_GPUS="$(IFS=,; echo "${GPUS[*]}")"
if (( ${#MEMORY_LIMITS_GIB[@]} )); then
    MEMORY_LIMIT_GIB="${MEMORY_LIMITS_GIB[0]}"
elif (( ${#GPUS[@]} == 2 )); then
    MEMORY_LIMIT_GIB=44
else
    MEMORY_LIMIT_GIB=40
fi
EXTRA=()
if (( ${#MEMORY_LIMITS_GIB[@]} )); then
    EXTRA+=(--memory-limits-gib "${MEMORY_LIMITS_GIB[@]}")
fi
case "$EXECUTION_MODE" in
    legacy) ;;
    resident) EXTRA+=(--resident) ;;
    *) echo 'EXECUTION_MODE 必须为 legacy 或 resident' >&2; exit 1 ;;
esac
for setting in PARAM_OFFLOAD USE_TORCH_COMPILE ENABLE_GRAD_CKPT FORWARD_PREFETCH OPTIMIZER_OFFLOAD; do
    case "$setting" in
        PARAM_OFFLOAD) flag=param-offload ;;
        USE_TORCH_COMPILE) flag=use-torch-compile ;;
        ENABLE_GRAD_CKPT) flag=gradient-checkpointing ;;
        FORWARD_PREFETCH) flag=forward-prefetch ;;
        OPTIMIZER_OFFLOAD) flag=optimizer-offload ;;
    esac
    case "${!setting}" in
        true) EXTRA+=("--$flag") ;;
        false) EXTRA+=("--no-$flag") ;;
        *) echo "$setting 必须为 true 或 false" >&2; exit 1 ;;
    esac
done
if [[ "$KEEP_ROLLOUTS" == true ]]; then
    EXTRA+=(--keep-rollouts)
elif [[ "$KEEP_ROLLOUTS" != false ]]; then
    echo 'KEEP_ROLLOUTS 只能为 true 或 false。' >&2
    exit 1
fi
export PYTHONPATH="$PROJECT_DIR/rl:$PROJECT_DIR"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-16}"
export HF_HUB_OFFLINE=1 WANDB_MODE=offline
export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
unset CUDA_VISIBLE_DEVICES
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY='*' no_proxy='*'

# 没有磁盘容量门槛检查；写入异常由训练进程报告，保留上次提交的检查点。
mkdir -p "$RUN_DIR"
LOG="$RUN_DIR/launcher_$(date +%Y%m%d_%H%M%S)_$$.log"
exec > >(tee -a "$LOG") 2>&1
printf '前台连续训练；调度器 PID=%s，Ctrl+C 可中断并保留已提交检查点。\n' "$$"
printf '输入资源：%s\n运行日志：%s\n' "$ROOT" "$LOG"
printf '调度状态：%s/state.json\n' "$RUN_DIR"
if [[ "$EXECUTION_MODE" == resident ]]; then
    printf '最新检查点：%s/resident/latest.json\n逐轮指标：%s/resident/update_v*.json\n' "$RUN_DIR" "$RUN_DIR"
else
    printf '最新检查点：%s/latest.json\n' "$RUN_DIR"
fi
echo '在 tmux 中 detach 可保持运行。当前配置和运行状态将决定新训练或续训。'
exec "$PY" -u scripts/continuous_rl.py --attention-backend "$ATTENTION_BACKEND" --precision "$PRECISION" \
    --project "$PROJECT_DIR" --output "$RUN_DIR" \
    --checkpoint "$INITIAL_CHECKPOINT" \
    --manifest "$ROOT/data/train.parquet" --environment "$ROOT/environment.yaml" \
    --assets-audit "$ROOT/assets_audit/complete.json" \
    --render-audit "$ROOT/render_audit_02/complete.json" \
    --gpus "$ACTOR_GPUS" --port "$PORT" --memory-limit-gib "$MEMORY_LIMIT_GIB" \
    --save-freq "$SAVE_FREQ" --env-pool-size "$ENV_POOL_SIZE" \
    --max-updates "$MAX_UPDATES" --epochs "$EPOCHS" --seed "$SEED" \
    --inference-batch-size "$INFERENCE_BATCH_SIZE" --inference-wait-ms "$INFERENCE_WAIT_MS" \
    --rollout-window "$ROLLOUT_WINDOW" --rollout-scheduler "$ROLLOUT_SCHEDULER" \
    --rollout-n "$ROLLOUT_N" --target-groups "$TARGET_GROUPS" --generation-groups "$GENERATION_GROUPS" \
    --optimizer-offload-mode "$OPTIMIZER_OFFLOAD_MODE" \
    --fsdp-size "$FSDP_SIZE" \
    --ppo-micro-batch-size "$PPO_MICRO_BATCH_SIZE" --log-prob-micro-batch-size "$LOG_PROB_MICRO_BATCH_SIZE" \
    --mini-batch-decisions "$MINI_BATCH_DECISIONS" --lr "$LEARNING_RATE" \
    --ppo-epochs "$PPO_EPOCHS" --keep-last "$KEEP_LAST" --keep-every "$KEEP_EVERY" \
    --wandb-project "$WANDB_PROJECT" "${EXTRA[@]}"
