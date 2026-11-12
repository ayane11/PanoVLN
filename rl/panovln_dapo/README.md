# PanoVLN DAPO adapters (in progress)

These adapters migrate the SimpleVLN RL recipe to the native Pano model. A full
real-reward batch has completed four optimizer steps and exported a checkpoint
using FP32 and CPU optimizer moment offload. Original/exported processors match
on real first-frame and long-history inputs. Full weight audit passed and the
exported model completed matched 64-episode development evaluation: SR was
unchanged and SPL increased by 0.1883 percentage points, with a paired interval
crossing zero. Distributed optimizer resume and reproducible navigation gains
remain separate acceptance gates.
Multi-round and resume entry points exist but their GPU integration is not yet
accepted. Failed earlier attempts and current evidence are recorded in the audit.


Run inside `dyj_vln`, from `/workspace/VLN/PanoVLN`:

```bash
conda activate panovln-rl
export PYTHONPATH="$PWD/rl:$PWD"
CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 \
  python -m pytest rl/panovln_dapo/tests scripts/test_continuous_rl.py -q -p no:cacheprovider
```

The `panovln-rl` Conda environment is cloned from `panovln`, with the pinned
additions in `requirements-cpu.txt`. The original `panovln` environment remains
unchanged. NumPy/PyTorch/Transformers/Habitat versions are preserved. The training
launcher explicitly uses `/root/miniforge3/envs/panovln-rl/bin/python`, independent
of the environment activated in the terminal.

Start continuous training in tmux with `bash scripts/train_rl.sh`. It runs in
the foreground, records W&B offline, reads inputs from `data/dapo`, and stores
checkpoints/resume state under `outputs/dapo_continuous/<RUN_NAME>`. Reusing the
same run name resumes its committed state. The isolated single-batch W&B wrapper
has been retired; the continuous driver handles logging directly.

Create a fresh manifest (existing output is refused):

```bash
python -m panovln_dapo.data \
  --annotation /data/vln/general_vln_dataset/r2r/train/train.json.gz \
  --dataset-name r2r --split train --dataset-type R2RVLNCE-v1 \
  --output outputs/dapo_migration_v1/manifests/train.parquet
```

Each row retains the entire episode and source metadata. For native Habitat
loading, partition by source, then call `restore_annotation(rows)`. Do not use the
SimpleVLN materializer unchanged: it drops R2R's required instruction vocabulary.
`build_inputs` in `inputs.py` is shared by sampling and actor replay.
It rejects over-budget prompts rather than truncating visual inputs.

For actual run state, inspect `outputs/dapo_continuous/<RUN_NAME>/state.json`
and `latest.json`. Launcher settings are in `scripts/train_rl.sh`; the durable
multi-round controller is `scripts/continuous_rl.py`. Historical validation
results described above are not evidence that a new run has completed.


## Entry points and current protocol

- `collect`: one configurable on-policy round; resume episode cursor/version from
  a trained checkpoint. `collect_pilot` is the original fixed pilot, not the
  multi-round training interface.
- `run_actor`: validate a batch, check the requested GPUs, snapshot sources and
  supervise the FSDP `actor`. AdamW moments move to CPU between optimizer steps.
- `train_loop`: sequential collection/update/export-audit rounds. Uses a separate,
  already-running training environment. Records durable stage state, preserves
  failed attempts, and checks live processes before a driver resume. Retains all
  checkpoints; automatic retention is still pending. Optional periodic evaluation
  is CPU-tested, with GPU integration still pending.
- `checkpoint_audit`: compare exported tensors, geometry preservation and cursor
  checksum; this does not replace forward/resume/evaluation checks.
- `evaluate` / `compare`: complete deterministic evaluation and matched-episode
  comparisons. Failed, incomplete, duplicated or incompatible runs are rejected.
- `after_update_eval`: supervise one explicitly identified live actor, then audit
  and evaluate its export. Can start/close its own dedicated evaluation service.

The present protocol is FP32/SDPA, fixed K=4, stop commit=10, 500 environment
steps and 500 decisions, collision recovery disabled. It differs from the native
Pano evaluator defaults; comparisons require the same protocol. BF16 probability
alignment has not met the current gate. The first new 64-episode val_seen baseline
is a development set, not independent evidence of unseen-scene improvement.

RxR manifests can explicitly select `--languages en-US en-IN` from guide source
files. Only the source's optional timed-instruction start/end NaNs are preserved
in embedded JSON strings; geometry and all other metadata remain finite. Native
RxR loader and Parquet round trips were checked for English guide splits.

For a later GPU-validated multi-round run, invoke `train_loop --help` and provide
an immutable initial checkpoint, manifest, dedicated server and fresh output.
A checkpoint descended from the initial pilot requires `--epochs 1`, matching
its saved cursor budget. `--resume-driver` requires the same configuration and
source hashes; a source fix requires a new run from the last validated checkpoint.
Do not treat `training_budget_complete` as a score-improvement claim.

The multi-round driver forwards `--temperature`, `--max-response-tokens`,
`--max-env-steps`, `--max-decisions`, and `--progress-coef` to collection and
`--ppo-epochs` to updates. These settings are immutable across driver resume.
Nondefault forwarding is CPU-tested; actual GPU acceptance still uses the
protocol reported above.

Enable periodic development evaluation with `--eval-baseline <complete-run>`,
`--eval-manifest <matching-parquet>`, `--eval-server <dedicated-validation-url>`,
and `--eval-every N`. Evaluation uses the rollout GPU after the actor exits;
its server must already be running on a separate available device. The driver
checks the baseline manifest hash, keeps baseline content hashes immutable,
and uses the baseline's episode limit. Complete paired comparisons are saved
as `round_XXXX/comparison.json`. The last budgeted update (and a valid update
that exhausts data) is evaluated even when off cadence. Failed evaluations keep
their artifacts and retry without recollecting or repeating a committed update.
An empty exhausted collection performs no update/evaluation; inspect the last
available comparison. These scheduling behaviors are CPU-tested only.

Both `train_loop` and `run_actor` expose `--clip-low`, `--clip-high`, and
`--dual-clip` (defaults 0.2/0.28/10). Actor resume requires the same values saved
in the checkpoint; changing them is not a transparent continuation.

`--entropy-coeff` is available in actor/launcher/driver (default 0). Nonzero
values subtract full-distribution response-token entropy from the PPO objective,
using the same global token denominator and no trajectory advantage weights.
CPU value/gradient checks pass; nonzero-entropy GPU training remains unvalidated.
At coefficient 0 entropy is not computed (`entropy: null` in progress metadata).
Resume requires the saved coefficient; legacy checkpoints used coefficient 0.


### BF16 + FLA 连续训练

`scripts/train_rl.sh` 默认 `PRECISION=bf16`、`GPUS=(1 3)`、新运行名 `continuous_bf16_aligned`。W&B 保持 offline，前台运行，支持 tmux。旧运行的源码与科学配置指纹不同，不应覆盖旧目录强行恢复。

训练保存 FP32 主参数，FSDP 使用 BF16 参数计算、FP32 梯度归约/缓冲区；几何坐标保持原始 FP32。采样使用原生 Pano BF16，FLA/causal-conv1d 必须可用；不使用 vLLM。训练前严格验证 actor train/eval 回放一致性，缓存采样与 actor 整段概率差记录为诊断；PPO 分母仍是 actor 重算的 old_log_probs。FP32 模式保留原有跨路径误差门槛。

检查点记录 precision，恢复时拒绝精度变化。换卡可修改 GPUS，恢复机制保留优化器与采样游标；同卡数可恢复 RNG，换卡数采用原有明确的重新播种规则。原生 BF16 与 FP32 浮点结果不保证逐位相同。

已验证环境额外依赖：`flash-linear-attention==0.5.2`、`fla-core==0.5.2`、`causal-conv1d==1.7.0`，保留现有 torch 2.10.0+cu128 / transformers 5.5.0。SimpleVLN 未发现 RoPE2D CUDA/xFormers 使用证据，本轮不引入。


当前启动批次配置已对齐SimpleVLN默认：ROLLOUT_N=8、TARGET_GROUPS=128、GENERATION_GROUPS=192、MINI_BATCH_DECISIONS=128、EPOCHS=20、MAX_UPDATES=0（不再10批提前停止）。generation_groups写入不可变科学配置；更改后使用新运行目录，不静默覆盖旧运行或旧检查点的训练配置。192个任务是取数批次，不代表192个GPU并发；采样并行、决策microbatch及每5批持久化仍在独立性能优化阶段，本次仅对齐上述批次/epoch配置。
