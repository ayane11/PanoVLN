# 两卡常驻 RL 启动说明

正式实验由用户手动启动；本次适配验证不会替用户启动正式训练。

在宿主机执行：

```bash
docker exec -it dyj_vln bash /workspace/VLN/PanoVLN/scripts/train_rl.sh
```

全部可调项在 `scripts/train_rl.sh` 顶部。默认 GPU 顺序为 `(4 3)`：GPU4 负责原生模型采集和 Habitat 环境池，PPO 更新使用两张卡的 FSDP。每个训练进程的显存上限依次为 34/26 GiB；环境进程另占显存，启动时会重新检查余量。

## 与 SimpleVLN 对齐的训练配置

- BF16 计算、FP32 主权重与归约；FA2、FLA、梯度检查点、参数阶段卸载及 Adam 状态卸载。
- `PPO_MICRO_BATCH_SIZE=4`、`LOG_PROB_MICRO_BATCH_SIZE=8`。
- `MINI_BATCH_DECISIONS=1024`，对应源代码实际的 `ppo_mini_batch_size=128 × rollout.n=8`，每轮 PPO epoch 为 1。
- 128 个有效任务组，生成批次 192 组，每组 8 条轨迹；轨迹级 GRPO 和逆决策数权重，token-mean dual-clip PPO。
- LR 为 5e-7，clip 为 0.2/0.28，dual clip 为 10，梯度裁剪为 1；不使用 KL reward/loss，entropy 系数为 0。
- Prompt 上限 4096、响应上限 256 token；训练 episode 最多 200 个环境步、64 次决策。导航评估配置未随之修改。
- sliding 窗口为 48；环境池为 8，原生推理批量上限为 8，实际批量随请求到达变化。这三个参数分别控制任务、环境和推理容量。
- 原生模型由单个执行线程串行调用，asyncio 事件循环可继续收发环境请求，取消时等待在途推理结束再卸载模型。
- 复用源环境客户端：reset 在 600 秒内重试；超时轨迹由 sliding 调度记录、丢弃，其余继续；动作 step 不重试。任务组按实际存活轨迹计算优势并进行奖励方差过滤。

vLLM 及其 TP、KV/cache、eager 等引擎参数不适用于本项目的自定义 Qwen。其余运行规模按用户指定的两张卡适配。

## 保存、恢复和文件位置

`SAVE_FREQ=5` 表示每 5 轮“采集→PPO更新”保存一次完整 checkpoint，不是每 5 个 Adam minibatch。正常结束时还会保存尚未落盘的尾部更新。`MAX_UPDATES=0` 表示按 20 个数据 epoch 运行。

正式输出目录默认为：

```
outputs/dapo_continuous/continuous_bf16_aligned_2gpu/
```

常驻运行期间查看：

- `resident/latest.json`：最近一个已保存并通过审计的 checkpoint。
- `resident/checkpoint_v5`、`checkpoint_v10` 等：模型、优化器、RNG 和游标。
- `resident/update_v*.json`：每轮更新指标，未必已落盘为 checkpoint。
- `resident/phase_metrics.jsonl`：采集、更新、导出审计耗时。
- `resident/collection_v*/complete.json`：分组筛选、轨迹失败数和实际推理批量统计。
- `resident/collection_v*/failures.jsonl`：单轨迹异常及原始栈。
- `resident/wandb_offline/`：离线 W&B 日志。

顶层 `state.json` 记录调度阶段，常驻过程中的实际提交进度以 `resident/latest.json` 为准。顶层 `latest.json` 在 worker 正常结束后同步。

同一运行目录支持从最近的完整提交继续。未落盘的内存更新会从最近提交处重做。运行代码、数据和训练配置有绑定校验；改变训练超参数应另开 `RUN_NAME`，不要修改旧 checkpoint 元数据强行续训。

`KEEP_LAST=0` 默认保留全部正式 checkpoint，与源默认保留策略一致。正数表示仅保留本运行最近若干个已提交 checkpoint。单个完整 checkpoint 实测约 55 GiB。

`KEEP_ROLLOUTS=false` 只清理本运行已消费的临时轨迹缓存，保留待导出所需的最后一份 batch；数据集、旧实验及评估结果不在此清理范围。没有磁盘容量门槛，实际写入失败仍会终止本次尝试，上一完整提交保留。

输入资源来自容器中的 `PanoVLN/data/dapo`（实际挂载 `/data/vln/dapo`），启动前检查资产审计与渲染审计。脚本保持前台运行，建议在 tmux 中启动；Ctrl+C 会清理自身子进程并保留已提交 checkpoint。

## 本次验收范围

用户已将今天的目标调整为核对加速配置、四卡启动与保存，随后自行启动正式实验。串行/批量 BF16 数值波动作为诊断记录，不再当作源逻辑正确性的判据。训练涨分不作为今天的结束条件，也未据此作任何涨分承诺。

旧 SDPA 隔离试跑已完成两轮真实采集更新、6 次 Adam 更新及完整张量审计，临时 checkpoint 已清理。FA2/1024/sliding48 联合验证已完成首轮真实更新和更新后采集。用户随后要求停止剩余验证并自行启动四卡实验，故四卡完整流程及第二轮最终审计未实跑。验证任务已停止，临时轨迹数据已清理；详细状态见 `ACCELERATION_STATUS.json`。

当前用户已将启动卡改为 GPU3、4。原四卡实验目录保留；脚本使用新的 `continuous_bf16_aligned_2gpu` 目录。四卡失败日志显示 SIGTERM，未观察到明确 OOM；GPU5、6当时另有约24 GiB新增占用。
