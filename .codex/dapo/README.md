# PanoVLN DAPO 当前状态与对话交接

本目录集中保存迁移计划、功能核查和使用说明，不是模型输入或训练检查点目录。新对话先读本文件，再读取实时 `state.json` 和进程状态；不要把历史文档中的状态当作现状。

## 当前入口与目录

- 容器：`dyj_vln`；项目：`/workspace/VLN/PanoVLN`；宿主机项目：`/data0/duyingjie/VLN/PanoVLN`。
- 启动：容器内 `bash scripts/train_rl.sh`。该脚本前台执行，适合 tmux，采样和更新日志显示在终端并写入输出目录。根目录 `train_rl.sh` 只是兼容转发入口。
- 公共输入：`PanoVLN/data/dapo`，实际通过 `data` 链接访问 `/data/vln/dapo`。包括官方 6901 条混合 RL 清单、Habitat JSON、环境配置和资源验证结果。
- 模型起点：`PanoVLN/checkpoints/PanoVLN`。
- 当前运行：`outputs/dapo_continuous/continuous_main`。目录内 `state.json` 为恢复状态，`latest.json` 指向最近已提交检查点。
- 源码：`scripts/continuous_rl.py` 调度，`rl/panovln_dapo/` 提供底层算法、采样和模型适配；每次新实验建立自己的 `runtime/` 副本。
- W&B：始终 offline，运行目录下 `wandb_offline/wandb/offline-run-*`。恢复生成新的本地记录段，保留逻辑 run ID。
- 不读取 backup，不核对原实验，不停止其他人的任务。没有授权时不因观察超时重启训练。

## 2026-11-10 本次检查的事实

用户自行清理旧 RL 输出并在 tmux 中重新启动。目前观测到调度器 PID 1750564、环境 PID 1750862、采样 PID 1751105 均存活；第一条轨迹正常完成，第二条继续推进。实际阶段为第 1 轮 `collecting`，`completed_updates=0`，尚未产出这次实验的新 RL checkpoint。PID 与进度会变化，后续必须重新检查。

实际启动参数：GPU `1,3,5,6`，`max_updates=10`，`epochs=20`，`rollout_n=8`，`target_groups=8`，mini-batch 128，lr 5e-7，PPO epoch=1，保留最近 2 个完整检查点，不额外保留周期检查点。采样用第一张卡，环境用第二张卡；更新阶段所有指定卡参与。不是多卡并行采样。

连续调度会在每轮完成后保存并审计模型、优化器和游标，提交 `commit.json`，更新 `state.json` / `latest.json`，然后从新 checkpoint 开始下一轮，直到累计 10 轮或数据提前耗尽。10 次采样更新不是 10 个数据 epoch；`MAX_UPDATES=0` 才取消更新次数上限。启动后编辑 sh 不会修改已运行进程的参数。

当前使用的调度器 SHA 与运行配置一致，runtime 文件哈希一致。本次仅移动文档，不改正在运行的代码和配置，也未中断进程。前序 11 项 CPU 测试覆盖循环、恢复、换卡、导出接纳、日志转发和检查点保留；真实 W&B offline SDK 已验证。**这些证据不等于本次已实测跨轮：需看到 round_00001/commit.json、新 checkpoint 以及 round_00002 的采样/恢复日志，才能证明当前 GPU 运行完成了实际接续。**

## 如何检查运行和保存

```bash
cd /workspace/VLN/PanoVLN
cat outputs/dapo_continuous/continuous_main/state.json
cat outputs/dapo_continuous/continuous_main/latest.json
```

每轮产物位于 `round_00001`、`round_00002` 等目录。`update_*/checkpoint/` 是完整可恢复模型；只有检查点导出与审计通过才提交。保留策略在新检查点提交后才执行。批次中途停止会重做未提交的更新；已完成的采样可以复用。详细说明见 [连续训练与恢复](RL_CONTINUOUS_TRAINING.md)。

## 历史文件用途和可删范围

`history/` 中的文件均不被当前训练代码读取，删除不会影响训练或恢复，但会失去迁移依据。已添加历史提示的 Markdown 保留了原始记录，其中旧 outputs 路径可能已不存在。

| 文件 | 用途 | 建议 |
|---|---|---|
| RL_MIGRATION_PLAN.md | 原始需求、文件/函数级迁移计划 | 保留，核对完整目标 |
| RL_FEATURE_PARITY.md | 功能对齐与未完成项 | 保留；状态需与当前源码重新核对 |
| RL_EXPERIMENT_AUDIT.md | 开发过程、实验配置和故障排查记录 | 保留供追溯，不作为实时状态 |
| RL_OFFICIAL_DATA_ALIGNMENT.md | 官方混合数据来源、转换和资源验证记录 | 保留数据来源依据 |
| RL_STORAGE_CLEANUP_AUDIT.md | 已结束旧实验的停止/清理记录 | 可删，旧输出已清理 |
| RL_ASSET_AUDIT.json | 早期模型资产和执行约定摘要 | 可删，非当前启动 gate |
| RL_CURRENT_DATA_AUDIT.json | 早期数据划分、依赖和场景检查 | 可删，保留有助追溯 |
| RL_DATA_ROUNDTRIP.json | 旧 R2R 数据转换往返结果 | 可删 |
| RL_PARQUET_AUDIT.json | 旧 R2R Parquet 检查摘要 | 可删 |
| RL_RXR_DATA_AUDIT.json | RxR 标注加载/划分检查 | 可删，非当前训练清单 |
| RL_INPUT_CONTRACT_AUDIT.json | 原生输入与历史帧选择检查 | 可删，保留有助定位输入问题 |

当前启动真正读取的 gate 在 **data/dapo/assets_audit 和 data/dapo/render_audit_02**，不要把它们与本目录内可删的 JSON 摘要混淆。

## 仍未完成的目标

当前已具备多轮 RL 训练机制，但还不能宣称 SimpleVLN 全部功能和训练规模已完全迁移。当前每批 8 个有效组小于源默认训练批量；并行采样调度、非默认 KL/调度器等功能仍有差距。正式泛化涨分还需固定协议的 R2R/RxR val_unseen 评测及进一步验证。本脚本不自动进行导航评测。

## 2026-11-11 加速适配进展

已接通 `scripts/train_rl.sh` 的 `ROLLOUT_WINDOW`、`ROLLOUT_SCHEDULER`，经连续控制器传入 collect。采样支持 sliding（完成即补位）和 window（整窗完成再启动下一窗），任务数受窗口约束，返回顺序保持输入顺序；任一轨迹失败取消剩余任务并回滚游标。当前默认窗口仍为 1，原生模型仍同步逐请求生成；不能据此宣称批量推理或整体提速已完成。

验证：采样 9 项通过；连续控制器 13 项中 12 项通过、1 项因宿主机无 torch 跳过；shell 语法检查通过。未启动 GPU 训练，未修改已有运行目录。后续继续训练/log-prob microbatch、集中批量生成、FSDP/卸载/内核配置核对、常驻 actor 与每 5 轮持久化，最终做 GPU 正确性和整轮性能验收。vLLM 专属配置排除。

训练与 log-prob microbatch 已接入 backend → actor → run_actor → continuous_rl → train_rl.sh。配置名为 `PPO_MICRO_BATCH_SIZE` / `LOG_PROB_MICRO_BATCH_SIZE`；暂保持 1，待真实 Pano GPU 等价与显存测试后设源 Qwen3.5 4B 的 4 / 8。不同长度 prompt+response 先分别拼接，再左填充；每例只返回有效响应 token；每个 microbatch 聚合 numerator 后一次 backward，mini-batch 分母和 optimizer step 边界保持原值。检查点记录生效 microbatch 值。

容器 CPU 验证：backend 6 项（含异长响应、不同温度、熵和 masked 梯度等价、局部图像索引与 FP32 geometry）；actor 7 项；continuous controller 13 项全部通过（含原先宿主机跳过的 torch 梯度测试）。这些不证明真实 Pano 模型批处理已通过 GPU 验证。下一步验证原生多图映射与 FSDP 批处理，然后启用源 microbatch 数值。

FSDP 配置接入：`FSDP_SIZE=-1` 使用全部所选卡，指定更小的可整除规模时构建 replicate/shard 二维 mesh 与 HYBRID_SHARD。`ENABLE_GRAD_CKPT`、`FORWARD_PREFETCH`、`OPTIMIZER_OFFLOAD` 已贯通启动脚本、控制器、run_actor、actor 和实际执行；actor 导出记录配置。启动脚本默认 allocator expandable_segments、CUDA_DEVICE_MAX_CONNECTIONS=16，子进程继承。FSDP 构造契约 mock 测试 3 项通过，actor 启动器 2 项通过；尚无实际多卡 hybrid FSDP 验证。参数卸载仍待实现上游的阶段间卸载语义，不能用 CPUOffload(offload_params=True) 冒充（后者会影响梯度累积）。

原生批量生成与集中队列已接入：NativePolicy.generate_batch 保留原始逐例 prompt/image_indices，批量左填充，响应保留首个 EOS、剔除结束后的 HF padding，逐例提取采样概率。非 uncertainty 模式不再额外请求 raw output_logits。InferenceQueue 单 worker 独占模型，无多线程同时调用；批量等待配置 `INFERENCE_BATCH_SIZE` / `INFERENCE_WAIT_MS` 经控制器进入 collect。模型错误传播到全部等待请求并阻止后续请求；取消的请求不送模型。默认 batch/window 保持 1，真实 Pano GPU 概率/几何/梯度与吞吐尚未验证。

验证：队列 CPU 测试 3 项通过；容器 backend 测试 7 项通过（新增 EOS/pad 与原 prompt/probability 映射）。这些 mock/小模型测试不能替代真实模型批量验收。仍需参数阶段间卸载、常驻 actor/采样内存同步与每 5 批保存、attention/compile 实际路径核对，以及完整 GPU 验收。

真实原生模型验收工具：`python -m panovln_dapo.batch_probe --checkpoint checkpoints/PanoVLN --gpu 3 --output outputs/batch_probe.json`，可加 `--gradients` 检查全部有梯度参数（CPU 保存逐条梯度参考，内存开销较大）。默认用异长 1/3 帧合成观测，比较逐条/批量 log-prob；两路径预热、同步计时，报告峰值显存、显式 atol/rtol。它不验证导航、真实 Habitat 数据、多卡 FSDP 或生成分布。GPU 已占用超过 1GB 时拒绝启动，不清理其他任务。本次只检查资源，GPU 1/3 均占用，未启动 GPU probe。

上游内核审计：`verl/workers/engine/fsdp/transformer_impl.py` 的 `use_torch_compile` 在构造器中用于 `compute_entropy_from_logits`，不是整模型 compile；forward 仅在 `calculate_entropy` 为真时调用。源默认 entropy_coeff=0，当前 Pano 默认也为 0。因此不能宣称缺少整模型 compile，仍需接通非默认熵计算 compile 开关才能完整配置对齐。remove_padding 源 Qwen3.5 默认 False，与当前 padded 路径一致；vLLM eager/prefix-cache/TP/cache-engine 设置仍按约束排除。

GPU4原生batch=2、历史1/3帧、BF16/SDPA验证已完成：峰值allocated 12,307,620,352 bytes；逐条约0.284s，批量约0.247s（单次同步计时，不是整轮加速结论）。最大log-prob差0.10524，虽然在atol=.05/rtol=.01的组合判据下通过，尚不能作为训练精度通过证据。已获用户许可继续GPU4，启动FP32 atol=.001/rtol=.0001复测，后续仍需梯度和生成验证。entropy torch.compile开关已完整接入，完整CPU回归104项通过。

首轮评测配对分析见ROUND1_PAIRED_ANALYSIS.json：R2R 1839条完全匹配，SR 75.6389%→75.2583%（-0.3806个百分点），SPL 69.3392%→68.9047%；40条失败变成功、47条成功变失败。配对正态近似区间跨0，不能宣称涨分或确定退化。历史有效配置仍需核对，尚无首轮RxR候选结果。用户要求最终验证正常实验与涨分，后续新的用卡任务先询问用户；当前已批准继续GPU4定位误差和验证梯度/生成。

GPU4 FP32/SDPA异长批量概率严格测试通过：max_logprob_error=1.33514e-5（atol=.001/rtol=.0001），峰值allocated=21,864,228,864 bytes；证据outputs/acceleration_validation/native_batch_gpu4_fp32_01.json。已在同卡启动FP32完整梯度比较，42GiB allocator上限，报告native_batch_gpu4_fp32_grad_01.json，日志/tmp/panovln_batch_gpu4_fp32_grad_01.log；未完成前不宣称梯度通过。

首轮训练元数据补充：target_groups=8、generation_groups=1，1460 decisions/27740 tokens，12 optimizer steps，rollout/actor max_logprob_error=.21752、mean=.008519；BF16只做actor自身strict replay，rollout误差是diagnostic。正式对齐配置已将target_groups默认128/gen_groups192，不能把首轮8组小样本更新当成源规模实验结论。基线eval.log显示flash_attention_2、uncertainty budget1.2、collision_recovery_steps=2；训练采样固定4动作、collision_recovery_steps=0，历史候选评测日志仍需核对。

梯度验收最新：native_batch_gpu4_fp32_grad_01 因42GiB限制在逐条backward OOM（peak allocated 42,303,720,960 bytes），没有梯度等价结论；native_batch_gpu4_bf16_grad_01 覆盖728个有梯度参数，但首个视觉patch_embed.proj.weight比较失败，max abs=.15234，2.3%元素不满足atol=.05/rtol=.01。默认microbatch仍1，不可据此前向通过启用4/8。已增加batch_probe --activation-offload（autograd saved tensors存CPU，保留完整参数/token），启动FP32严格完整梯度复测：outputs/acceleration_validation/native_batch_gpu4_fp32_grad_offload_01.json，日志/tmp/panovln_batch_gpu4_fp32_grad_offload_01.log。该选项仅属于验收工具，不代表生产activation offload已接通。

ActorSession重构：actor.py的模型/优化器初始化、update和完整export成为同一Session的方法；main仍按原单批路径调用，控制器尚未切换常驻模式。真实CPU update循环两批测试通过，验证Adam step=1→2、动量保留、拒绝过期策略版本；控制器13项通过。export后将Adam moments按配置恢复CPU，为周期导出后继续更新准备。

native_batch_gpu4_fp32_grad_offload_01仍在逐条反向同一2.37GiB分配处OOM（42GiB上限）；saved tensor CPU存储没有解决此梯度分配峰值。GPU进程已退出。下一步严格FP32梯度需FSDP分片，已向用户询问是否允许GPU4+0+1三卡，不在答复前启动。

用户已批准GPU4+0+1三卡验证。启动前GPU0 free=31034MiB、GPU1 free=32223MiB，按每进程24GiB allocator预算+3GiB余量检查足够。已启动torchrun三卡fsdp_batch_probe，物理映射4,0,1；输出outputs/acceleration_validation/fsdp_batch_fp32_3gpu_01.json，日志/tmp/panovln_fsdp_batch_fp32_3gpu_01.log。工具比较全部本地梯度分片（所有rank汇总）、异长历史概率，使用FP32 atol=.001/rtol=.0001；不涉及optimizer更新或导航。未看到终态不能宣称通过。

常驻接口继续接入：collect.run可显式传入内存policy、EpisodeCursor和policy_version；三者绑定校验，已有cursor不再restore旧checkpoint，不重新加载模型/重新设种。采样返回序列化payload，actor.parse_batch用于内存/磁盘统一校验。resident exhausted-cursor测试通过，证明版本5不回退checkpoint且无模型重载；collect相关4项通过。控制器尚未调用resident接口，因此每5轮持久化仍未实现，不得跳过现有每轮save冒充周期保存。

三卡fsdp_batch_fp32_3gpu_01终态：各rank在serial backward申请2.37GiB时触及24GiB allocator上限，未得到梯度结论；非物理显存耗尽。重新检查free MiB为GPU0=35008、GPU1=34218、GPU4=48497，均满足28GiB+3GiB余量，在同一授权三卡任务内启动02：allocator28GiB、日志/tmp/panovln_fsdp_batch_fp32_3gpu_02.log、报告outputs/acceleration_validation/fsdp_batch_fp32_3gpu_02.json。全量CPU回归106项通过。

新增resident_loop常驻checkpoint周期状态机（尚待接GPU/控制器adapter）：同一actor连续更新，每5批export/commit、结束/数据耗尽flush尾批；仅export成功推进durable_version，失败后禁止复用可能已变化的内存actor，必须从上次完整checkpoint重建。测试4项通过：12批保存5/10/12、最大预算尾批、更新中断从第5批恢复重做、导出失败不推进持久版本。累计optimizer steps和间隔optimizer steps分别保存。当前生产Driver仍旧每轮保存，不能宣称save_freq=5已完整启用。

三卡严格FP32验证02已通过：全部rank完成异长概率、逐条backward、batch backward及本地全部梯度分片比较（atol=.001/rtol=.0001），报告outputs/acceleration_validation/fsdp_batch_fp32_3gpu_02.json。该证据证明FP32 FSDP当前batch=2计算等价，不能代替BF16/4或8 microbatch、Adam更新、导航或整轮吞吐验收。进程已正常退出。

常驻可执行适配层已新增resident_train.py（尚未GPU集成验收，不替换当前默认启动脚本）。JSON示例resident_example.json；由torchrun -m panovln_dapo.resident_train --config .codex/dapo/resident_example.json启动，要求现成的Habitat server。初始化ActorSession一次；采样阶段完整CPU state_dict直接load进常驻原生policy，无checkpoint磁盘权重交接；本地FSDP shards和冻结PanoVGGT按阶段迁移CPU/GPU，采样policy训练阶段驻CPU；每5轮/尾批完整导出+审计后原子发布latest；未提交完整导出恢复时须核对源批次SHA、previous durable版本再审计接纳。采样/actor初始化保留RNG序列，恢复在采样前还原RNG。checkpoint_audit新增显式expected_updates，默认仍1，多轮必须验证前序绑定，完整张量审计保持；新测试8项通过（含周期审计），中断接纳3项通过。公共actor已直接支持三卡非整除128决策分配，控制器不再重复注入此修复。正在请求下一次三卡集成验证授权；不能宣称常驻路径已可用于正式训练。

用户对下一次常驻集成测试明确指定GPU0、3、4，取代0、1、4；须重新核对这三卡显存后再启动。

按用户指定GPU0/3/4启动resident_phase_probe 01，CUDA_VISIBLE_DEVICES=4,0,3（GPU4 rank0采样），预算分别36/26/26GiB，启动前均满足预算+3GiB余量；日志/tmp/panovln_resident_phase_043_01.log，逐rank报告outputs/acceleration_validation/resident_phase_043_01_rank*.json。测试用合成观测/优势，真实BF16 FSDP+FP32 master、FLA、两次Adam更新、中间完整CPU state_dict同步原生采样模型、FSDP本地shards和冻结PanoVGGT卸载往返、批量生成，最后检查Adam step=2；不证明真实奖励或涨分。阶段迁移指针/梯度安全CPU测试2项通过；未完成.incomplete导出目录不会被恢复接纳，重试将残留目录改名隔离保留。常驻resume绑定manifest SHA；rank错误直接退出交torchrun结束同组，避免NCCL销毁等待其他rank。

本轮完整CPU回归117项通过。resident_phase_043_01已进入第一次真实Adam更新；三个rank通过actor train/eval replay，记录global norm约107.955（合成优势测试值，不是正式训练指标）。仍未完成phase roundtrip或第二次更新，不据此宣称集成通过。

resident_phase_043_01已失败：rank1（物理GPU0）在首次Adam状态分配时触及26GiB上限（报告resident_phase_043_01_rank1.json，peak allocated 26,930,547,712 bytes），其他rank随后等待collective。已精确SIGTERM本任务torchrun PID2119228结束同组；修复probe失败路径先写报告再os._exit，避免单rank destroy NCCL死锁。尝试在同一授权GPU4/0/3上以36/26.5/26.5GiB预算重测02，但GPU0实时空闲显存未满足预算+3GiB，preflight拒绝，02未启动；没有新的GPU进程。未通过前不宣称Adam或phase roundtrip集成完成。

用户最新说明：显存不够时可改两卡/单卡，不应反复提高共享卡压力。三卡集成02先前preflight未通过，后来GPU0free恢复至30802MiB且GPU3=30624MiB，重新preflight通过后已实际启动（36/26.5/26.5GiB），日志/tmp/panovln_resident_phase_043_02.log，报告resident_phase_043_02_rank*.json；当前已进入update1。

新增可选optimizer_offload_mode=parameter（默认bulk保持现有行为），逐参数将Adam moments载回GPU、调用原生AdamW一次、立即卸载CPU，减少所有moment同时常驻的step峰值。CPU4步、多参数组、AMSGrad、缺失grad情况下参数、全部moment、step与普通AdamW逐位一致，2项测试通过；拒绝step hooks、foreach/fused/capturable/differentiable模式。可供后续两卡/单卡阶段验证选择，尚无GPU证据，不能宣称其提速；checkpoint export仍需单独验证moment聚合峰值。

resident_phase_043_02已完成第一次真实Adam更新（全部rank打印step结束），进入gather_state、offload_actor并已到rank0 sampler_weight_sync。该阶段证明本次预算可完成第一轮bulk Adam及阶段卸载，但还未验证生成/加载回GPU/第二轮。新增可选optimizer offload后完整CPU回归119项通过。

resident_phase_043_02终态失败：权重同步、原生batch生成、actor reload完成，进入第二轮前的batch2概率重算时rank2（物理GPU3）申请736MiB触及26.5GiB上限（peak allocated 28,272,200,192 bytes）；因此整体未通过，不能宣称Adam跨阶段成功。用户明确显存不足可减两卡/单卡，已据此在GPU4/3启动两卡测试resident_phase_43_parameter_01，预算36/26.5GiB，使用已CPU逐位等价的parameter Adam offload；预重算改为与生产默认一致microbatch1，不减少参数/token覆盖。采样仍检验batch2生成；更大训练microbatch仍待验证。日志/tmp/panovln_resident_phase_43_parameter_01.log，报告resident_phase_43_parameter_01_rank*.json。逐参数模式也已接通旧启动脚本/控制器/run_actor/actor，默认bulk不变。常驻控制消息已改Gloo避免采样等待忙轮询GPU；本次新两卡测试含该变更。

两卡resident_phase_43_parameter_01已通过：两个rank均完成第二次Adam更新并stage=passed；第一次更新后完整权重CPU汇总、actor CPU卸载、原生批量生成、actor reload、第二次更新均执行。使用物理GPU4/3，parameter Adam offload，当前训练microbatch1。旧控制器13项CPU测试通过。仍需查看逐rank报告中Adam step=2，再开展实际周期导出/恢复与Habitat训练；不能据此宣称涨分。

用户已批准继续GPU4/3保存恢复验证。当前已启动resident_save_43_01：同两卡36/26.5GiB，parameter Adam offload，updates=5/save_freq=5，第五次更新后完整export+audit；会话日志/tmp/panovln_resident_save_43_01.log，逐rank报告outputs/acceleration_validation/resident_save_43_01_rank*.json，checkpoint目录outputs/acceleration_validation/resident_save_restore_43_01/checkpoint_v5。测试产物明确标记synthetic_probe与scope，不作为导航涨分模型。

已读取本机PyTorch _optim_utils：_convert_all_state_info逐flat参数将CPU状态移动到compute_device，故parameter模式export不再预加载全部Adam moments到GPU。恢复改为CPU mmap完整状态→FSDP.shard_full_optim_state_dict→CPU占位参数调用原生AdamW load_state_dict→将CPU状态绑定原参数，避免原load_state_dict自动全量GPU拷贝；CPU恢复后下一步参数和全部moment/step逐位一致，形状损坏拒绝，optimizer测试3项通过。公共ActorSession已直接去掉旧70GiB磁盘门槛（与既有用户约定一致），控制器patched_actor不再注入替换；13项控制器测试通过。尚未获得真实GPU保存或恢复通过证据。

五轮GPU保存测试已完成update1–5，两个rank均进入checkpoint_export；仍待导出、审计和独立恢复结果。当前控制器已接通--resident分支，复用setup资源gate/隔离runtime、进程组清理、环境服务和离线W&B ID；torchrun只启动一次，结束从resident/complete.json同步父state/latest。ENV_POOL_SIZE、SAVE_FREQ、PARAM_OFFLOAD经scripts/train_rl.sh贯通，EXECUTION_MODE默认legacy不变，常驻模式在science绑定以避免混用旧目录。常驻环境使用主卡，避免额外挤占共享训练卡；残留resident worker纳入检测。CPU控制器14项通过（新增常驻单worker/最终状态同步测试），常驻retention2项通过。提交后只清理本运行已审计旧checkpoint和已提交轨迹，保留未完成、未审计目录。GPU Habitat闭环尚未启动，不宣称已可正式涨分训练。

最新逐项加速对齐状态与未验证项集中见ACCELERATION_STATUS.json；CPU全量122项通过，控制器14项通过。五轮GPU模型分片已写出，optimizer.pt仍在写入.incomplete目录；没有atomic rename/audit前不能恢复或宣称保存通过。
