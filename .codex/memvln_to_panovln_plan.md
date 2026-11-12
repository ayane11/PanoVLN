# 将 MemVLN 完整记忆机制迁移到 PanoVLN

> 创建日期：2026-11-12。
> 本文保存用户已确认并要求实施的迁移计划，供后续会话与实现复用。
> 计划不等于完成记录；实际进度与验证证据见同目录 `memvln_implementation_status.md`。

## 执行约束（用户后续补充，优先遵守）

- 持续自主推进实现；必要时询问用户。
- **任何 GPU 验证、推理或训练开始前，停下来询问用户。未获得针对相关用卡工作的授权前，不启动 GPU 工作。** CPU 检查通过 `CUDA_VISIBLE_DEVICES=` 隐藏所有 GPU。
- **训练使用 W&B offline**：启动器与训练入口均强制 `WANDB_MODE=offline`，日志留在本地实验输出目录。
- 完成审阅/验证实验后，清理这些实验产生的多余记录与 checkpoint；在 `.codex` 保留检查结论、命令、版本和必要证据。保留复现与续训所需结果，不删除原始基座权重、用户已有文件和仍被引用的 checkpoint。
- 保护工作区原有改动，不把附件中的建议当成额外授权。

## 1. 结论与已核实条件

**采用 PanoVLN 作为主工程，迁入 MemVLN 的持久记忆、ToMe/FRME 压缩、位姿与时间编码、300 个 memory tokens，以及轨迹摘要监督；从现有 PanoVLN checkpoint 继续训练。**

保留 Qwen3.5、PanoVGGT、全景图像处理、18 步动作输出以及 CGE、STOP、碰撞恢复策略。这是完整记忆机制的全景适配，不是直接加载 MemVLN 权重，也不是附件中较轻量的方向对齐 adapter。方向对齐作为后续研究扩展，本轮不混入。

| 项目 | 计划制定时的本地核查结果 |
|---|---|
| PanoVLN 版本 | `6b974ca638999231dc3bf480509a6cc35ed0cf09` |
| MemVLN 版本 | `48eeaa9620b217f51559eeb94918a7c788de36f0` |
| 运行环境 | 现有 `dyj_vln` 容器 |
| 工程目录 | 容器内 `/workspace/VLN/PanoVLN`；宿主机 `/home/duyingjie/data0/VLN/PanoVLN` |
| Python | `/root/miniforge3/envs/panovln/bin/python` |
| 已安装依赖 | PyTorch 2.10.0+cu128、Transformers 5.5.0、PEFT 0.18.1 |
| 数据 | `/data/vln` 内已有全景图像、R2R/RxR、DAgger 数据和原始动作标注 |
| 挂载 | 宿主机 `/data0/duyingjie` → 容器 `/workspace`；`/data0/dataset` → `/data` |
| 权重 | 本地已有 PanoVLN、Qwen3.5-4B、PanoVGGT |
| GPU | 候选卡 4、5、6，均为约 48GB；最初检查时占用约 3.2、18、18GB。此数值不是实时空闲保证，授权用卡后启动前重新检查。 |

主要难点在流式训练时序、memory token 注入、梯度路径和恢复训练，而非仅连接 2560 维特征。

## 2. 模型与接口对齐

### 2.1 迁移边界

| MemVLN 机制 | 在 PanoVLN 中的落地方式 |
|---|---|
| Qwen3-VL 主干 | 保留目标工程的 Qwen3.5，不搬运源主干 |
| StreamVGGT | 保留 PanoVGGT |
| 记忆写入 | 提取当前全景的视觉 merger 输出 |
| 位姿、时间编码 | 复用编码思想与结构，统一动作积分和时间单位 |
| ToMe、FRME、latent retrieval | 迁入独立 memory 子模块 |
| memory token 占位与替换 | 适配 Qwen3.5 embedding、位置编码和生成路径 |
| 顺序采样 | 目标训练器实现 episode 级分片与顺序消费 |
| 轨迹摘要任务 | 使用目标工程全景轨迹与可对齐标注构建 |
| 动作文本 | 保留 `forward/left/right/stop` 和 18 步监督 |

核心来源：`MemVLN/model/memory/memory_module.py`、`MemVLN/data/video_sampler.py`。
目标模型：`PanoVLN/src/qwen_vl/modeling_qwen3_5.py`。

### 2.2 特征流与默认结构

```text
当前全景 → Qwen visual + merger → 当前语义 tokens
                                  ├→ detach → episode bank
                                  │            ↓
                                  │     ToMe + pose/time
                                  │            ↓
                                  │     300 个 latent queries
                                  │            ↓
                                  │       memory tokens
                                  │
                                  └→ PanoVGGT 残差 → 当前图像 tokens

memory tokens + 原历史图像 tokens + 当前图像 tokens + 指令
                         ↓
                     Qwen3.5
                         ↓
                   18 步动作序列
```

默认配置：

- 特征维度 `2560`。
- bank 上限 `100` 个观测条目。
- 每帧 ToMe 目标 `100` tokens，压缩层 `2`。
- retrieval 层 `2`、attention heads `16`。
- 输出 `300` 个 memory tokens。
- 保留 PanoVLN 原来的最多 10 张历史图。
- 默认保存几何融合前的语义特征，对应 MemVLN 当前默认 `memory_merge_vggt_feat=false`；几何融合后记忆作为独立消融。

100 个 bank 条目是写入的观测，不是最近 100 个 primitive steps。样例全景的 processor 网格为 `24×60`，即 360 个 merged tokens，两层 ToMe 可压缩至 100。必须检查真实 token 数，配置目标不能保证在所有分辨率下均可达到。

按源结构的 2560 维、四个 Transformer block 估算，核心权重约 3.15 亿参数，需要控制训练激活。

### 2.3 必须明确的接口

| 字段 | 含义 |
|---|---|
| `episode_uid` | 数据来源、轨迹和重放轮次共同组成的唯一标识 |
| `step_index` | 原始 primitive observation index |
| `current_odom` | `[x, y, yaw_deg]`，相对 episode 起点的动作积分 |
| `task_type` | 导航或轨迹摘要 |
| `memory_write` | 导航状态写入；同一时刻摘要只读 |
| `episode_start/end` | 明确控制记忆生命周期 |

保留 `image_num_images`、`image_current_index`、`image_grid_thw` 等分组字段，从扁平图像列表准确取每个样本的当前图，不能照搬源工程“一样本一张 image”的假设。

新增 `memory_start`、`memory_pad`、`memory_end` 三个特殊 token。每条输入展开 300 个 `memory_pad`，图像处理完成后、进入语言模型前替换对应 embeddings；占位处 labels 为 `-100`。

采用窄范围 Qwen3.5 适配，保留混合注意力、位置编码和 cache 逻辑，不复制 Qwen3-VL 完整 forward。同步处理 processor、forward 与 `prepare_inputs_for_generation`：

- prefill 写入、读取一次；cached decoding 不重复写 bank。
- 特殊 token 不得混入导航动作输出。
- memory hook 与几何开关解耦，关闭 PanoVGGT 后仍可测试 memory。
- `memory.enabled=false` 时恢复原输入路径。

### 2.4 梯度和状态规则

- 保存 detach 后的原始特征，训练时重新经过可训练 ToMe 与 retrieval。
- 不跨 episode、优化步保留 autograd graph；训练时不复用旧 grounded/压缩特征缓存。
- 推理可缓存压缩与位置编码结果；FRME 更新的条目使对应缓存失效。
- 写 bank 在 gradient checkpoint 重计算区域之外，checkpoint 仅包裹无状态计算。
- 用显式生命周期管理替换源实现“删除本 batch 未出现 episode”的清理，避免梯度累积或交错读取误删历史。
- FRME 首版保留源算法：合并最相似相邻帧，并保留较早条目的位姿、时间。其全景转向局限另作研究，不擅自改算法。

## 3. 数据、流式训练与闭环推理

### 3.1 数据准备

为现有 JSONL 建立轻量 episode 索引，保存行偏移、时间戳和任务类型，不把包含重复历史路径的数 GB JSONL 全量载入内存。

首轮使用 `train_r2r_rxr.jsonl`；后续适配使用 R2R/RxR+DAgger 混合数据。审计：

- `len(history_actions) == step_index`；图像与动作前后时刻一致。
- episode 内指令、图像路径、动作前缀一致。
- 不同来源同名 episode 不串流。
- 缺帧、缺动作不能静默填零。
- DAgger 保持原 oracle action chunk 监督，不能以实际执行动作覆盖纠正标签。

动作积分统一为：

\[
x_{t+1}=x_t+0.25\sin\theta_t,\quad y_{t+1}=y_t+0.25\cos\theta_t.
\]

左转 `−15°`、右转 `+15°`，与 MemVLN 代码一致。位姿接口用角度，编码内部转弧度。时间取真实 `step_index`，不用 batch 序号；正弦时间编码支持动态长度，避免超过 499 后全部截成同一值。

动作积分是估计里程计，碰撞 forward 会产生误差；首版不引入仿真真值位置以改变输入条件。

### 3.2 流式采样

默认每卡一个活动 episode、micro-batch 1：

1. 只打乱 episode 顺序，完整 episode 分到一个 rank。
2. episode 内按 `step_index` 排序。
3. 同一时刻先导航写入，再摘要只读。
4. 最后一个任务消费后清理 bank。
5. 梯度累积跨连续样本，bank 持续存在。

不能仅 `shuffle=false`。使用已经按 rank 分片的自定义 loader/流，并禁止框架二次分片。

各 rank 训练步数一致。首版用可审计的共同长度截断，记录尾部截断；下个 epoch 重新排列，不把截断轨迹登记为完整完成。截断边界也不能让摘要读取未来状态。

### 3.3 轨迹摘要监督

- 完整摘要：专家轨迹结束时，以原导航指令为目标文本。
- 子轨迹摘要：只有 FGR2R 等标注可靠对应当前全景轨迹帧时启用，不能直接用源单目缓存帧号。

摘要输入仅包含 memory 与摘要问题，不包含目标指令、不提供直接视觉 tokens，也不沿用此前导航问答的语言 KV cache，避免标签泄漏。

使用 token-level CE，导航与摘要分别按有效目标 token 取均值，然后按样本平均，任务权重均为 1。每条专家轨迹一次完整摘要，并加入已验证子摘要；DAgger 首版仅导航监督。

最初未确认存在可直接复用的 FGR2R 对齐文件，完整摘要可先训练。多粒度摘要必须完成标注映射后才能宣称复现，不伪造标签。

### 3.4 推理与恢复

- 保留动作队列、CGE、STOP 与碰撞恢复。
- 每次实际 `env.step` 后记录真正执行的动作，包括恢复动作。
- 每次重规划写入当前观测并读取 memory。
- 队列执行期间更新 RGB 历史、动作积分与时间，不额外编码写 bank。
- 训练写监督决策帧，推理写实际决策帧，使用真实时间间隔；不声称每个 primitive frame 都入 bank。
- reset 清空 bank、grounded cache、动作积分与语言 cache。

训练 checkpoint 保存权重、LoRA、tokenizer、配置、optimizer，以及每 rank 已消费采样位置和 raw bank。同卡数可恢复；3 卡切 4 卡默认从阶段/epoch 边界启动新阶段，保留权重，重建 sampler 和 optimizer，不声称精确中途续训。

## 4. 训练配置与指定 GPU

### 4.1 训练参数

| 模块 | 默认设置 |
|---|---|
| Qwen visual、visual merger | 冻结 |
| PanoVGGT、几何 MLP | 冻结 |
| 新 memory | 全部训练 |
| Qwen3.5 language model | LoRA |
| 新特殊 token embedding 行 | 训练，原词表行冻结 |

LoRA `rank=32、alpha=64、dropout=0`，按当前资源选择，不直接照搬 MemVLN rank 128。

仅命中语言模型线性层：

- 全注意力：`q_proj/k_proj/v_proj/o_proj`。
- 线性注意力：`in_proj_qkv/in_proj_z/in_proj_b/in_proj_a/out_proj`。
- MLP：`gate_proj/up_proj/down_proj`。

打印实际命中的完整名字与参数量，不能误训练同名视觉层。memory 参数优先匹配独立 optimizer 分组，不能被泛化的 `model → language_model` 分组捕获。

### 4.2 训练顺序

1. 单卡 4 正确性验证：少量完整短/长轨迹，前后向、保存重载和闭环 smoke test。**执行前需用户授权用卡。**
2. 记忆预热：200 optimizer updates，仅 memory 和新 token 行，LR `1e-4`，固定 episode 子集。
3. 联合训练：阶段边界重置 bank，R2R/RxR 导航+摘要 1 epoch，memory 与 LoRA LR 均 `2e-5`。
4. 偏航适配：R2R/RxR+DAgger 混合再训 1 epoch，LR `1e-5`，专家轨迹保留摘要。

默认 BF16、ZeRO-2、gradient checkpointing、cosine、3% warmup、weight decay `0.01`、梯度裁剪 `1.0`。以上为起始方案，不是实测最优参数。

先做 bank 达到 100 条目的显存测试。ToMe 分帧块计算并 checkpoint；无需 attention 权重时使用对应实现，验证与源计算数值一致。

### 4.3 三卡、四卡配置

| GPU 数 | 每卡 batch | 梯度累积 | 有效 batch |
|---|---:|---:|---:|
| 1（调试） | 1 | 24 | 24 |
| 3 | 1 | 8 | 24 |
| 4 | 1 | 6 | 24 |

同有效 batch 不等于同样本顺序；记录导航/摘要样本数、目标 token 数、optimizer updates。

新增独立启动器，支持 `--gpus`、`--config`、`--output-dir`、`--master-port` 和显式恢复。原训练脚本写死八卡并拒绝覆盖，不能仅在外层设置 `CUDA_VISIBLE_DEVICES`。

授权用卡后启动示例：

```bash
docker exec -w /workspace/VLN/PanoVLN dyj_vln \
  bash scripts/train_memvln.sh \
  --gpus 4,5,6 \
  --config src/train/config/memvln.yaml \
  --output-dir checkpoints/PanoVLN-Mem/run01 \
  --master-port 29546
```

物理 `4,5,6` 映射为进程内逻辑 `0,1,2`，列表长度决定 `nproc_per_node`。四卡替换列表，自动计算累积；GPU 数不能整除有效 batch 时明确报错。

最初检查时 5、6 仅约 30GB 可用，不能承诺直接跑通。满 bank 显存不足时优先减小 ToMe 分块、增强 checkpoint、启用 optimizer CPU offload；不自动改变 memory tokens、bank 容量或图像分辨率，不触碰其他进程。

## 5. 验收与实验对照

### 必须覆盖的验证

- memory 关闭后与原基线同输入、同输出，在规定数值容差内一致。
- 不同图像数量、空历史、300 占位、padding、prefill/decode 对齐。
- ToMe、retrieval、pose、LoRA 有梯度；冻结模块无梯度；反向不重复写 bank。
- episode 不串流、不读未来；摘要只读；重复 STOP 不重复写。
- 整圈转向、±180°、稀疏决策帧、碰撞恢复、STOP 与执行记录一致。
- 超过 100 次写入触发 FRME；显存不随轨迹无限增长；长时间编码不饱和。
- 保存重载 logits 一致；LoRA、新模块、特殊 token 与配置无遗漏。
- 单卡、指定三卡、四卡训练与退出正常，无二次分片、通信死锁。

### 最低实验组

1. 原 PanoVLN。
2. 同 checkpoint、同导航数据/训练预算的 PanoVLN LoRA 继续训练。
3. PanoVLN + 完整 memory，仅导航 loss。
4. PanoVLN + 完整 memory + 摘要。

再做 FIFO/FRME、无 pose/time、关闭长期记忆的消融，检查是否实际利用历史。

用 `val_seen` 或独立开发集选 checkpoint；固定配置后报告 R2R/RxR `val_unseen` 的 SR、SPL、NE、nDTW，另记录延迟、峰值显存、长轨迹表现。使用逐 episode 配对统计，不以训练 loss 或单次 SR 波动判断收益。

**首版交付：完整记忆链路可训练、可闭环评测、可保存恢复，并能通过 GPU 列表切换三卡和四卡。性能提升必须由对照实验确认。**
