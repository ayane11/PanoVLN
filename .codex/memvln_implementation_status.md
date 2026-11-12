> 最新输出约定：`outputs/memvln_joint_4gpu/`（复数 outputs）。用户将自行重启；旧任务退出不需排查，不自动启动。

> 当前执行约定（2026-11-12）：所有交付脚本在 Docker 容器内部执行。四卡入口为 `bash /workspace/VLN/PanoVLN/scripts/start_memvln_joint_4gpu.sh`，脚本不调用 docker，自动定位项目目录。输出 `output/memvln_joint_4gpu/`，W&B offline。下文旧的“宿主机入口”记录已被此约定取代。启动由用户执行，代理不自动开训。

# MemVLN → PanoVLN 实施状态

更新：2026-11-12。目标仍在进行，不能将本文件视为完整交付证明。
计划：`memvln_to_panovln_plan.md`。

## 已实现，仍需完整审阅

- `src/memory/`：detach raw bank、FIFO/FRME、可训练 ToMe、位姿/动态时间编码、latent retrieval、分块/checkpoint、推理缓存与状态导出。
- Qwen3.5：图像 merger 之后、语言模型之前注入 memory；显式 generation 元数据接口，cached decode 不重复读写；关闭 memory 保留原分支。
- JSONL offset 索引：动作/图像前缀审计、episode rank 分片、只在实际消费后推进游标；完整与显式验证的子轨迹摘要接口。
- 流式训练入口：语言模型 LoRA、新特殊 token 行、独立 memory optimizer 分组、ZeRO-2、offline W&B、逐 rank raw bank/游标恢复和推理 bundle 导出。
- 评测端已接 bundle、重规划记忆和实际 env.step 后动作积分；未运行 Habitat 闭环验证。
- 三阶段配置与显式 GPU 列表启动器；dry-run 隐藏 GPU。

## 首轮验证证据（历史记录；最新结果见末尾）

容器：`dyj_vln`；Python：`/root/miniforge3/envs/panovln/bin/python`。
所有 Python 验证通过 `CUDA_VISIBLE_DEVICES=` 隐藏 GPU。

1. `python -m unittest discover -s tests -p 'test_*memory*.py' -v`：最近完整运行 11 个测试，其中 10 通过、1 个真实数据测试按开关跳过。验证 tiny Qwen3.5 混合注意力、视觉分组、generation cache、LoRA/token 行梯度、adapter 重载/合并、memory 梯度、容量、轨迹隔离、采样恢复与 offline/batch 约束。之后新增的细节修正还需要最终重跑。
2. `PANOVLN_LOCAL_DATA_TEST=1 ... -p test_memory_data_local.py`：1 个真实数据测试通过。使用真实 PanoVLN processor 和 R2R 首条完整 episode，验证 300 个占位符、导航数据与摘要无指令泄漏。
3. 首条 R2R episode 有 20 个监督样本，step 从 0 到 42，最终监督以 stop 开头。
4. 全量 `train_r2r_rxr.jsonl` offset/前缀审计通过：28,755 episodes、811,334 navigation samples、28,755 full summaries。CPU 耗时约 90 秒。三卡各 rank 共同长度 278,527；四卡 208,555。此次审计只读，没有持久化大型索引。
5. 三卡启动器 dry-run 通过：world_size=3、accumulation=8、offline。四卡 dry-run 通过：world_size=4、accumulation=6、offline。
6. 已运行 `git diff --check`，无输出。曾因原有 root 所有的 __pycache__ 目录导致 compileall 写缓存权限失败；后续改用 AST 解析和 `PYTHONDONTWRITEBYTECODE=1`，不修改已有目录权限。

## 首轮待办（历史记录；部分已在后续轮次完成）

- 收尾审阅训练更新次数、scheduler、epoch/phase 初始化、同卡数完整 optimizer/stream 恢复和阶段切换。
- 用 CPU 测试推理 bundle 的完整加载路径、已验证子摘要接口、checkpoint/梯度重计算与真实冻结参数范围。
- 检查预热阶段固定子集和 200 updates 是否与真实采样策略一致。
- 扩展数据审计到 DAgger 混合数据，并检查帧存在性、子轨迹标签映射；不能声称已完成 FGR2R 对齐。
- GPU 前后向、满 100 条目显存测试、三/四卡通信与续训、真实 checkpoint 闭环评测均未验证。**开始前先询问用户。**
- GPU 验证和最终实验组/消融尚未运行；无性能提升结论。
- 审阅/验证实验结束后删除多余日志和 checkpoint，保留必要复现证据在 `.codex`；不删除原基座或用户原有文件。

## 工作区注意

实施前已存在 `.gitignore` 修改和未跟踪 `data` 符号链接，未回退或覆盖。评测源码由 root 所有，使用现有容器在对应挂载工作区编辑。未创建 commit，未启动正式训练，也未生成训练 checkpoint。


## 第二轮 CPU 审阅与回归（2026-11-12）

- 实际修改：FRME 后独立保存 last_written，防止被合并删除的最新时间戳被重复写入；恢复 bank 时转移到模型的实际 device/dtype。
- 固定预热子集：seed=42、256 条专家 episode；按所有 epoch 的实际 rank 截断长度计算 optimizer 预算，预算不足报错，不悄悄少训。
- 流式学习率修复：scheduler 在 accelerator.prepare 替换/offload optimizer 后创建，手动每个真正 optimizer update 前进一次，独立注册 checkpoint，避免框架按 world size 多次推进。
- checkpoint：增加完整性标记、配置/world-size 契约、schedule fingerprint；阶段切换用 --init-from，仅同阶段状态用 --resume。共享 embedding 的 PEFT 重复别名按唯一学习参数验证。
- offline 日志记录跨 rank 平均 loss、累计 navigation/summary samples 和 target tokens。RNG 恢复移到日志初始化之后。
- 仅清理本训练器带 complete.json 的多余 checkpoint，默认保留最近两个，保护仍被引用的初始化/恢复 checkpoint、未知目录、符号链接和未完成写入。
- 当前完整测试命令（容器中）：
  `CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PANOVLN_LOCAL_DATA_TEST=1 /root/miniforge3/envs/panovln/bin/python -m unittest discover -s tests -p 'test_*memory*.py' -v`
- 最近一次完整回归：17 项全部通过（10.295 秒）。包括源 ToMe 与 SDPA 数值一致、FRME 最新帧去重、子摘要因果顺序、固定子集/更新预算、共享输入输出 embedding 的 adapter/阶段重载、真实 tokenizer/processor bundle 重载、optimizer/scheduler/RNG 恢复、checkpoint 清理保护。随后只调整了训练统计与恢复时机，尚待 GPU 集成验证。
- 全量 DAgger 混合 JSONL 审计通过：57,480 episodes，其中 r2r=10,692、rxr=18,063、dagger=28,725；navigation=1,357,126、summary=28,755；CPU 约 240.5 秒。此审计不持久化大型索引。
- 该混合数据中的专家子集（固定 256 条）可提供三卡 295 次、四卡 293 次完整 optimizer updates，满足预热 200 次预算。
- 在 general_vln_dataset 与 sub_dataset 标注目录未发现 FGR/子轨迹帧对齐文件。已支持显式 verified 子摘要输入，不声称已构建 FGR2R 标签。早先扫描整个图像目录的只读 rg 已主动结束，避免无意义遍历。
- 新 GPU 探针：scripts/smoke_memvln.py，需要显式 --run，包含真实模型导航前后向、重复真实特征的满容量压力测试和 bundle 重载。容量测试是合成压力测试，不代表导航性能。
- **已向用户提出仅用物理 4 号 GPU 完成本次探针检查的请求；截至本记录写入仍未收到授权，未运行任何 GPU 工作。**

## 阶段运行方式（均须先有相应用卡授权）

```bash
# 所有命令在 dyj_vln 内 /workspace/VLN/PanoVLN 运行。
# CPU dry-run（不使用 GPU）
bash scripts/train_memvln.sh --gpus 4,5,6 --config src/train/config/memvln_warmup.yaml --output-dir checkpoints/PanoVLN-Mem/warmup01 --dry-run

# 单卡探针：当前等待授权，不能自行运行
CUDA_VISIBLE_DEVICES=4 /root/miniforge3/envs/panovln/bin/python -m scripts.smoke_memvln --run

# 预热：memory+新 token 行，固定 256 episodes、200 updates
bash scripts/train_memvln.sh --gpus 4,5,6 --config src/train/config/memvln_warmup.yaml --output-dir checkpoints/PanoVLN-Mem/warmup01

# 新阶段：从预热 inference bundle 初始化；重建 optimizer/sampler
bash scripts/train_memvln.sh --gpus 4,5,6 --config src/train/config/memvln.yaml --init-from checkpoints/PanoVLN-Mem/warmup01/checkpoint-200/inference --output-dir checkpoints/PanoVLN-Mem/joint01

# 同阶段中断恢复：配置不变、卡数不变、显式指定完整 checkpoint
# bash scripts/train_memvln.sh --gpus 4,5,6 --config src/train/config/memvln.yaml --resume <joint-checkpoint> --output-dir checkpoints/PanoVLN-Mem/joint01
```

所有训练入口强制 W&B offline。真实多卡训练、满容量 GPU 显存、Habitat 闭环、评测实验组和性能结论仍未验证；CPU 测试不能证明这些部分完成。


## 第三轮 CPU 分布式检查（2026-11-12）

本轮仍未收到 GPU 用卡请求的答复，自动继续目标不算授权，未运行 GPU 工作。

- 将生产训练循环的完整单样本消费提取为 `step_stream_sample`，CPU 分布式检查直接调用该实现，而非复制另一套训练逻辑。新增非有限 loss 检查；消费游标仅在前后向完成后推进。
- 模型接口检查每个 batch 的 episode/step/odom 数量，禁止同一 batch 带同一 episode 的多个状态；检查 memory 占位的监督 label 为 -100。安装 memory 时检查特征维度匹配主干。
- 新增不同图像数量（2 张/1 张）且含 padding 的双样本测试，验证扁平视觉特征正确归属各自 episode。
- **三进程 CPU/Gloo**：2 个 optimizer updates、每 rank 消费 4 个样本、scheduler 前进 2 步；全局 navigation=9、summary=3；各 rank 所有可训练参数相同。
- **四进程 CPU/Gloo**：同样 2 updates、每 rank 4 样本；全局 navigation=12、summary=4；在第 1 次 update 后、episode 尚未结束时保存 optimizer/scheduler/各 rank raw bank 和游标，恢复后得到与不中断训练相同的参数、loss 统计和学习率。临时 checkpoint 自动删除，检查 `/tmp/panovln-ddp-review-*` 剩余数量为 0。
- CPU worker 使用 Gloo、小型 Qwen3.5、合成图像输入和 find_unused_parameters=True；它证明训练控制流和 CPU 通信/恢复，**不能证明 CUDA/NCCL/ZeRO-2、真实 4B 模型或 GPU 显存已通过**。
- 完整回归：18 项全部通过，9.912 秒，包含本地真实 processor/tokenizer 与 tiny 模型 bundle/阶段重载；没有跳过本地数据测试。
- `git diff --check`、相关模块 AST 解析通过；正式训练目录 `checkpoints/PanoVLN-Mem` 尚不存在。

复用 CPU 检查命令：

```bash
# 容器内项目根目录；明确隐藏所有 GPU。
CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /root/miniforge3/envs/panovln/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=3 tests/memory_distributed_worker.py
# 四进程恢复检查需传入本次创建的临时共享目录，并在检查后删除：
# ... --nproc_per_node=4 tests/memory_distributed_worker.py --checkpoint-dir <temporary-directory>
```

**下一步仍是等待用户明确允许仅用物理 4 号卡进行本次真实模型探针，之后才可运行 GPU 检查。正式训练、满容量 GPU 压力测试、GPU 多卡恢复和 Habitat 闭环均未完成，不能将目标标记为完成。**


## 等待用卡授权

2026-11-12：经过连续多轮目标执行，GPU 授权请求仍未获答复。本轮再次核查工作区、`git diff --check` 和既有证据，确认下一项必要验证为真实模型 GPU 探针。已完成的 CPU 检查不再重复；目标按 blocked 条件停止自动推进，等待用户明确授权。

待确认的具体动作：只使用物理 GPU 4，运行 `scripts/smoke_memvln.py --run`，验证真实 4B 模型前后向、100 条 bank 容量及推理 bundle 重载。本次不启动正式训练，临时 checkpoint 检查后删除。GPU/NCCL/ZeRO-2 和闭环评测尚无通过证据。


## 本次单卡探针已获授权（2026-11-12）

用户明确答复“允许，仅使用 4 号卡进行本次检查”。授权范围仅为真实 PanoVLN 前后向、100 条记忆容量与重载检查，不包括正式训练或其他 GPU。启动时物理 4 号卡占用 17,636 MiB，总容量 49,140 MiB。

执行命令：容器 dyj_vln 内 `CUDA_VISIBLE_DEVICES=4 WANDB_MODE=offline PYTHONDONTWRITEBYTECODE=1 /root/miniforge3/envs/panovln/bin/python -u -m scripts.smoke_memvln --run`。当前 unified exec 会话 14028；须观察该会话，不可因输出暂时停顿重复启动。启动后读完 730 个模型权重条目，构建得到 381,763,200 个可训练参数、5,724,056,193 个总参数；尚无探针通过结论。


## 物理 GPU 4 单卡检查完成（2026-11-12）

授权来源：用户回复“允许，仅使用 4 号卡进行本次检查”。本轮包含定位问题后的重试，所有 GPU 工作都仅暴露物理 GPU 4；没有运行正式训练或使用其他卡。

### 发现与修复

- 首次真实模型导航前后向、100 条 bank 容量均可运行，但重载 logits 的少量元素未通过 `atol=.03, rtol=.03`；没有放宽比较阈值。
- 最初排查了 BF16 LoRA 合并：加载器改为保留训练时的 PEFT 计算路径，避免把 FP32 LoRA 增量强制合并到 BF16 基座。此改动本身没有消除误差，因此不能把初始问题归因于 LoRA 合并。
- 比较全部参数、buffer、memory/visual/text embeddings、position IDs 和两种 cache 模式，将问题定位到 memory 压缩：视觉和文本 embeddings 完全一致，cache 模式差异为 0，只有 memory embeddings 不同。
- 独立 2560 维、100 条 bank 的复现实验中，同一模型清空并重建 grounded cache 也有约 0.046875 的输出差异，排除了 checkpoint 丢参数。压缩差异约 0.03125，grounded tokens 最大差异 4.25。
- 根因：源 ToMe 的 CUDA BF16 `scatter_reduce(sum)` 使用原子累加，顺序不固定；重复压缩不完全一致，下一层匹配会放大低精度扰动。
- 修复：保持 ToMe 二部匹配/加权平均结构，稳定排序处理相同分数，使用固定分组的 FP32 矩阵累加，加权平均后只做一次 dtype 转换。源代码数值适配已写入 NOTICE.md。没有替换为另一种记忆算法。
- 独立模块修复后，重载和清空缓存重建两项最大误差均为 **0.0**，通过 `atol=0, rtol=0`。

### 最终整模型结果（退出码 0）

| 检查项 | 结果 |
|---|---|
| 模型总参数 | 5,724,056,193（包含冻结几何编码器） |
| 可训练参数 | 381,763,200 |
| 真实首帧导航 loss | 0.4344579875469208 |
| 满 bank 前后向 loss | 0.4159335196018219 |
| ToMe / retrieval / pose 梯度 | 均有有效非零梯度 |
| FRME 压力写入后 bank 数 | 100 |
| 本进程峰值 allocated | 16.79190969467163 GiB |
| 本进程峰值 reserved | 18.8203125 GiB |
| cache 模式差异 | 0.0 |
| 重载后的有效参数/buffer 差异 | 0 个 |
| memory / visual / text embeddings 差异 | 均为 0.0 |
| position IDs | 完全相同 |
| cache 开启/关闭各自重载最大 logits 差异 | 均为 0.0 |

显存数据为本次首帧、重复同一个真实视觉特征构造的 100 条 bank 压力测试；不是最长历史、真实多轨迹训练或 DAgger 闭环的显存上界。不能据此声称所有 3/4 卡训练条件均可运行。

### CPU 回归与清理

- 完整 CPU 回归：19 项全部通过，9.065 秒。新增 BF16 确定性归并的单次舍入与梯度测试；仍覆盖源 FP32 计算一致性。
- 随后补测保留 PEFT 的 bundle 生成路径（CPU）：2 项本地数据测试通过，13.991 秒；bundle 可执行自回归生成，cached decode 没有重复写 bank。
- 全部 GPU 探针临时目录 `/tmp/panovln-memory-probe-*` 剩余 0 个；`scripts.smoke_memvln` 进程剩余 0 个。正式训练目录 `checkpoints/PanoVLN-Mem` 尚不存在。
- 当前只保留本文件和实现/测试代码，没有保留这些实验的临时 checkpoint。

### 当前授权边界与下一步

本次单卡探针已结束。已向用户请求新的范围：仍只用物理 4 号卡，完成 **3 次参数更新的 ZeRO-2 训练入口与断点恢复检查**，W&B offline，检查后清理临时 checkpoint；不启动 200 updates 预热或整轮训练。尚未收到该新请求的授权，不能自行执行。

尚缺：真实 ZeRO-2 训练/恢复、CUDA 多卡通信与三/四卡验证、Habitat 闭环、正式训练及实验对照。整个迁移目标尚未完成，不应调用 complete。


## ZeRO-2 短程检查已获授权并启动（2026-11-12）

用户回复“允许，仅用 4 号卡做短程训练验证”，随后将目标状态设为 active。范围为逻辑预算 3 updates 的 ZeRO-2 训练与断点恢复；验证方式是先完整走 3 步，再从第 1 步恢复并重放剩余 2 步比较。只暴露物理 GPU 4，W&B offline，不启动 200 updates 预热或整轮训练。

使用真实专家轨迹的起始/终止观测与完整摘要，形成 2 个导航样本和 1 个摘要样本；这是训练接口检查，不是性能实验。控制器为 scripts/review_memvln_trainer.py，临时数据、checkpoint 和 W&B offline 日志均放在项目挂载盘上的临时目录（自动清理）。训练 checkpoint 排除冻结基座，恢复时基座来自原模型路径。

最初自动审批服务连接失败、命令未执行；通过相同审批流程恢复后核查确认文件未创建，再完成准备并启动。当前 unified exec 会话 43652，应持续观察这一会话，不可因输出少而重复启动。


## ZeRO-2 三步训练与断点恢复检查通过（2026-11-12）

仅物理 GPU 4，逻辑训练预算 3 optimizer updates。使用真实轨迹起始/终止状态与完整指令摘要（2 navigation + 1 summary，65 target tokens）。完整执行后从 checkpoint-1 恢复，重放第 2、3 步；未启动 200 updates 预热或整轮训练。

### 检查中发现并处理的问题

- 最初未开启完整 CUDA 反向确定性时，步数/计数/学习率正确，但最终部分 tensor 比较失败；没有放宽 `atol=2e-5, rtol=1e-4`。
- 增加确定性设置：CUDA 初始化前设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8`、`FLASH_ATTENTION_DETERMINISTIC=1`；禁用 TF32、cuDNN benchmark，启用 cuDNN deterministic 与 `set_seed(..., deterministic=True)`。
- 加入恢复前严格审计：加载 optimizer/model 后，与同一 checkpoint 的 inference bundle 逐项比较 adapter 和 memory 张量，要求零误差。三套阶段配置默认开启 deterministic 与 verify_resume_state。
- checkpoint 排除冻结基座；恢复时先从配置中原基座构建模型，再恢复训练状态；严格 learned-tensor 审计防止非严格 frozen-state 加载掩盖可训练权重缺失。
- 训练 checkpoint 重写时先使旧 complete.json 失效，最后原子提交新的完成标记，避免从较早 checkpoint 恢复后覆盖未来编号时误读旧完成标记。该标记逻辑在 CPU 上有中断覆盖回归。

### 最终 GPU 结果（退出码 0）

| update | task | uninterrupted loss | resumed loss | next LR |
|---|---|---|---|---|
| 1 | navigation | 0.43369778990745544 | 恢复源 checkpoint | 1.5e-5 |
| 2 | navigation | 0.6706113219261169 | 0.6706113219261169 | 5e-6 |
| 3 | summary | 13.006841659545898 | 13.006841659545898 | 0.0 |

- 恢复后、任何新 forward 前，可训练张量与 checkpoint bundle 完全一致。
- 最终 553 个 learned tensors 全部通过比较，最大绝对差异 **0.0**。
- 恢复后 updates、schedule（含 fingerprint/已消费游标）、navigation_samples=2、summary_samples=1、target_tokens=65 与不中断路径一致。
- 两次进程均明确报告 W&B offline，并验证两个 offline-run 目录存在；实验结束后这些临时日志随临时目录清除。
- checkpoint_excludes_frozen_base=True，验证了该模式下真实 ZeRO-2 optimizer/model/scheduler/stream 恢复。
- 本轮 global_batch=1、每卡 batch=1；正式单卡 global_batch=24 的长程累积，以及 CUDA 多卡/NCCL 尚未实测。已只读核查 Accelerate 1.13 在 ZeRO>=2 时不会使用不兼容的 no_sync。

### 收尾验证与清理

- CPU 完整回归 **20 项全部通过**，9.862 秒，包括原子完成标记/中断覆盖检查。
- `git diff --check` 与语法检查通过。
- `.memvln-trainer-review-*` 临时目录剩余 0；本任务训练进程剩余 0；正式 `checkpoints/PanoVLN-Mem` 输出目录仍未创建。
- 用户原有基座权重、数据、.gitignore 与 data 符号链接未删除。

### 后续边界

已完成这次授权的三步验证。200 updates 的 memory 预热、完整联合训练、其他 GPU、多卡检查和 Habitat 闭环仍未获本轮授权，不能自动启动。

预热候选：仅物理 GPU 4，`src/train/config/memvln_warmup.yaml`，固定 seed=42 的 256 条专家轨迹，global_batch=24、单卡 accumulation=24、memory/token 行训练、LoRA 冻结，最多 200 updates。配置改为每 50 updates 保存，保留最近 2 个必要 checkpoint；W&B offline。候选输出 `checkpoints/PanoVLN-Mem/warmup01`，尚未创建。

预热命令的 CPU dry-run 已通过：world_size=1、accumulation=24、wandb_mode=offline。已向用户发出仅物理 4 号卡、最多 200 updates 预热的授权请求；截至本条写入未获答复，未启动预热。


## 200 updates 预热授权澄清与启动（2026-11-12）

用户在预热授权问题中回复“用3卡”，随后明确纠正：“我说你用卡3，而不是用三张卡”。因此授权为**物理 GPU 3 单卡**，不是三卡，也不是 GPU 4；先前关于 4/5/6 或 1/3/4 三卡的讨论不构成启动这些组合的许可。

已通过单卡 dry-run：world_size=1、accumulation=24、wandb_mode=offline。启动前 GPU 3 为 49,140 MiB 总显存、18,035 MiB 已用。

启动：容器 dyj_vln，项目 /workspace/VLN/PanoVLN，scripts/train_memvln.sh --gpus 3 --config src/train/config/memvln_warmup.yaml --output-dir checkpoints/PanoVLN-Mem/warmup01 --master-port 29546。环境 CUDA_VISIBLE_DEVICES=3、WANDB_MODE=offline、OMP/MKL threads=8。最多 200 optimizer updates，固定 256 个专家 episode，batch=1、accumulation=24、memory/token 行训练，语言 LoRA 冻结；每 50 updates 保存，仅保留最近 2 个必要 checkpoint。完成后停止，不自动进入联合训练。

**唯一训练 unified exec 会话：38591。宿主机临时 stdout 日志：/tmp/panovln-warmup01.SmBY0i.log。** 正式输出：/workspace/VLN/PanoVLN/checkpoints/PanoVLN-Mem/warmup01。尚未取得完成证据，不得将启动视为完成；观察会话/日志/complete.json，不能因观察超时或无新输出而重启。训练完成并将必要结论写入 .codex 后可删除冗余 stdout 临时日志；保留必要续训 checkpoint 和 offline W&B 数据。


## 开发评测准备与数据核验（2026-11-12，CPU-only）

新增 config/vln_r2r_mem_dev.yaml，继承原 R2R 配置且仅将 dataset.split 设为 val_seen。通过真实 habitat_baselines get_config 加载，确认 split=val_seen、全景宽度1280；未启动 Habitat Env、模型或 GPU 推理。原 val_unseen 配置不变。

核对压缩 JSON 中全部 scene_id 对应 data/scene 下的场景文件：R2R val_seen 778 episodes/53 scenes，val_unseen 1839/11；RxR guide val_seen 原始6746/57，val_unseen 原始11006/11；四组缺失场景文件均为0。RxR 数量是语言过滤之前的原始 guide 记录数，不能当作实际英语评测样本数。依赖 habitat、habitat_sim、hydra 可定位，配置导入成功；此证据不等于模拟器渲染或闭环验证通过。

现有预热 PID 2524995 本次仍存活，stdout 已到12/200 updates（278 navigation、10 summary、6146 target tokens）。不重启、不变更训练设置；尚无完成或导航涨分证据。后续评测授权仍需询问用户。


## 单卡开发评测入口（CPU 验证，2026-11-12）

新增 scripts/eval_memvln.sh：必须显式 --gpu、--model-path、--output-dir、--max-episodes；仅一张物理卡、一个 worker，默认 config/vln_r2r_mem_dev.yaml。拒绝已有结果目录，避免原评测跳过 episode 时混用 checkpoint。动作执行/CGE/STOP/碰撞恢复参数与原评测脚本当前默认一致，W&B offline。--dry-run 仅 shell 打印命令，不启动 Python、Habitat 或 GPU。

bash -n、正常 dry-run 与四项参数拒绝检查通过（多卡列表、负 episode 数、缺参数值、已有目录）。真实训练导出子目录为 checkpoint-N/inference，不是 bundle。候选评测命令：bash scripts/eval_memvln.sh --gpu 3 --model-path checkpoints/PanoVLN-Mem/warmup01/checkpoint-200/inference --output-dir outputs/eval/memvln_dev_smoke --max-episodes 8。该命令尚未获新增 GPU 授权，也尚未执行；先 dry-run 和准备入口不等于闭环通过或涨分。


## 用户更新执行方式（2026-11-12）

用户提出后续实际实验可考虑物理 GPU 0,1,2,3，但明确要求“帮我写个脚本，我来执行”。因此该列表不是代理自动启动新训练/评测的授权。后续交付可审阅脚本、命令与 CPU dry-run，实际启动由用户执行。用户尚未选择先评测预热 checkpoint 或先联合训练；先解释了预热含义，不应擅自把其中一项视为用户已选定。

已有单卡 GPU 3 预热与新实验分开处理：目标暂停/恢复并未发送停止后台进程的命令。本次只读核验 PID 2524995 存活、elapsed 32:29，最新 update=17/200。此处仅记录现有任务事实，不构成新任务授权。原任务完成后仍停止，不自动进入联合阶段。


## 旧预热终止与直接四卡联合训练交付（2026-11-12）

按用户要求向旧 torchrun PID 2524950 发送 SIGTERM，训练子进程2524995随之退出；两个 /proc 条目均已不存在，会话38591以信号终止。最后观察 update=18（418 navigation、14 summary、9158 target tokens）；未到首次50步保存，完整 checkpoint 数0。这不是训练故障，也不证明导航收益。

用户确认不需要独立 memory 预热；后续从原 checkpoints/PanoVLN 直接训练随机初始化 memory + 语言 LoRA + 新 token 行，视觉与几何冻结。scripts/start_memvln_joint_4gpu.sh 是宿主机入口，通过 dyj_vln 调用现有 train_memvln.sh，GPU 0,1,2,3、global batch24、accumulation6、联合训练1epoch、LR2e-5、W&B offline，每500updates保存、保留2个checkpoint。未传 --init-from 或 --resume，不依赖旧预热。CPU dry-run 已通过：valid=true/world_size=4/accumulation=6。真实多卡训练未由代理启动；用户手动执行。

清理仅限本次旧 warmup01（清理前文件总计0 bytes）及宿主机 /tmp/panovln-warmup01.SmBY0i.log。旧审阅的临时 checkpoint 已在先前验证后删除；可复用审阅脚本和测试保留，其他 /tmp 中来源不明的 panovln 实验日志不删除。


## 最终输出位置更新（2026-11-12）

用户要求直接保存到同训练任务目录。四卡宿主机脚本默认 OUTPUT=output/memvln_joint_4gpu，不再增加 PanoVLN-Mem/时间戳层。checkpoint 保存为 output/memvln_joint_4gpu/checkpoint-N/，offline W&B 与其他训练产物同在该任务目录。仍支持 --output-dir 显式指定其他任务目录；现有训练器拒绝覆盖非空目录。CPU dry-run 通过；未启动新训练。

容器内入口修复验证：从容器 /tmp 调用四卡脚本 --dry-run 成功，world_size=4、accumulation=6、offline。检查本次新增 train/eval/index/smoke/review 脚本，未发现其他 docker exec 或宿主绝对路径依赖。仅 CPU 检查，未启动训练。


## 用户已手动启动四卡联合训练（2026-11-12）

只读检测到 torchrun PID2546402，workers 2546446/2546447/2546449/2546450，对应 rank0/1/2/3。参数为 src.train.train_memvln --config src/train/config/memvln.yaml --output-dir output/memvln_joint_4gpu，物理卡0,1,2,3，WANDB_MODE=offline。代理未执行启动命令。最新进程观察 elapsed约1分25秒、各rank存活；尚无offline run目录或完整checkpoint，不能声称已完成参数更新。stdout/stderr 指向容器 /dev/pts/3，当前聊天未附加该终端；不要读取TTY设备以免干扰用户。后续优先检查产生的offline日志和checkpoint，观察超时不能作为重启理由。


## 四卡联合训练已完成真实更新（2026-11-12）

用户启动的PID2546402与四个worker仍存活，elapsed约3分33秒。offline-run-20261112_145548-n1khcczk/files/output.log 已记录 update1 loss=1.0210262139638264，update2 loss=0.9002441565195719，累计navigation48/summary0/target_tokens912。这是实际四卡训练更新证据（不仅dry-run），尚不证明长期稳定、checkpoint恢复或导航涨分。Epoch0尾截断 per rank=[2839,0,1446,1584]，accumulation tail=1；无完整checkpoint。未更改任务或额外启动实验。


## 用户指定 update=200 检查（2026-11-12）

当前运行：outputs/memvln_joint_4gpu，offline-run-20261112_150650-ju8oxp0k。torchrun PID2554447，workers2554498/2554499/2554500/2554501，本次均存活，最新已到201。此前用户要求等待200再检查，并未要求在200停止；未向进程发送信号或改配置。

前200条update日志完整连续1..200，loss全部有限。update200：loss=0.30973756313323975，navigation_samples=4638，summary_samples=162，target_tokens=102528，LR=3.838771593090212e-6。每50次更新混合loss均值：1–50为1.0245680010，51–100为1.1561646914，101–150为0.8346271241，151–200为0.8210080218。不同窗口包含不同轨迹、导航/摘要比例和难度，均值下降只作描述，不能当作固定验证集改善或导航涨分。当前LR仍处于配置3%学习率warmup，未达到2e-5目标值；这是联合训练的scheduler warmup，不是已取消的独立memory预热阶段。

尚无完整checkpoint，符合save_updates=500；未运行闭环导航、记忆干预或对照组，无法证明记忆利用/涨分。保留当前运行和offline日志，后续首个完整checkpoint可用于用户手动启动的开发评测；不得自动占用额外GPU。


## 300 步后保存与耗时排查（2026-11-12）

用户授权：找到当前实验完整 checkpoint 后停止，再排查；R2R 全量评测须先告知，新增 GPU 检查仍须先获许可。只读核查 torchrun PID2554447 和四个 worker2554498–2554501 均存活，最新观察 update320，输出 outputs/memvln_joint_4gpu 尚无 checkpoint。resolved_config 的 save_updates=500；训练代码仅在保存间隔和循环结束调用 save，无信号保存接口。不能靠修改磁盘配置改变现有进程，也不能先终止再指望获得当前权重。待 complete.json 及对应四 rank 状态、推理 bundle 全部存在后再终止该任务，保留首个有效 checkpoint。

CPU 代码审查得到的效率事实（不是瓶颈占比测量）：
- step_stream_sample 在每次 forward 前同步执行 dataset.sample、collator、设备搬运；没有 DataLoader worker 或下一样本预取。图片打开、RGB 解码、裁剪/缩放和 processor 均在训练进程串行完成。后续可测量 CPU sample 耗时，再决定是否加有界顺序预取；任何预取不得推进已消费 sampler 游标或写 bank。
- StreamDataset.sample 读取导航 JSON 行后，父类 __getitem__ 再读取同一行；这是重复解析，但尚无证据它是主要瓶颈。
- 训练 memory.read 对整个 raw bank 重算可训练压缩，chunk_frames=4；最多100帧时有25个分块。压缩与 retrieval checkpoint 导致反向重算。源 MemVLN 训练也重算 raw bank，因此不能简单缓存 detach 的压缩结果来提速，否则会改变训练梯度。
- bank.py 显式调用 deterministic=True 的 ToMe；其合并使用 FP32 one-hot/bmm。即使仅关闭 training.deterministic，该路径仍是确定性合并；性能评估需分别识别设置，不能把单个配置开关当成全部确定性计算的开关。
- 每个 microstep 有有限 loss 检查、float(loss.detach()) 与目标 token 数转 int 的设备同步；rank 间轨迹/图数/bank 大小不同，也可能产生等待。均须计时才能归因。

下一步先保留并验证 checkpoint，随后以用户批准的小规模 GPU 计时/开发评测决定投入；目前无导航收益证据，11天仅是早期均速外推。未修改正在执行的代码或配置，未启动任何新 GPU 工作。


CPU-only 数据预处理抽样计时：使用当前保存的 processor、同一 episode_index、rank0/world4/seed42 的 schedule，位置0/30/120/400/900/1800，CUDA_VISIBLE_DEVICES为空，OMP/MKL=8；未加载模型。六条均为导航样本，图数1/11/11/11/11/11，序列长度836/1709/1768/1590/1734/1672。sample耗时依次0.1254/0.6263/1.6585/0.7147/0.2107/0.3954秒，均值0.6219秒；collate约0.0012–0.0120秒。抽样少、与训练同时运行、文件缓存及图像不同，不能据此得出端到端占比或预计加速倍数。结果证明 CPU 预处理不是零成本，但不足以认定为11天的主因。下一次获准GPU计时应分别记录CPU sample、设备搬运、visual/PanoVGGT、memory压缩/检索、LM forward/backward/通信，随后决定优化；不以关闭记忆或减少容量代替完整机制验证。临时命令无磁盘产物，无需删除训练数据或checkpoint。


前341次更新的 bank 负载重建（CPU，依据当前 index 与 EpisodeSchedule(world4,seed42)，各rank前2046项，非GPU实时探针）：rank0/1/2/3平均条目18.079/16.309/18.580/17.075，中位16/15/15/15，最大66/61/100/73，满100条目的样本数0/0/14/0；摘要数66/71/68/70，合计275，与update341日志一致。推导遵循episode_start重置、同step导航写一次、摘要只读、容量100；它说明前341次更新绝大多数样本并非满bank，不能用“每一步都压缩100帧”解释当前耗时。源代码检索对照：目标RetrievalBlock为300 queries对历史tokens做cross-attention，FFN只作用于queries，need_weights=False；未发现把最多10000历史tokens错误地做全量自注意力的情况。源实现把所有历史帧一次batch压缩，目标chunk4；应测量分块大小/重计算的性能影响后再决定，不能直接承诺扩大chunk可提速或不OOM。


## checkpoint-500 已保存并按用户授权停止（2026-11-12）

已确认 outputs/memvln_joint_4gpu/checkpoint-500/complete.json 的 updates=500、world_size=4，四rank优化器分片、model状态、RNG、scheduler、流状态与inference bundle齐备，总文件大小7.045 GiB。在核对PID2554447命令仍对应当前四卡训练后发送SIGTERM；随后ps确认torchrun及workers2554498–2554501均已退出。最终日志update=503，持久化恢复点是500（后续少量更新未保存）。用户已授权找到checkpoint后停止；未重启训练。

CPU加载四个stream_rank文件成功，均epoch0/update500/consumed3000，rank和world4匹配；bank条目分别25/22/12/19。episode_memory.pt的56个state张量和adapter_model.safetensors的497个张量均可读且浮点值有限，后者包含trainable_tokens_delta。已确认完整标记和文件内容可读，但尚未执行本次四卡checkpoint的GPU重载logits/恢复训练验证。update500累计导航11600、摘要400、目标tokens256443，loss0.3175136646，LR9.5969289827e-6。无导航收益结论。

保留唯一有效checkpoint及对应offline日志，无本轮临时checkpoint/计时产物需要删除。后续新增GPU计时、短程闭环评测仍须先获许可；R2R全量评测未启动，必须先告知用户。

CPU恢复状态补充检查：checkpoint-500的scheduler last_epoch=500、_step_count=501、三组_last_lr均9.59692898272553e-6。四个ZeRO-2优化器分片均可反序列化，base_optimizer_state各组step均500，每rank的9个浮点状态张量均有限。未执行GPU恢复，因此仍不宣称本次checkpoint精确重放已验证。两组各8条R2R val_seen闭环命令已CPU dry-run通过；物理3号卡用卡请求已发出，尚待回复，不得自动启动。


## 获批物理3号卡两组各8条闭环检查（2026-11-12）

用户明确允许仅物理3号卡执行本次检查；依次运行scripts/eval_memvln.sh，baseline=checkpoints/PanoVLN，candidate=outputs/memvln_joint_4gpu/checkpoint-500/inference。两组均config/vln_r2r_mem_dev.yaml、seed42、split1/0、max_episodes8、相同CGE/STOP/碰撞策略、WANDB_MODE=offline，完整退出码0。计算进程GPU UUID映射确认在物理3号卡，未启动训练或全量评测。输出dev8_baseline、dev8_checkpoint500，CPU严格比较dev8_comparison.json确认两组8条scene+episode集合一致、指标齐全。

结果：baseline SR8/8=1，SPL0.9680528361，NE0.2526532980，nDTW0.9168040365；checkpoint500 SR6/8=.75，SPL0.7167404331，NE2.6722684456，nDTW0.7835308593。仅这8条便利小样本，不能外推完整验证集或判断训练充分性；当前无涨分证据。闭环均可运行，但还没有记忆干预实验，不能据此证明模型有效利用长期记忆。

失败两条均scene r1Q1Z4BcV1o：episode7基线成功、候选oracle_success0，最后显式STOP时NE14.080m，执行61动作（基线77）；episode8候选oracle_success1但最终NE6.104m、显式STOP、执行87动作（基线60），说明曾进入成功距离范围但最终离开后才停止。未发生运行异常或超时终止；原因尚不能归到memory/LoRA/训练不足中的某一个。保留逐轨迹动作记录作为必要复查证据，未生成临时checkpoint或视频，不删除唯一有效checkpoint。

速度线索：两次日志均显示Qwen3.5 linear-attention fast path不可用；CPU find_spec确认fla与causal_conv1d均未安装，flash_attn存在。另有xFormers不可用、RoPE2D无CUDA编译版提示。与训练相同panovln环境，值得后续获准后进行独立性能测试，但目前不能量化加速幅度或仅凭日志断言全部耗时根因。评测导航进度计时基线约141.6秒、候选193.9秒，包含不同动作数量和场景切换，不是固定输入模型延迟benchmark。未修改依赖或环境。
