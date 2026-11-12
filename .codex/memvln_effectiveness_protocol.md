# MemVLN 机制利用与导航收益验证协议

更新：2026-11-12。用户最终目标是通过实验使模型实际利用 MemVLN 的机制并提高导航指标；实现完整、梯度非零或训练 loss 下降均不能单独证明该目标实现。

## 当前边界

当前实现为完整记忆迁移基线：语义特征 bank、ToMe、FRME、帧级位姿/时间编码、300 latent queries 与完整专家轨迹摘要。它尚不具备 ERP token 方位对齐、指令条件 retrieval 或显式子指令进度监督。用户提出的三个方向应作为候选研究扩展逐项验证，不能宣称已实现。

已获授权的运行仅是物理 GPU 3 单卡预热，最多 200 updates，W&B offline，结束后停止。后续 GPU 评测或联合训练需按用户要求先询问。不要修改正在运行的训练定义。必要 checkpoint 保留；临时审阅输出在结论入档后清理。

## 第一关：闭环可运行

预热结束核对 complete.json、实际 updates、采样计数、offline W&B、保存权重与可恢复状态。获授权后，在固定开发集 episode 上先检查加载 bundle、episode reset、动作队列/恢复动作积分和结束行为。短 smoke test 仅判断接口正确，不评价收益。

现有 scripts/eval_r2r.sh 写死 GPU 0,1,2,3 和每卡三进程，不能直接用作本轮启动器。应使用明确暴露已授权单卡的独立入口或直接调用 src/eval/eval.py。当前 config/vln_r2r.yaml 默认 val_unseen；开发评测必须使用独立 val_seen 配置，不修改默认配置去混淆已保存结果。运行前核对真实数据与场景是否齐全。

## 第二关：模型是否使用记忆

在固定 checkpoint、固定样本与相同当前图/指令/原历史图条件下，准备以下只改变长期记忆的干预：

- 正常历史记忆。
- 保留当前观测，仅移除更早 bank 条目，保持 300 个 memory 占位与生成设置一致。
- 使用受控错误历史作负对照，保持时间、条目数量等条件可比，明确这是分布外诊断。
- pose/time 消融、FIFO/FRME 对照作为后续独立机制实验。

先用 teacher-forced 动作目标 NLL、动作分布变化检查敏感性，再用相同 episode 的闭环 SR/SPL 检查是否有益。输出变化只证明敏感，不证明利用正确；正常记忆相对干预表现更好才支持“记忆有帮助”。避免仅统计 attention 权重或梯度就作结论。保留原来的历史图输入是既定实验条件，应明确其可能稀释长期记忆的增益。

## 第三关：是否比原模型涨分

至少四组：原 PanoVLN；相同 checkpoint/导航数据/预算的 LoRA 继续训练；完整 memory 仅导航；完整 memory 加摘要。记录实际导航/摘要样本数、目标 token 数和 optimizer updates，摘要带来的额外计算需披露。优先用 val_seen 或固定开发集选配置，固定配置后报告 val_unseen，不根据最终测试集反复调参。

控制 checkpoint 来源、episode 清单、seed、图像分辨率、原历史图数量、CGE、STOP、碰撞恢复与动作执行预算。每个条件单独输出目录，因为评测会跳过目录中已存在的 episode。失败或未完成 episode 不能静默排除；每次比较先核对全量 episode 集合与缺失指标。

可复用 src/eval/analyze_results.py 的 --baseline-path 配对 bootstrap。现有实现取 episode 交集，因此调用前必须额外核对两组完整 episode 集合相等，不能把交集上的改善冒充全量改善。报告 SR、SPL、NE、nDTW、配对差值及 95% 区间，同时报告延迟、峰值显存和按预先规定长度分组的结果。小样本仅做筛查；区间跨零应如实报告证据不足。

## 三个研究方向的递进实验

1. 全景空间对齐：当前是帧级 odometry 编码；若实现 token 方位对齐，必须处理 ToMe 后方位聚合与 FRME 跨帧合并，不能直接旋转已经失去网格关系的序列。先检查转向前后地标对应，再比较导航收益。
2. 指令感知选择：当前 queries 仅由 learned latent、pose 和 time 构成。候选实现应显式加入指令条件，分别比较只改 retrieval 与同时改变保留策略；已丢弃的信息不能靠 query 恢复。
3. 进度监督：当前完整轨迹摘要不等于进度预测。只有可靠子指令—帧映射才可构造已完成/当前/剩余目标；不能用时间百分比伪造语义进度。STOP 的额外机制需要独立提前停止/超时等诊断。

以上是后续实验协议，并非已执行结果或新增 GPU 授权。若当前机制无收益，记录失败并按开发集证据选下一项改动，不同时叠加三项导致无法归因。


## 严格比较入口

新增 scripts/compare_memvln_results.py（容器内CPU运行），要求预先规定的 --expected-episodes，核对两组episode集合相等及SR/SPL/NE/nDTW有效性，拒绝损坏JSON和冲突重复记录。不会修改输入或覆盖旧报告。实际评测配置一致性仍需实验清单核对。

命令模板：`python scripts/compare_memvln_results.py --baseline outputs/eval/baseline --candidate outputs/eval/memory --expected-episodes 778 --output outputs/eval/paired_memvln.json`。778仅适用于完整R2R val_seen，不应给其他数据集照抄。CPU人工数据已验证已知差值/区间及6种无效结果拒绝，临时数据自动删除；尚无真实导航比较结果。
