# 无 vLLM 的性能优化与验收

用户确认：只在GPU3空闲时验证，不停止其他任务，不读backup；保持模型/PanoVGGT/RL算法；每5个采样更新批次保存完整checkpoint，接受中断重做最多4批。

## 已完成的第一阶段

checkpoint_audit.compare_tensor_chunks：全元素检查分块执行，不抽样，不减少几何/非有限值/形状/原始dtype舍入检查。7项audit测试通过（含末尾块NaN与损坏）。CPU内存驻留16M元素BF16/FP32张量、2线程合成基准，原实现约0.116s，分块约0.051s，中位速度比2.26；不是全checkpoint磁盘实测。

batching.collate_inputs：左填充文本，拼接图像数据，保留每例image_num_images和局部image_current_index，geometry严格FP32。异长历史、非法字段、低精度geometry测试通过（与audit合计9项）。这是未启用的原型，必须通过GPU概率/梯度/几何映射/速度验证。

## 后续依赖GPU验证的阶段

1. GPU3空闲（当前仍占用约17GB）后，原生模型batch=1/2测试单帧及不同历史长度，检查每个样本几何特征映射、有效token log-prob、DAPO token-mean损失及梯度；计时需预热和同步，记录峰值显存。不能简单提高asyncio concurrency：当前policy.generate同步阻塞事件循环，必须配合单模型动态批量生成队列；禁止多个线程同时调用有Pano运行时字段的模型。
2. 采样采用SimpleVLN sliding-window思路：多个独立Habitat session，集中推理队列收集待决策请求，统一批量前向后分别执行环境；保留每组8轨迹、标准差过滤、同策略版本采样、原奖励和动作执行语义。错误需让整批失败，不能静默丢轨迹。
3. actor决策microbatch先小规模2，再按token预算测试；padding不参与loss和分母，保持128-decision mini-batch与原优化器step边界、clip及token-mean归约。不能仅用no_sync假装批处理；FSDP no_sync会积累完整梯度，必须额外验显存。
4. 每5批保存需要常驻actor/optimizer与内存权重同步。旧controller每批退出actor并从磁盘加载下一批，因此仅跳过save会丢Adam或读旧策略，禁止这样实现。需要在GPU实测内存预算后决定常驻actor与采样模型共存/卸载的实现；batch1~4仅内存推进，batch5/正常结束/数据耗尽导出模型+Adam+RNG+游标；中断从最后原子提交checkpoint重做。基于最后持久化policy_version计算累计更新数量，不能继续要求保存前后版本永远+1。
5. 保留离线W&B与真实policy版本，恢复绑定manifest/精度/算法配置和最后持久化游标；CPU mock验证5批保存、强制中断、恢复重做、最后不足5批保存；GPU验证实际优化器步数连续及保存恢复。

## 当前状态

未启动GPU基准，未修改正在运行的任务、旧运行目录或默认采样/训练批处理行为。每5批持久化尚未接通；新batching原型尚未用于生产。不宣称整个优化完成或整轮提速已验证。下一步先通过GPU3批处理正确性与内存测试，再接入常驻执行路径，避免用尚未知的显存预算设计生产切换。
