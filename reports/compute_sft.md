# 内部compute履约执行器SFT

2026-09-18。承接[固定产物因果诊断](compute_repair.md)，采用固定预算训练与新旧执行器配对验收。

## 固定方案

从handoff_sft执行器复制独立LoRA，根门控gate_refresh、正文和内部策略MULTI_BODY冻结。沿用原始执行提示、完整协议和已固定32步训练预算，学习率2e-5；每步2条目标、2条保持样本，合计128次曝光。不按评估结果选择checkpoint。

四个独立TRAIN变体为calendar→matrix、intervals→matrix、chunks→chunks、words→words，参数由旧开发任务的4改为6，并使用新输入。执行参考教师通过可见及私有根验证后，抽取10条aggregate/compute/solve写代码动作；不训练规划动作。保持数据为此前handoff_sft完整36条监督，涵盖转换、直属产物绑定、中间包装及原任务实现。教师输入仅为当时可见观察；私有测试用于验收，不进入模型观察。

目标包括：prepared已为数值列表时正确筛选求和；返回完整chunks/length对象；compute返回原生值，根负责JSON序列化。训练包含根实现，避免只学会内部输出而不能完成最终任务。

验收固定为48条完整模型执行：BEFORE/AFTER各24条，4个FIT变体×2种子、4个已用开发compute状态×2种子、4个保持任务×2种子。两类compute组采用相同的固定根fused及内部parent_aggregate/pipeline计划，叶节点ATOMIC；保持组采用多项式/路径的嵌套计划和两类原任务ATOMIC。所有消费、写代码和根完成均由执行器自主生成，没有人工代码修改或消费字段改写。模型失败、拒绝和重试均保留。

这是已知家族的训练及开发对照，不称为新家族泛化。固定正确计划的收益不等于当前正文新采样SPLIT均值；本轮不据此训练内部门控。先完成执行器评估，再决定重新采集A/S价值。

## 结果

**执行器履约SFT通过本轮自主验收。48条配对执行中，训练与开发组均由2/8提升到8/8，保持组8/8不变。**

| 组别 | 旧执行器根成功 | 新执行器根成功 | 说明 |
|---|---:|---:|---|
| FIT，新参数6及输入 | 2/8 | **8/8** | 本轮4个训练状态，各2种子 |
| DEVELOPMENT，原参数4 | 2/8 | **8/8** | 本轮未训练、此前已用于诊断的4个状态 |
| RETENTION | 8/8 | **8/8** | 多项式/路径嵌套交接各2条，原任务直接实现各2条 |
| 合计 | 12/24 | **24/24** | 12胜、0负、12平 |

FIT和DEVELOPMENT中，calendar、intervals、chunks均各0/2→2/2；words各2/2→2/2。两个组的compute局部私有通过数也分别2/8→8/8。新模型16/16均被根实际调用，旧模型分别为6/8和7/8，调用存在不等于正确。保持组没有compute节点，其compute局部与调用指标不适用。

生成代码核对显示：calendar当前节点完成保留的筛选求和；intervals由aggregate子节点处理prepared数值列表；chunks返回完整对象；compute保留原生结果，根完成紧凑JSON序列化。全部执行采用原提示，没有人工消费或代码干预。旧模型13次UNVERIFIED_CONSUMPTION拒绝已保留，新模型24条执行无协议拒绝。

## 验证与运行记录

- 完成32次更新、128次样本曝光，目标与保持各64次；固定最终checkpoint，重载logit最大差为0。
- 24组配对的已注册计划逐项一致；根门控、MULTI_BODY及原执行器文件哈希保持，评估期间参数指纹保持。
- 5项定向测试通过，覆盖训练/开发状态分离、完整保持监督、直属产物绑定、教师执行及责任结构。
- 最终48条均COMPLETE且无基础设施错误。运行过程中曾有3个GPU3工作进程在完成首条之前被SIGKILL，原因未确认；排除该GPU后按相同checkpoint与种子重跑。另一个刚开始的旧模型保持任务在收回GPU范围时被主动停止，未完成目录已归档；一个等待进程重新启动。没有丢弃任何已完成模型结果，未完成尝试不计作功能分数。详见[中断记录](../runs/refinement/compute_sft/interrupted/note.json)。

## 决策

**新执行器作为下一轮价值重估候选，继续冻结根门控与MULTI_BODY。** 本轮证明完整计划的自主履约改善，并保持有限测试中的转换、递归交接和原子执行能力。开发组是已知家族、已用诊断状态，不表述为封存新家族泛化。

下一步在这4个内部compute状态重新比较ATOMIC与当前正文新采样SPLIT，每模式4种子，共32条；保留完整拒绝、重试和失败。只在新的平均执行价值支持时推进独立内部门控。固定作者计划下的8/8不能代替新采样SPLIT均值，也不能证明内部细分优于直接实现；旧执行器的内部价值不继续当作当前标签。本轮未开展价值重采或门控训练。

候选适配器：`runs/refinement/compute_sft/training/final_adapter`，只放入executor槽位；原执行器`runs/refinement/handoff_sft/training/final_adapter`保留。实现入口：`python -m audittrainer.refinement_compute_sft prepare/train/evaluate/report`，evaluate使用`--method BEFORE/AFTER --shard 0/1/2`。

产物：[完整结果](../runs/refinement/compute_sft/result.json)、[验证记录](../runs/refinement/compute_sft/validation.json)、[训练记录](../runs/refinement/compute_sft/training/result.json)。
