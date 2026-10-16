# 多结构完整算法执行器SFT

2026-09-18。承接[新旧执行器对照](complex_executor.md)。本轮完成固定预算训练和48条自主配对执行，未修改门控、正文或执行协议。

## 主要结果

**课程根成功0/8→2/8，开发根成功仍0/8，保持8/8→8/8。当前不能把课程局部进展推广成复杂任务执行已可靠。**

| 组别 | 原compute_sft | 新complex_sft |
|---|---:|---:|
| TRAIN课程 | 0/8 | 2/8 |
| 已用同类开发题 | 0/8 | 0/8 |
| 简单任务、转换与嵌套交接保持 | 8/8 | 8/8 |

24组配对：2胜、0负、22平。课程两次新增成功分别来自加权路径和库存事件，各1/2；覆盖查询和多项式乘积求导均0/2。开发四任务均0/2。

课程prepare局部私有通过0/8→3/8，compute局部0/8→2/8；开发prepare局部0/8→2/8，compute局部1/8→0/8。这包含一次旧模型局部通过而新模型未通过，不能将根结果无净负配对解释为所有局部能力均无退步。开发路径两条prepare均通过，但父聚合失败，说明一部分局部算法已改善，尚未形成完整执行收益。

## 训练与验收

从`runs/refinement/compute_sft/training/final_adapter`复制执行器，AdamW学习率2e-5、固定32步、128次曝光，每步2条新目标加2条保持样本。使用唯一最终检查点，不按开发结果选模型、不追加训练。根门控`gate_refresh`、MULTI_BODY正文冻结。

四类课程为加权路径、库存事件、覆盖查询、多项式乘积求导。每类采用新的TRAIN输入，分别生成parent_aggregate和pipeline两种教师轨迹，共8条完整教师执行全部通过根测试。28条write_code目标覆盖prepare完整算法、aggregate、compute保留职责/调用组合和根序列化；保持原compute_sft全部46条监督，含简单任务、转换与直属子消费。训练中每条新目标仅2–3次曝光，这是一轮小预算可学习性试验，课程本身尚未稳定拟合。

48条验收为4个课程根、4个已用开发根、4个保持根，各2种子、两套执行器。课程中路径/覆盖采用parent_aggregate，库存/多项式采用pipeline；开发统一parent_aggregate；保持为calendar→matrix、chunks→chunks及多项式/路径嵌套prepare。所有计划固定，执行器自主消费、实现和组合；拒绝、重试、截断及最终失败全部计入。不是自然规划或当前新采样SPLIT的价值估计。

**这里教授了与开发题相同的算法和契约，检验的是课程拟合及同类新输入执行，不能称未见算法/契约泛化。** 原开发轨迹、原测试样例和路径边序反例不进入监督。路径增强私有根测试在双方执行前固定保留。

最初评估入口发现同一算法族跨TRAIN/VALIDATION，被清单检查拒绝，尚未产生执行case。随后仅在本轮目录复制4个已用开发任务，将其登记为TRAIN开发范围，原文件和测试不变；新旧模型使用同一份复制任务。此调整避免把已教授算法族继续称独立VALIDATION。两次入口失败日志保留；最终48条均完整完成，无执行基础设施错误。

## 剩余的具体失败

- 路径：prepare可正确完成最短路；compute聚合中仍有对整数重复下标、错误生成器/排序及最远节点选择，最终根失败。见[开发路径轨迹](../runs/refinement/complex_sft/evaluation/AFTER/development/validation_internal_complex_weighted_routes/repeat_00/runtime/trace.jsonl)。
- 库存：开发prepare已学会按(time,原索引)排序，但遗漏只出现在targets中的SKU；课程另一条将三元事件按四元解包，aggregate又将targets列表当字典。见[开发库存轨迹](../runs/refinement/complex_sft/evaluation/AFTER/development/validation_internal_complex_stock_events/repeat_00/runtime/trace.jsonl)。
- 覆盖：仍有错误分段结构、未定义_segments和错误曝光量计算，且出现输出截断/协议失败。截断是本次模型执行失败，未剔除。
- 多项式：仍出现未定义_multiply、_derive、eval_poly以及错误乘积初值，未形成完整算法。见[开发多项式轨迹](../runs/refinement/complex_sft/evaluation/AFTER/development/validation_internal_complex_polynomial_chain/repeat_00/runtime/trace.jsonl)。

因此，失败不只是消费ID或最终包装，完整算法和父节点保留计算仍是主要未完成项。不能因为两次课程成功就将这些状态标为分解正收益。

## 验证和决定

12项定向测试通过；24组首次IMPLEMENT完整观察一致。模型运行前后指纹保持，旧执行器/门控/正文文件哈希保持；训练重载最大差0，原冻结执行器不变，训练样本哈希不变。私有局部检查为执行后诊断，不反馈给模型。

**保留原compute_sft作为当前执行器；complex_sft保留为实验分支，不推进价值重估或门控训练。** 本轮保持集未见退步，但开发根成功未改善，未达到预先规定的推进条件。

下一步先解决课程拟合：沿用这批TRAIN监督和旧能力保持，对完整prepare算法与父聚合做等量、充分曝光的固定预算正对照，先要求课程自主执行稳定，再复验已用开发题。每目标仅2–3次曝光的负结果不足以否定完整算法课程；也不应继续扩任务、将失败改成永久停止标签，或用人工修正替代模型成绩。待端到端改善后，再比较ATOMIC与当前正文新采样SPLIT。

产物：[完整结果](../runs/refinement/complex_sft/result.json)、[验证记录](../runs/refinement/complex_sft/validation.json)、[固定方案](../runs/refinement/complex_sft/request.json)。入口：`python -m audittrainer.refinement_complex_sft prepare/train/evaluate/report`；evaluate指定`--method BEFORE/AFTER --shard 0/1/2`。
