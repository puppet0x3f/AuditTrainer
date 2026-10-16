# 复杂任务上的compute SFT前后执行器对照

2026-09-18。承接[复杂内部任务诊断](complex_values.md)。本轮检验最近compute SFT是否与新任务失败有关，不将复杂任务双零直接归因于训练退步。

## 固定方案

BEFORE为`runs/refinement/handoff_sft/training/final_adapter`，AFTER为`runs/refinement/compute_sft/training/final_adapter`。比较的是最近32步compute履约SFT之前和之后，不是原始F01与当前模型。

沿用加权路径、库存事件、覆盖查询、多项式乘积求导四个已用开发任务，各2个相同种子、两套执行器，共16条。执行种子481000713＋任务序号×100＋重复序号。固定根fused计划、内部compute采用parent_aggregate、prepare叶节点ATOMIC；双方使用完全相同的公开任务、计划、预算及原提示。所有消费和源代码自主生成，不人工改写动作字段或代码。拒绝、重试及失败全部计入。

加权路径在执行前就对两组使用同一增强私有测试，含原测试和上轮边序反例；其他任务测试不变。增强私有用例不进入模型观察。作者按本轮固定计划完整运行4/4通过可见/私有根测试，两项定向测试通过。

主指标为最终根成功；辅助检查prepare、compute局部私有通过和根实际调用。局部通过仍仅指通过当前有限测试。双零不能证明两个执行器等价或排除其他场景退步。这里没有ATOMIC/SPLIT或自然门控对照，不估计门控价值。

## 结果

**16条全部完成，当前未发现最近compute SFT造成根成功率退步的配对证据；新旧执行器在这些任务上均未解决局部计算。**

| 任务 | BEFORE根成功 | AFTER根成功 | prepare局部私有 B / A | compute局部私有 B / A |
|---|---:|---:|---:|---:|
| 加权路径 | 0/2 | 0/2 | 0/2、0/2 | 0/2、0/2 |
| 库存事件 | 0/2 | 0/2 | 0/2、0/2 | 0/2、0/2 |
| 覆盖查询 | 0/2 | 0/2 | 0/2、0/2 | 0/2、0/2 |
| 多项式乘积求导 | 0/2 | 0/2 | 0/2、0/2 | 0/2、0/2 |
| 合计 | **0/8** | **0/8** | **0/8、0/8** | **0/8、0/8** |

8组配对0胜、0负、8平。根实际调用compute合计两组均5/8，但逐任务不同：BEFORE为2、2、0、1，AFTER为0、2、1、2。这不是两个执行器行为等价的证据，更不能将提交根程序误作功能正确。

## 训练前已经存在的具体缺口

旧执行器的保存代码已经包含以下错误，说明它们不能全部归因于最近的compute SFT：

- 路径prepare读取输入不存在的`nodes`字段；父聚合将节点和查询阈值混用，并对二元结果做三元解包。见[旧路径repeat_00](../runs/refinement/complex_executor/BEFORE/validation_internal_complex_weighted_routes/repeat_00/runtime/trace.jsonl)。
- 库存prepare按(time,SKU)而非(time,原索引)排序，记录了排序后的拒绝位置，输出结构也不符；父聚合引用未定义targets并递归调用自己。见[旧库存repeat_00](../runs/refinement/complex_executor/BEFORE/validation_internal_complex_stock_events/repeat_00/runtime/trace.jsonl)。
- 覆盖prepare仅调用未定义的`_prepare`；父节点未正确实现分段交集与重数计算。见[旧覆盖repeat_00](../runs/refinement/complex_executor/BEFORE/validation_internal_complex_coverage_queries/repeat_00/runtime/trace.jsonl)。
- 多项式prepare把输入对象误作多项式列表，乘法与字段提取不正确；compute调用未定义`_aggregate`。见[旧多项式repeat_00](../runs/refinement/complex_executor/BEFORE/validation_internal_complex_polynomial_chain/repeat_00/runtime/trace.jsonl)。

新模型也未通过prepare或compute私有测试。失败不只是根节点交接，局部算法与契约履行本身就不完整。两组仍有协议失败，均保留：BEFORE为UNVERIFIED_CONSUMPTION 8、UNKNOWN_EVIDENCE 3、INVALID_SOURCE 1、UNAPPROVED_IMPORT 2；AFTER为UNVERIFIED_CONSUMPTION 7、UNAPPROVED_IMPORT 3、UNKNOWN_EVIDENCE 2、INVALID_SOURCE 2、CHILD_NOT_PENDING 1。此前局部交接修复的成功不能外推为任意新任务均不会发生绑定错误。

## 验证与结论边界

16条均COMPLETE、0基础设施错误、0参数更新。8组注册计划及首次IMPLEMENT完整观察逐项一致；门控、正文、两套执行器文件哈希保持，运行前后指纹保持。两项定向测试通过。加权路径边序反例在本轮执行前统一纳入两组私有根测试，没有再接受上轮单遍松弛的表面成功。

**这些新任务的履约缺口在最近compute SFT之前已经存在。** 本轮没有观察到“旧执行器成功而新执行器失败”的根结果，因而不支持为了这4个任务回退到旧执行器。但每任务仅2个种子，且根与局部均处于零成功，不能证明训练完全无害、两个模型等价，或更早SFT/底座不存在其他问题。

## 推进决定

保留compute_sft候选及旧模型对照，保留此前简单任务、转换和嵌套交接的已验证进展；暂不训练内部门控，也不将本轮双零变成ATOMIC标签。

**下一步转向多结构TRAIN执行履约课程。** 在独立TRAIN变体上学习完整算法、正确输入结构和父节点保留职责，同时保持既有转换与直属子交接监督；不能只监督“调用一个未实现的辅助函数”。验收继续包含自主子实现、父组合和最终根，并保留本轮开发任务及边序反例，不直接把这些开发轨迹当训练样本。确认执行能力改善后，再用当前正文新采样SPLIT与ATOMIC重估收益。

本轮已完成16条对照，未训练新模型、未改提示或协议、未增加其他基线。比较固定作者计划的执行器性能，不等于测量当前采样分解或自然门控收益。

产物：[完整结果](../runs/refinement/complex_executor/result.json)、[验证记录](../runs/refinement/complex_executor/validation.json)。实现入口`python -m audittrainer.refinement_complex_executor prepare/run/report`，run指定`--method BEFORE/AFTER --shard 0/1`。

