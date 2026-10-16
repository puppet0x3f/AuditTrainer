# 根返回契约保持修复

上轮 transfer_sft 已将两个新契约的课程、开发根成功各从 0/4 提升到 4/4，但加权路径保持任务出现一次退步。失败根把直属 compute 返回的 `queries` 当作 prepare 返回的 `distances`，重复聚合并触发 KeyError。本轮训练根正确消费已聚合结果，并检查新旧能力同时保持。

## 固定设计

- 从 transfer_sft 最终执行器复制新适配器；complex_fit、transfer_sft 原件、根/内部门控、MULTI_BODY 均保持。
- 使用已有四个复杂算法 TRAIN 任务、两种责任结构的 8 条根 write_code 示例。将 observation 与目标动作中的 `_compute_child` 一致改名为 `_prepare_child`，直属子契约、函数符号 compute、返回职责和输入均不变。训练根依据契约消费结果，不因别名而重复承担聚合。
- 保持上轮全部 88 条监督（含新契约的 14 条），未将失败开发轨迹或评分用例入训。
- 固定 44 步、学习率 1e-5、种子 551000713；每步 2 条根目标＋2 条保持。每条目标 11 次曝光，88 条保持各一次。只验收最终检查点。
- 新增 24 条自主执行，与上轮 transfer_sft 的 24 条缓存结果配对；相同计划、任务、提示和种子。两组新契约各4条，8个旧任务共16条。缓存基线为4/4、4/4、15/16。
- 预定通过条件：新契约8条与旧任务16条全部通过。否则保留候选、记录实际缺口，不按成绩追加训练或挑选检查点。

本轮是已用开发状态的保持修复验收，不是独立泛化验证，也不是新采样 SPLIT 价值。别名变化是最小的有针对性训练干预；若成功，不能单独归因于别名增强，因为本轮还包含保持训练。

## 结果

44 步训练与新增 **24 条自主执行全部完成**。根目标各 11 次曝光，88 条保持样本各一次；检查点重载最大差 0。原始适配器、训练数据与缓存基线哈希保持，24 组首次 IMPLEMENT 观察一致。

| 根成功 | 上轮 transfer_sft 缓存 | 本轮 root_retention |
| --- | ---: | ---: |
| 新契约课程 | 4/4 | **4/4** |
| 已用新契约开发 | 4/4 | **4/4** |
| 8 个旧任务保持 | 15/16 | **16/16** |
| 合计 | 23/24 | **24/24** |

配对 **1 胜、0 负、23 平**。加权路径 **1/2→2/2**，其余七个旧任务均保持 2/2。新契约 prepare/compute 私有测试及根实际调用均保持 8/8。两组均 0 协议拒绝、0 编排/评分错误；轨迹输入＋输出 token 总数 584,278→584,137，基本持平。新增模型执行24条，基线24条复用原产物，没有重跑或覆盖失败成绩。

原失败种子 542001113 自主恢复。新生成根将 `compute` 导入为 `_compute_child`，调用后直接 JSON 序列化完整结果；不再读取不存在的 `distances` 或重复聚合。这里证明本轮训练组合恢复了根职责履行，不能单独断言模型在任意强制别名下都已正确，也不能将收益独立归因于别名增强。

- [恢复后的根源码](../runs/refinement/root_retention/evaluation/AFTER/validation_internal_complex_weighted_routes/repeat_00/final_program/artifacts/N0/v1.py)

## 决定

**保持修复通过，可以结束本轮局部修复，进入新执行器下的分解价值重估。** 使用 `runs/refinement/root_retention/training/final_adapter` 作为下一轮冻结执行器候选；保留原 complex_fit、transfer_sft 供对照，根/内部门控和 BODY 不更新。

下一步在既有四个新契约内部状态（路径、库存、重复覆盖、多项式联合求值）保留根计划，比较 ATOMIC 与冻结 BODY 新采样 SPLIT，各4种子，共32条新执行。两种模式均使用同一新执行器，纳入拒绝、重试、失败与成本；内部 ATOMIC 仍保留根分解。根据当前平均执行价值再决定门控校准，不能继续复用旧执行器下库存等任务的估值，也不能将本轮固定作者计划 24/24 直接当作新采样 SPLIT 的价值。

## 产物

两项定向测试通过：别名变化前后契约保持、目标根消费已聚合对象的可执行检查，以及完整保持数据和缓存配对设计。

- [完整结果](../runs/refinement/root_retention/result.json)
- [配对与冻结检查](../runs/refinement/root_retention/validation.json)
- [实现](../src/audittrainer/refinement_root_retention.py)
- [固定配置](../runs/refinement/root_retention/request.json)
- [测试](../tests/test_refinement_root_retention.py)

```bash
conda run -n audittrainer python -m audittrainer.refinement_root_retention prepare
conda run -n audittrainer python -m audittrainer.refinement_root_retention train
# 分别运行 shard 0、1、2
conda run -n audittrainer python -m audittrainer.refinement_root_retention evaluate --shard 0
conda run -n audittrainer python -m audittrainer.refinement_root_retention report
```
