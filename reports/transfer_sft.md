# 新契约执行履约 SFT 与固定计划配对验收

从 complex_fit 执行器复制独立适配器，学习重复覆盖、多项式联合求值两个新契约的完整履约。根门控、内部门控、MULTI_BODY 与原执行器保持不变。

## 固定设计

- 两个独立 TRAIN 输入变体，每个包含六组输入，均不同于原开发测试。准备、父聚合、compute 与根组合的完整 write_code 动作监督，覆盖 `parent_aggregate`、`pipeline` 两种责任结构，共 **14 条目标**。
- 原 complex_sft 的 **74 条样本全部作为保持监督**，覆盖原完整算法、转换、组合与交接。没有将本轮开发失败轨迹或人工修正轨迹直接作为监督样本。
- 固定 **56 步**，学习率 `2e-5`，每步 2 条目标＋2 条保持；每条新目标 **8 次曝光**，目标/保持各 112 次。只使用最终检查点，不按验收成绩选检查点。
- 新旧执行器各 **24 条**：课程 2 状态×2 种子，已用开发 2 状态×2 种子，保持 8 状态×2 种子；共 **48 条配对执行**。同一状态使用相同固定计划和种子，执行器自主实现、消费、验证和组合。拒绝、重试与失败完整计入。
- 保持验收包含原四个复杂算法任务，以及 calendar、chunks、系数转换、路径转换四个既有场景。

本轮原开发契约现在已通过独立 TRAIN 输入得到监督；开发副本登记为 TRAIN 开发，原文件与测试不变、原开发轨迹不入训。本轮不是未见契约泛化，也不是当前正文新采样 SPLIT 的平均价值。

## 结果

训练已完成：56 步，目标与保持各 112 次曝光，每条目标恰好 8 次；最终适配器重载最大差为 0。

两个新契约的自主验收已完成：课程根成功 **0/4→4/4**，已用开发根成功 **0/4→4/4**。合计 prepare 私有测试 **2/8→8/8**，compute 私有测试 **0/8→8/8**，根实际调用 compute **3/8→8/8**。这些结果来自模型自主生成和交接，没有人工替换代码。

全部 **48 条执行完成**，最终结果如下。

| 验收组 | 原 complex_fit | 新 transfer_sft |
| --- | ---: | ---: |
| 新契约课程 | 0/4 | **4/4** |
| 已用新契约开发 | 0/4 | **4/4** |
| 8 个旧任务保持 | 16/16 | **15/16** |
| 合计 | 16/24 | **23/24** |

配对 **8 胜、1 负、15 平**。拒绝次数 22→0，全部轨迹输入＋输出 token 总数 637,618→584,278（减少约 8.4%）；拒绝、重试及失败均保留。48 条均 COMPLETE，无编排或评分错误。24 组首次 IMPLEMENT 观察完全一致；原执行器、根门控、内部门控和 BODY 文件保持不变，运行时冻结参数检查通过。

### 唯一保持失败

加权路径 `repeat_00`（种子 542001113）由通过变为失败，另一种子通过，其余七个保持任务均 2/2。模型生成的根 `solve` 调用了直属 compute，但将其返回的 `{'queries': ...}` 当成 prepare 的 `{'distances': ..., 'limits': ...}`，重复承担聚合并读取不存在的 `distances`，触发 `KeyError`。源码与评分证据支持这是根对直属返回契约的误用，不是评分基础设施异常，也不是越级消费绑定错误。不据两种子断言总体遗忘，但这一实际配对退步必须计入。

- [失败根代码](../runs/refinement/transfer_sft/evaluation/AFTER/validation_internal_complex_weighted_routes/repeat_00/final_program/artifacts/N0/v1.py)
- [私有评分证据](../runs/refinement/transfer_sft/evaluation/AFTER/validation_internal_complex_weighted_routes/repeat_00/final_program/evidence/SCORE_PRIVATE_5.json)

### 决定

**新契约履约学习成功，并已转化为自主根成功；旧任务完全保持的预设条件未通过。** 因此 `proceed_to_fresh_values=false`，保留 transfer_sft 为候选，暂不替换 complex_fit、不更新门控。下一步针对根消费已聚合结果的职责边界做有限保持修复与配对复验，同时保留新契约收益。通过后再用同一新执行器重采 ATOMIC 与当前 BODY 新生成 SPLIT 的平均价值；旧库存价值不能直接当成更新后执行器的价值。此次不根据验收结果追加训练或挑选检查点。

进度更新曾漏读加权路径首条失败；本报告以完整分片汇总为准，最终为 23/24，而非 24/24。

## 产物

三项定向测试通过，核对训练/开发输入分离、原有保持监督完整、目标曝光次数、固定预算、教师根测试与四套原始适配器哈希。

- [完整结果](../runs/refinement/transfer_sft/result.json)
- [配对与冻结检查](../runs/refinement/transfer_sft/validation.json)
- [固定配置](../runs/refinement/transfer_sft/request.json)
- [执行实现](../src/audittrainer/refinement_transfer_sft.py)
- [训练数据](../data/refinement/transfer_sft/training.jsonl)

```bash
conda run -n audittrainer python -m audittrainer.refinement_transfer_sft prepare
conda run -n audittrainer python -m audittrainer.refinement_transfer_sft train
# BEFORE/AFTER × shard=0/1/2，各运行一次：
conda run -n audittrainer python -m audittrainer.refinement_transfer_sft evaluate --method BEFORE --shard 0
conda run -n audittrainer python -m audittrainer.refinement_transfer_sft report
conda run -n audittrainer pytest -q tests/test_refinement_transfer_sft.py
```
