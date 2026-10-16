# 独立内部门控：有限收益校准与新种子自然执行

本轮只训练第一次内部 `compute` 的模式选择。从原 MULTI_BODY 复制独立 LoRA，根门控、所有 SPLIT 正文、complex_fit 执行器及后续节点/重试策略冻结。固定已注册根计划，因此内部 ATOMIC 表示在 compute 停止细分，根任务仍然分解。

## 设计

校准使用 complex_refresh 的当前执行器、当前正文新采样平均值。四个状态等量曝光，32 步、每步四状态、学习率 1e-5；单一固定预算最终检查点。

| 已用 TRAIN 开发状态 | ATOMIC | SPLIT 原始成绩 | 处理 | 目标 SPLIT 概率 |
|---|---:|---:|---|---:|
| 加权路径 | 4/4 | 4/4 | 保持原概率 | 18.24% |
| 库存事件 | 3/4 | 3/4 | 含评分启动 ERROR；保持原概率 | 16.45% |
| 覆盖查询 | 1/4 | 4/4 | 正收益校准 | 99.75% |
| 多项式乘积求导 | 2/4 | 4/4 | 正收益校准 | 97.07% |

正收益状态的软目标为 `sigmoid(logit(p_body) + (QS-QA)/0.1)`，对应期望根成功减去到原模式先验的 KL 约束的最优两模式分布，再用交叉熵蒸馏。目标概率不是执行成功率。持平/不确定状态使用保持目标，不赋予强负标签；库存评分异常不进入方向性回报。成本权重保持 0，计费 token 单独报告。

验收：四状态 × 四个新种子 × BEFORE/AFTER，共 32 条。种子 `513000713 + state_index*100 + repeat`，与价值采样不同；模式自然采样，SPLIT 正文重新生成，保留所有拒绝、重试、失败与评分异常。两条件重放同一根计划，并核对第一次 compute 的完整观察一致。后续轨迹按冻结原策略继续，因此此次估值与干预决策范围一致。

这些状态属于已用开发，算法和契约已进入执行器课程；本轮测已知收益状态的调用能力，不作为跨任务泛化证据。

## 门控与冻结检查

32 步、128 次状态曝光完成；最终门控已实现正收益状态与保持状态的差异化选择：

| 状态 | 校准前 P(SPLIT) | 校准后 P(SPLIT) |
|---|---:|---:|
| 加权路径 | 18.24% | 16.45% |
| 库存事件 | 16.45% | 18.24% |
| 覆盖查询 | 18.24% | **99.59%** |
| 多项式乘积求导 | 18.24% | **98.41%** |

两个保持状态最大漂移 1.79 个百分点，没有随正收益状态整体升高。仅独立内部门控参数改变；根门控、正文与执行器指纹保持，四个根状态的门控概率完全一致。四份配对 SPLIT 正文与训练前保存样本逐 token 一致。检查点重载最大 logit 差为 0。

## 自然执行结果

**新种子配对根成功从 10/16（62.5%）提升到 15/16（93.75%），增加 31.25 个百分点；5 胜、0 负、11 平。** 校准后的新增成功来自覆盖查询四条、多项式一条。两个保持状态未回退。

| 状态 | 根成功 BEFORE→AFTER | 首次内部 SPLIT 次数 | 平均轨迹 token BEFORE→AFTER |
|---|---:|---:|---:|
| 加权路径 | 4/4 → 4/4 | 2/4 → 1/4 | 22,680 → 19,838 |
| 库存事件 | 3/4 → 3/4 | 0/4 → 0/4 | 17,829 → 17,828 |
| 覆盖查询 | **0/4 → 4/4** | 0/4 → 4/4 | 18,476 → 30,807 |
| 多项式乘积求导 | **3/4 → 4/4** | 2/4 → 4/4 | 29,420 → 33,646 |
| 合计/均值 | **10/16 → 15/16** | **4/16 → 9/16** | **22,101 → 25,530** |

平均轨迹 token 增加 **15.5%**，这是调用更多有收益分解的代价；成本未混入训练目标。上述记录包含两条件相同的重放根计划开销。不能将本轮解释为“分解数量不变、只改变位置”：实际增加了五次分解，其中覆盖查询增加四次，多项式增加两次，加权路径减少一次。

校准后的九份首次 SPLIT 均为实质责任分配：六份 `parent_aggregate`、三份 `pipeline`；无同契约重复委托。校准前四份为两份 `parent_aggregate`、两份 `pipeline`。本轮执行体现的是内部计算责任继续细分后带来的根成功收益。

最终 32 条均为 COMPLETE，无评分或编排错误。保留所有模型失败：库存同一配对种子前后均自主弃权，仍计失败；BEFORE 含一次 UNVERIFIED_CONSUMPTION 与四次 UNAPPROVED_IMPORT 拒绝，AFTER 无协议拒绝。16 组完整首次内部观察及根计划逐组相同。

## 决定

**保留独立内部门控作为候选，结束这四个状态的内部调用可学习性诊断。** 已有证据支持：冻结责任正文与执行器后，内部门控能够区分正收益状态与保持状态，并在新执行种子下提高完整根任务成功率。收益不仅是局部通过，也不是无实质细化的同契约转发。

本轮训练和验收共享四个开发状态，固定根计划且仅校准第一次 compute 决策，因此尚不证明未训练状态上的门控泛化，也未评估根与内部门控全自然联动。下一步冻结本轮候选，在未用于门控训练的内部状态上比较 ATOMIC、当前新采样 SPLIT 和自然选择，确认新的正收益是否存在及是否被调用；暂不扩大 RL 预算。

完整结果：[result.json](../runs/refinement/internal_gate/result.json)；配对与责任结构核对：[validation.json](../runs/refinement/internal_gate/validation.json)。

## 验证与复现

五项定向测试通过，覆盖收益目标/评分异常处理、首次内部决策与后续重试的路由隔离、完整状态匹配、冻结配置及新种子、输出身份字段协议。

验收首次启动暴露一处新增身份字段的封装类型错误：协议要求字符串，封装传入对象。已改为 JSON 字符串；原始错误轨迹保存在 `runs/refinement/internal_gate/orchestration_errors/identity_type`，两个受影响分片以原种子重新执行。此编排错误不算模型失败；最终模型拒绝、弃权、重试和功能失败全部保留。

原始配置：[request.json](../runs/refinement/internal_gate/request.json)；训练：[training/result.json](../runs/refinement/internal_gate/training/result.json)；实现：[refinement_internal_gate.py](../src/audittrainer/refinement_internal_gate.py)。

```bash
conda run -n audittrainer python -m audittrainer.refinement_internal_gate prepare
conda run -n audittrainer python -m audittrainer.refinement_internal_gate train
# 每个 method=BEFORE/AFTER、shard=0/1/2 各执行一次：
conda run -n audittrainer python -m audittrainer.refinement_internal_gate natural --method BEFORE --shard 0
conda run -n audittrainer python -m audittrainer.refinement_internal_gate report
conda run -n audittrainer pytest -q tests/test_refinement_internal_gate.py
```
训练目录存在时不会覆盖训练；自然执行读取已有完整 case_result 后跳过重新生成。

