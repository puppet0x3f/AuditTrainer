# B3 回报估计稳定性：等执行预算的广采样与重复续跑

状态：**COMPLETE（2026-09-14）**。本轮推进任务覆盖对照之后的第二项：
在相同执行预算下，更可靠的回报估计是否值得减少分解候选的数量？

**本轮重复组的优势估计均方误差下降 62.3%，自然通过率为 49/60，高于广采样组和起点 B3 的 48/60。**
相对广采样组为 1 胜、0 负、59 平，新增成功对应首次合法计划中的责任细化。
这是相同执行预算下的正向 pilot 结果；证据范围为一条训练随机链及既有 TRAIN 任务。

## 预定设计

两组从上一轮轮换 B3 的相同最终权重及 Adam 第 50 步继续，各新增 10 次更新。
KL 参考仍为共同结构 SFT，固定 F01，保持统一 B3 优势、奖励、学习率、clip、KL/SFT 权重、
全局缩放及梯度上限不变。根规划、深度 1、关闭修复；既有协议重试仍允许。
没有新增 SFT 阶段。保持示范从原偏移 150 继续，两组每次更新均累积相同 6 根，
使用相同顺序的 60 个保持示范槽位；正文损失按各自实际候选数取平均。

| 条件 | 广采样 B3_wide | 重复续跑 B3_repeat |
| --- | ---: | ---: |
| TRAIN 根数 | 6 | 6 |
| 每根 ATOMIC 候选 | 1 | 1 |
| 每根 SPLIT 候选 | 8 或 9 | 3 |
| 每候选续跑 | 2 | 4 |
| 每根独立自然基线 | 1 | 4 次取平均 |
| 候选续跑总数 | 114 | 96 |
| 自然基线总数 | 6 | 24 |
| 训练轨迹总数 | **120** | **120** |
| 新增 optimizer 更新 | **10** | **10** |

广采样组将原 B3 的 M=2、单次基线配置用于更多候选，从而花完与重复组相同的执行预算。
它不是上一轮 K=3 配置的原样重跑。本对照检验执行预算分配，不能将全部差异归因于单独改变 M 或基线次数。
三个根使用 K=8，三个使用 K=9，分别消耗 19、21 条轨迹；重复组每根 20 条。
总预算严格相同，每根预算并不逐项相等，任务及 K 的分配在新运行前固定。

六根来自已有 TRAIN 校准与采样：时间去重→整除汇总、字符展开→切块、区间占用→整除汇总、
库存准备→节点筛选、库存准备→账本汇总、账本准备→节点筛选。
前四根在上一轮固定计划校准中重复出现分配差异，后两根提供过原始合法分配比较或明显的续跑波动。
复用已有任务、组件与私有检查，不增加算法家族，不依据本轮结果再筛选任务或计划。

统一优势为 `A(s,a)=Q_hat(s,a)-V_hat_independent(s)`。
广采样与重复组的前 3 个 SPLIT 使用相同候选种子，前两次候选续跑及第一次基线也配对；
为 M=4 及更多候选预留互不重叠的根种子区间。失败、截断、重复及初始注册失败均保留。

## 独立回报检查

重复组实际采样的 ATOMIC 与前 3 个 SPLIT 保持不变，每个再执行 4 次独立续跑；
每根另执行 4 次独立自然基线，共 **120 条检查轨迹**，没有 optimizer 更新。
这些回报既不进入训练，也不用于挑选 checkpoint。

在相同的 6 个状态、24 个完整动作上，对比两种训练估计与独立检查均值的：

- Q 的均方差；
- 自然基线均值的均方差；
- 统一优势的均方差及包含零值的符号不一致数。

独立检查本身只有 4 次重复，是有噪声的参考，不是真实 Q；同一根的动作也不是独立任务样本。
广采样组额外候选进入其训练，但不混进相同动作的估计误差比较。
另保留首次计划注册后成功、首次计划失败后恢复成功的计数，避免把恢复收益解释成初始分配质量。

## 回报估计与训练信号结果

240 条训练轨迹及 120 条独立检查轨迹均已完成。全部 6 个状态及 24 个共同候选的实际动作一致。
相对于独立四次检查均值，重复组的优势均方误差下降 **62.3%**。

| 相同 24 个动作的估计诊断 | B3_wide | B3_repeat |
| --- | ---: | ---: |
| Q 均方误差 | 0.041667 | 0.015625 |
| 自然基线均方误差（6 根） | 0.281250 | 0.104167 |
| 统一优势均方误差 | **0.359375** | **0.135417** |
| 优势符号不一致，包含零值 | 17/24 | 16/24 |

符号不一致的补充分解：两组均有 16 个“零值与非零值”的差异；两端都非零但方向相反的情况为 1→0。
因此，不能把 17/24 或 16/24 解读为同样数量的正负方向错误。
广采样组的六个单次自然基线恰好全部为 1；重复组分别为 0.5、0.75、0.75、1、0.25、1，
独立检查分别为 0.25、1、0.75、0.5、0.25、0.5，顺序与配置中的六根一致。
本轮单次基线高估较明显，增加重复缩小了估计差距；这些数值仍是相对于有噪声的独立检查。

| 原始训练信号 | B3_wide | B3_repeat |
| --- | ---: | ---: |
| 模式均值有差异的任务 | 5/6 | 4/6 |
| SPLIT 候选回报有差异的任务 | 5/6 | 4/6 |
| 已注册、局部检查覆盖的不同责任分配有回报差异 | **5/6** | **3/6** |
| 首次注册成功的 SPLIT 候选 | 39/51 | 12/18 |
| 首次注册且局部检查覆盖的 SPLIT 候选 | 35/51 | 11/18 |

广采样提供了更多不同责任分配之间的比较，重复续跑则改善了共同候选的回报估计。
最终自然复测中，重复组获得一条额外成功，详见下文。

## 自然评测

起点 B3（内部标识 B0）、B3_wide、B3_repeat 使用全部 12 个既有 TRAIN 根各 5 个相同新种子，
共 **180 条自然轨迹**；每组另执行 12 个重点根条件计划探针，共 36 个。
主比较是 B3_repeat 对 B3_wide 的通过率及配对胜负，同时报告新增更新的六根与其余六根。
全部属于 TRAIN，且只有一条训练随机链；不能作为未见任务泛化或多训练种子稳定性结论。

| 方法 | 本轮更新的六根 | 本轮未采样的其余六根 | 总体通过 |
| --- | ---: | ---: | ---: |
| 起点 B3（内部 B0） | 19/30 | 29/30 | 48/60 |
| B3_wide | 19/30 | 29/30 | 48/60 |
| B3_repeat | 19/30 | 30/30 | **49/60** |

广采样组与起点的 60 条配对结果全部相同。重复组对两者均为 **1 胜、0 负、59 平**：
更新六根的 30 条配对全部相同，其余六根为 1 胜、0 负、29 平。
这里的起点是上一轮轮换 B3；本轮使用新种子，不能把 48/60 与上一轮的 52/60 直接比较。

### 新增成功来自责任细化

唯一变化为 `train_transfer_polynomial_polynomial_03`，种子 `1550717`。
该任务要求先合并同次数项、去除零系数并排序，再在 x=3 处求值并返回最高非零次数。

| 决策与执行 | 起点 B3 / B3_wide | B3_repeat |
| --- | --- | --- |
| 根模式及首次注册 | SPLIT，注册成功 | SPLIT，注册成功 |
| 子任务 | aggregate | prepare → aggregate |
| 父节点保留职责 | 解析、预处理、格式化 | 解析、格式化 |
| 最终功能检查 | 失败 | 通过 |

起点与广采样组的根组合代码完全相同：把 `aggregate(items)` 的结果误当成预处理数组，随后又尝试自行汇总。
重复组把 PREPARE 下放给独立子节点，明确 `prepare.return → aggregate.prepared`，父节点正确地串联两次调用。
三组的 aggregate 子组件源码完全相同；执行器固定，三条轨迹的首次计划均注册成功。
这条收益展示了更细的责任分配如何帮助固定执行器正确组合组件。
该根未参与本轮六根更新，但已出现在上一轮 12 根 TRAIN 训练中，因此属于本轮未采样任务上的单例迁移，不能称为完全未见任务泛化。

原始证据：[起点组合代码](../runs/refinement/B3_variance/evaluation/B0/natural/train_transfer_polynomial_polynomial_03/repeat_04/runtime/artifacts/N0/v1.py)、
[重复组计划](../runs/refinement/B3_variance/evaluation/B3_repeat/natural/train_transfer_polynomial_polynomial_03/repeat_04/runtime/plans/N0.v1.json)、
[重复组组合代码](../runs/refinement/B3_variance/evaluation/B3_repeat/natural/train_transfer_polynomial_polynomial_03/repeat_04/runtime/artifacts/N0/v1.py)。

### 模式、正文与恢复诊断

| 诊断 | 起点 B3 | B3_wide | B3_repeat |
| --- | ---: | ---: | ---: |
| 重点根固定观察的 P(SPLIT) | 20.1813% | 20.1813% | 20.1813% |
| 自然轨迹中已注册的根 SPLIT | 10/60 | 10/60 | 10/60 |
| 条件计划探针注册成功 | 12/12 | 12/12 | 12/12 |
| 探针注册且局部检查覆盖 | 12/12 | 11/12 | 12/12 |
| 探针 parent_aggregate / fused / 未匹配 | 9 / 1 / 2 | 8 / 1 / 3 | 9 / 1 / 2 |
| 首次计划注册后成功 | 47 | 47 | 48 |
| 首次计划被拒后恢复成功 | 1 | 1 | 1 |

重点根模式概率基本不变，条件探针的类别频数也未显示重复组相对起点的整体变化。
本轮自然收益体现在具体计划的责任分配及组合结果；新增成功没有增加协议重试恢复计数。
下一项仍按既定顺序测量模式与正文的实际加权梯度贡献，再决定是否调整模式权重。
该后续实验已完成：权重 4 为 48/60，原重复组为 49/60；本轮保留权重 1，见 [模式权重报告](B3_mode_weight.md)。

## 成本、模型与验证

本轮完成 **540 条完整轨迹**：240 条训练轨迹、120 条独立回报检查、180 条自然评测；
另完成 36 个条件计划探针、两组共 20 次新增 optimizer 更新。没有新增独立 SFT 阶段。
两组实际 Adam 状态均从 50 衔接到 60，原始 optimizer 文件仍为 50；全程固定共同 SFT 参考和 F01。

广采样组处理 51 个 SPLIT 正文，重复组处理 18 个，各重复训练 10 轮。
记录到的训练循环耗时分别为 3295.55 秒和 1887.19 秒；运行使用共享 GPU，耗时不能视为受控硬件速度比较。
本实验匹配完整执行轨迹数及更新次数，候选生成、训练计算量与墙钟时间没有匹配。
逐轨迹调用、token 和执行成本保存在原始记录及[聚合结果](../runs/refinement/B3_variance/result.json)中。

44 项定向测试通过；旧 PR1C 多轮与 B3 覆盖实验的评测、训练信号聚合结果保持一致。
实际采样种子、独立基线平均、24 个共同动作、数据隔离、Adam 衔接、保持示范顺序、模型冻结、
保存权重与评测回接均已核对。两组初始 teacher-forced 评分差及保存后重载差均为 0，
自然评测与独立检查均没有 optimizer 更新。
验证记录：[测试](../runs/refinement/B3_variance/pytest.log)、[兼容性](../runs/refinement/B3_variance/compatibility_checks.log)、
[采样](../runs/refinement/B3_variance/collection_checks.log)、[训练](../runs/refinement/B3_variance/training_checks.log)、
[评测](../runs/refinement/B3_variance/evaluation_checks.log)。

重复组首次进程在启动阶段被终止，当时尚无采样或更新；重启后完成了全部固定运行。
额外加载开销未计入训练循环耗时，原始[启动日志](../runs/refinement/B3_variance/collect_repeat_attempt1_killed.log)已保留。

模型：[B3_wide](../runs/refinement/B3_variance/B3_wide/final_adapter)、
[B3_repeat](../runs/refinement/B3_variance/B3_repeat/final_adapter)。

## 实现与运行

配置：[refinement_variance.yaml](../configs/refinement_variance.yaml)。
实现：[refinement_variance.py](../src/audittrainer/refinement_variance.py)。
复用原采样器、B3 更新器和自然评测器，增加可配置候选数、重复次数及独立基线平均，旧配置默认值保持一致。

```bash
python -m audittrainer.refinement_variance prepare
python -m audittrainer.refinement_pr1 collect --config runs/refinement/B3_variance/arms/wide/config.yaml --method B3
python -m audittrainer.refinement_pr1 collect --config runs/refinement/B3_variance/arms/repeat/config.yaml --method B3
python -m audittrainer.refinement_variance prepare-audit
python -m audittrainer.refinement_pr1 collect --config runs/refinement/B3_variance/arms/audit/config.yaml --method B3
python -m audittrainer.refinement_variance train --arm wide
python -m audittrainer.refinement_variance train --arm repeat
python -m audittrainer.refinement_variance evaluate --method B0
python -m audittrainer.refinement_variance evaluate --method B3_wide
python -m audittrainer.refinement_variance evaluate --method B3_repeat
python -m audittrainer.refinement_variance report
```

使用 audittrainer conda 环境、`OMP_NUM_THREADS=4`，通过 `CUDA_VISIBLE_DEVICES` 指定设备。
复现时使用新输出目录，保留已完成的固定任务、模型与运行记录。
