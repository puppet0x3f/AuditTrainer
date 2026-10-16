# B3 首次规划信用分配对照

状态：COMPLETE（2026-09-16）。数据、预算和采用标准在训练与新评测前固定，全部预算已完成。

**结论：不采用“首次被拒则回报归零”作为当前 B3 的替代目标。** 两批的首次接受且最终成功数分别为 29→29、31→31；总体成功为 30→30、33→32。两批根规划重试均从 2 次降到 1 次，但主指标没有净增加，第二批还损失了一次终局成功。保留原 B3 的完整后续回报；本轮结果不支持据此扩大任务覆盖或进入递归训练。

## 问题与改动

上一轮 B3 的终局成功数在两批均超过成功计划 SFT，但“首次计划被接受且最终成功”为 35/48、37/48，成功计划 SFT 为 36/48、35/48。需要检验：把无效首次计划的恢复收益归零，能否稳定提高首次合法规划后的成功，而不牺牲总体完成率。

仅改变训练阶段的回报归属：

`R_first = R_full`（首次根计划被接受）；否则 `R_first = 0`。

`R_full` 保留原有最终成功、已确认违规及成本定义；首次被接受的负回报也不截断。每个候选仍使用原有四次续跑的平均回报，独立自然基线也按同一规则对四条记录重新计算。B3 优势仍为 `Q_i - V_independent`。无效、失败和重复候选全部保留，原始执行记录不修改。

该目标把原先衡量完整后续过程的收益改为衡量“首次被接受后的完整后续收益”，并不预设这种偏好更有效。评测继续允许原运行时的拒绝后重试。首次宿主接受也不等于语义职责已被完整验证。

## 固定训练与评测

- 复用 [成功计划 SFT 对照](B3_success_sft.md) 的两个 TRAIN 批次：91000713、92000713，三个根状态不变。每批 108 条候选续跑和 12 条独立基线，共复用 240 条；新增训练续跑为 **0**。
- 原 B3 复用既有最终检查点。两个新 B3_FIRST 模型均从共同结构 SFT 和空 Adam 开始，各更新 **20 次**，共新增 **40 次**，不从旧 RL 检查点继续。
- 学习率 1e-6、模式权重 1、clip 0.2、正文 token 求和、KL/SFT 保持权重各 0.02、全局缩放 1/256、累积三个状态、示范顺序及冻结执行器均保持原配置。根规划最大深度 1，代码修复关闭。
- 四个模型使用同样的三个状态、每状态 **16 个新执行种子**，共 **192 条自然评测**。种子为 `100600713 + sorted_root_index*1000 + repetition`；与复用训练数据及上一轮评测分离。不增加条件探针，不选择中间检查点，不根据中途结果改预算。

新实验目录通过只读用途的路径链接复用原始候选和基线，载入时重算优化用回报。原始 `case_result.json`、原 B3 检查点及既有报告不被重新计分覆盖。

## 预登记指标与采用标准

主指标为 **首次根计划被接受且最终通过功能检查** 的运行数，分母为每模型全部 48 条，不是仅在被接受样本中求成功率。同时报告最终成功数、首次接受数、根规划重试次数和首次被拒后成功数。

根规划重试次数定义为：轨迹中 `N0 / POLICY / NEW` 的动作次数减一，包含不能解析的首次输出，排除子节点规划及其他协议动作。首次被拒但未重试的轨迹不会被虚记为一次重试。

每个批次分别以新种子配对比较原 B3 和 B3_FIRST，报告两项成功指标的胜/负/平。**仅当两个批次都增加主指标，且任何一批的总体成功数都不下降，才采用本项改动。** 重试次数作为解释指标，不单独决定采用与否；不根据两批合并均值掩盖某一批的退步。

本轮是三个已有 TRAIN 状态上的目标对照，不是新任务泛化或递归细化验证。所有评测均无参数更新。

## 实现与验证

在共享 B3 训练器中增加 `reward_credit: first_registered`，候选和基线调用相同的回报归属规则；默认仍为 `full_return`。新入口只负责复用数据、训练配置、配对自然评测和结果汇总。

**24 项定向测试通过**，覆盖候选/基线一致重计分、失败样本保留、原始数据不变、负回报保留、TRAIN 隔离、仅信用归属变化、新评测种子分离、根重试统计，以及既有 B3/SFT 目标回归。

配置见 [refinement_first_credit.yaml](../configs/refinement_first_credit.yaml)，回报变化见 [credit_relabeling.json](../runs/refinement/B3_first_credit/credit_relabeling.json)，冻结请求见 [request.json](../runs/refinement/B3_first_credit/request.json)。

## 实际结果

### 已完成回报重计分

| 批次 | 状态 | 回报改变的 SPLIT 候选 / 8 | 独立基线均值：原 → 新 |
| --- | --- | ---: | ---: |
| 91000713 | calendar→matrix | 0/8 | 0.50 → 0.50 |
| 91000713 | chunks→chunks | 1/8 | 1.00 → 1.00 |
| 91000713 | intervals→matrix | 5/8 | 0.75 → 0.75 |
| 92000713 | calendar→matrix | 3/8 | 0.50 → 0.50 |
| 92000713 | chunks→chunks | 1/8 | 0.75 → 0.75 |
| 92000713 | intervals→matrix | 2/8 | 0.50 → 0.50 |

48 个 SPLIT 候选中，12 个回报改变，合计去掉 32 条首次被拒后成功的续跑奖励。其余首次被拒且原本失败的记录仍为零。六组独立基线均已重新计分，数值恰好不变：其中唯一首次被拒的基线原本就没有成功。这不意味着跳过了基线重算；测试使用了基线均值实际变化的对照数据。

### 新种子配对结果

各模型的分母均为全部 48 条自然执行。原 B3 检查点复用，但评测是新运行、新种子；不能将本表的通过数与上一轮另一组 48 条种子直接相减。

| 批次 | 模型 | 首次接受且最终成功 | 总体成功 | 首次计划接受 | 根规划重试次数 | 首次被拒后成功 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 91000713 | 原 B3 | 29/48 | 30/48 | 46/48 | 2 | 1 |
| 91000713 | B3_FIRST | 29/48 | 30/48 | 47/48 | 1 | 1 |
| 92000713 | 原 B3 | 31/48 | 33/48 | 46/48 | 2 | 2 |
| 92000713 | B3_FIRST | 31/48 | 32/48 | 47/48 | 1 | 1 |

| 批次 | 主指标胜/负/平 | 总体成功胜/负/平 | 达到采用条件 |
| --- | ---: | ---: | --- |
| 91000713 | 1/1/46 | 1/1/46 | 否：主指标持平 |
| 92000713 | 0/0/48 | 0/1/47 | 否：主指标持平、总体下降 |

两批都提高了一次首次接受并减少了一次根重试，说明目标修改确实改变了协议行为；但**首次接受更多，没有自动转化为首次合法规划后的成功更多**。第二批所有 48 个主指标标签均与原 B3 相同，不能把恢复成功变少误报为首次规划变好。

### 分根变化

箭头均为原 B3 → B3_FIRST，每根每模型 16 条。

| 批次 | 状态 | 首次接受且成功 | 总体成功 |
| --- | --- | ---: | ---: |
| 91000713 | calendar→matrix | 7 → 6 | 8 → 7 |
| 91000713 | chunks→chunks | 14 → 14 | 14 → 14 |
| 91000713 | intervals→matrix | 8 → 9 | 8 → 9 |
| 92000713 | calendar→matrix | 7 → 7 | 9 → 8 |
| 92000713 | chunks→chunks | 14 → 14 | 14 → 14 |
| 92000713 | intervals→matrix | 10 → 10 | 10 → 10 |

### 三条改变结果的轨迹

1. **第一批 intervals→matrix，种子 100602718：有效的首次规划改善。** 原 B3 引用了未声明的 `parse`、`prepare` 子节点，因 `UNKNOWN_OBLIGATION_OWNER` 被拒；重试 ATOMIC 后实现错误。B3_FIRST 首次生成合法 `prepare` 子任务，父节点保留聚合，最终通过。这是本轮唯一一次主指标胜出。见 [原 B3](../runs/refinement/B3_first_credit/evaluation/B3_91000713/natural/train_transfer_intervals_matrix_03/repeat_05/runtime/trace.jsonl)、[B3_FIRST](../runs/refinement/B3_first_credit/evaluation/B3_FIRST_91000713/natural/train_transfer_intervals_matrix_03/repeat_05/runtime/trace.jsonl)。
2. **第一批 calendar→matrix，种子 100600713：相同结构化计划，下游实现不同。** 两模型首次计划解析后的 JSON 完全相同，均下放 `compute`；原 B3 的组件正确，B3_FIRST 的组件把一维结果错误地按三项分组并触发异常。两次规划输出长度分别为 212、208 tokens，后续 observation 的剩余 token 预算也相差 4，采样过程并未固定到相同的执行器输入与随机数状态。因此，这条损失不能单独证明责任划分退化；它表明当前自然评测中的终局变化还包含下游执行差异。它与上条收益抵消，使第一批主指标持平。见 [原 B3](../runs/refinement/B3_first_credit/evaluation/B3_91000713/natural/train_transfer_calendar_matrix_03/repeat_00/runtime/trace.jsonl)、[B3_FIRST](../runs/refinement/B3_first_credit/evaluation/B3_FIRST_91000713/natural/train_transfer_calendar_matrix_03/repeat_00/runtime/trace.jsonl)。
3. **第二批 calendar→matrix，种子 100600715：减少重试但损失最终成功。** 原 B3 首次出现重复 JSON 字段，被拒后重试为合法 `prepare` 分解并成功；B3_FIRST 首次选择 ATOMIC 并被接受，但实现错误地将预处理结果按三项分组，最终失败。两者主指标都为零，总体成功由一变零。它直接说明：把恢复收益从初始动作的回报中去掉，不保证策略会选择更有效的首次方案。见 [原 B3](../runs/refinement/B3_first_credit/evaluation/B3_92000713/natural/train_transfer_calendar_matrix_03/repeat_02/runtime/trace.jsonl)、[B3_FIRST](../runs/refinement/B3_first_credit/evaluation/B3_FIRST_92000713/natural/train_transfer_calendar_matrix_03/repeat_02/runtime/trace.jsonl)。

### 决策与交付

按照预登记标准，本轮不采用 B3_FIRST，原 B3 继续使用 `full_return`。实验选项、两个新检查点和原始轨迹保留用于复现。局部首次分解改善确实存在，但固定预算下未形成两批主指标的净提升，不能用重试次数下降替代任务收益。本轮不追加调参、任务覆盖或递归训练。

实际完成 **0 条新增训练续跑、240 条复用训练续跑、40 次新增更新、192 条自然评测**。两个新模型均从空 Adam 完成 20 次更新；同批新旧模型除 `output_dir` 和 `reward_credit` 外训练配置一致，原候选数据摘要、累积/保持示范顺序、共同起点和参考权重一致。原始结果与旧检查点哈希保持不变；执行器冻结，模型重载差异为零；四组评测均完整且没有参数更新。24 项定向测试及最终完整性检查通过。

机器可读汇总见 [result.json](../runs/refinement/B3_first_credit/result.json)，检查点见 [evaluation_checkpoints.json](../runs/refinement/B3_first_credit/evaluation_checkpoints.json)。训练规则见 [refinement_pr1.py](../src/audittrainer/refinement_pr1.py)，实验入口见 [refinement_first_credit.py](../src/audittrainer/refinement_first_credit.py)。
