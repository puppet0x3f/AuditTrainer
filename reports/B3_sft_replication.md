# 新执行种子上的 SFT 基线补评测

状态：COMPLETE（2026-09-16）。144 条补充评测全部完成，无新增训练或奖励修改。

**结论：原 B3 的额外终局收益没有在两批中一致复现，暂停增加训练量。** 新种子上共同 SFT 为 30/48，两批成功计划 SFT 均为 31/48，原 B3 为 30/48、33/48。B3 相对成功 SFT 第一批 1 胜 2 负，第二批 3 胜 1 负，方向不一致，未达到预定继续扩展条件。第二批仍有正向收益，但现有证据不足以支持扩大 RL；下方轨迹已定位到预处理/聚合接口误用、父节点职责遗漏及下游执行差异。

## 要回答的问题

首次规划信用对照中，原 B3 在新执行种子上通过 30/48、33/48，缺少共同 SFT 和两批成功计划 SFT 的同种子对照。本轮补齐这些基线，判断原 B3 的额外收益能否在第二组执行种子上复现。

本补充评测在已知原 B3 结果后设计；三个 SFT 模型尚未进行本轮评测时，固定以下预算、比较和判断标准。任务与种子沿用已完成的信用对照，不重新挑选。

## 固定条件与预算

| 模型 | 来源 | 本轮运行数 |
| --- | --- | ---: |
| B0：共同结构 SFT | 既有检查点 | 新增 48 |
| SUCCESS_SFT，91000713 | 既有第一批检查点 | 新增 48 |
| SUCCESS_SFT，92000713 | 既有第二批检查点 | 新增 48 |
| B3，91000713 | 首次规划信用对照的原 B3 | 复用 48 |
| B3，92000713 | 首次规划信用对照的原 B3 | 复用 48 |

合计 **144 条新增自然执行、96 条复用自然执行、0 次参数更新、0 条新增训练续跑**。B0 只运行一次，两批比较共用它；两批之间相同的任务/种子不视为不同新任务。

五模型均使用 calendar→matrix、chunks→chunks、intervals→matrix 三个已有 TRAIN 状态，每根 16 条。执行种子为 `100600713 + sorted_root_index*1000 + repetition`，与 [首次规划信用对照](B3_first_credit.md) 完全一致。运行预算、协议、最大深度 1、关闭代码修复、允许拒绝后重试及冻结执行器 F01 均沿用原设置。

直接调用同一自然评测函数，使用相同执行配置与原终局奖励，不重新归属训练回报。原 B3 的评测目录只用于读取和链接，原检查点与结果不修改。配置见 [refinement_sft_replication.yaml](../configs/refinement_sft_replication.yaml)，固定请求见 [request.json](../runs/refinement/B3_sft_replication/request.json)。

## 指标与判断

主比较为每批 **B3 相对同批成功计划 SFT 的最终功能通过数与配对胜/负/平**。共同 SFT 用于判断两类训练相对起点的变化。

“首次计划被接受且最终成功”、首次接受数、根规划重试次数及首次被拒后的成功数仅用于解释收益；重试减少不替代主指标。所有指标均使用完整 48 条为分母，功能失败仍保留。

**只有两批 B3 的最终成功数都严格高于各自成功计划 SFT，才支持继续投入原 B3，之后考虑小规模任务覆盖验证。** 任一批持平或方向不一致，则暂停增加训练量，先定位执行器与计划接口问题，不据此扩大 RL。本轮只完成补充评测及决策，不自动启动下一项实验。

这检验同一组 TRAIN 状态上的执行种子复现，不是未训练任务泛化、递归能力或统计显著性结论。

## 实现与检查

复用首次规划信用对照的自然执行和指标统计，仅把已选检查点的评测提取为共享函数。11 项定向测试通过，覆盖执行条件与槽位一致、既有检查点身份、配对胜负方向、首次接受与终局成功区分，以及两批均领先的判断规则。

## 实际结果

### 同种子主结果

每模型均为同样的 48 个任务—种子组合；原 B3 两行复用已完成结果，其余三行是本轮新增执行。

| 模型 | 最终成功 | 首次接受且最终成功 | 首次计划接受 | 根规划重试次数 | 首次被拒后成功 |
| --- | ---: | ---: | ---: | ---: | ---: |
| B0：共同 SFT | 30/48 | 28/48 | 46/48 | 2 | 2 |
| SUCCESS_SFT，91000713 | 31/48 | 28/48 | 45/48 | 3 | 3 |
| B3，91000713 | 30/48 | 29/48 | 46/48 | 2 | 1 |
| SUCCESS_SFT，92000713 | 31/48 | 29/48 | 46/48 | 2 | 2 |
| B3，92000713 | 33/48 | 31/48 | 46/48 | 2 | 2 |

| B3 相对同批成功 SFT | 最终成功差 | 最终成功胜/负/平 | 首次接受且成功胜/负/平 |
| --- | ---: | ---: | ---: |
| 91000713 | −1/48 | 1/2/45 | 1/0/47 |
| 92000713 | +2/48 | 3/1/44 | 3/1/44 |

相对 B0，两批成功 SFT 均为 1 胜、0 负、47 平；两批 B3 分别为 1 胜 1 负 46 平、3 胜 0 负 45 平。B0 只执行一次，没有把共用基线记作两份独立运行。

上一组执行种子上，B3 相对成功 SFT 的差为 **+1/48、+2/48**；本轮变为 **−1/48、+2/48**。第二批的净收益保持为正，第一批反向。因此，“两批 B3 均稳定超过成功 SFT”的结论没有通过本次复现检查，不能用两批合计的小幅优势替代预定的逐批判断。

首次接受指标在两批都对 B3 有利，但它是解释指标。第一批 B3 的首次接受且成功多一次、重试少一次，仍因两条 SFT 恢复成功而在终局少通过一次；本轮不能据此更换主判据。

### 分任务通过数

每格分母为 16。

| 模型 | calendar→matrix | chunks→chunks | intervals→matrix |
| --- | ---: | ---: | ---: |
| B0 | 7 | 14 | 9 |
| SUCCESS_SFT，91000713 | 8 | 14 | 9 |
| B3，91000713 | 8 | 14 | 8 |
| SUCCESS_SFT，92000713 | 7 | 15 | 9 |
| B3，92000713 | 9 | 14 | 10 |

### 解释收益与损失的轨迹

**首次规划与终局结果仍受下游实现影响。** 第一批 calendar→matrix、种子 100600713 是 B3 相对成功 SFT 的唯一终局胜出。两者首次计划解析后的 JSON 完全相同，均选择 `compute` 子任务；规划输出分别为 212、208 tokens。固定执行器在 B3 轨迹中生成正确聚合，在 SFT 轨迹中把一维预处理结果错误地按三项分组，最终抛出异常。该样本不能单独证明 B3 学到了更好的责任划分。不同生成长度会改变后续预算上下文及采样过程；相同初始执行种子并不保证相同的下游实现。见 [成功 SFT](../runs/refinement/B3_sft_replication/evaluation/SUCCESS_SFT_91000713/natural/train_transfer_calendar_matrix_03/repeat_00/runtime/trace.jsonl)、[原 B3](../runs/refinement/B3_first_credit/evaluation/B3_91000713/natural/train_transfer_calendar_matrix_03/repeat_00/runtime/trace.jsonl)。

**恢复成功并非 B3 独有。** 第一批的两条终局损失分别在 calendar→matrix/100600715 和 intervals→matrix/100602718。成功 SFT 都在首次计划被拒后重试成功；原 B3 前者首次 ATOMIC 实现错误，后者同样重试 ATOMIC 但聚合实现错误。这解释了第一批为何首次规划指标更好、终局却更差。见 [calendar 的 SFT 恢复](../runs/refinement/B3_sft_replication/evaluation/SUCCESS_SFT_91000713/natural/train_transfer_calendar_matrix_03/repeat_02/runtime/trace.jsonl)、[intervals 的 SFT 恢复](../runs/refinement/B3_sft_replication/evaluation/SUCCESS_SFT_91000713/natural/train_transfer_intervals_matrix_03/repeat_05/runtime/trace.jsonl)。

**有效分配与接口失败同时存在。** 第二批 intervals→matrix、种子 100602724 中，SFT 的父节点把原始区间直接交给期望预处理结果的 `aggregate`，导致异常；B3 下放 `prepare`、父节点保留聚合并正确完成，是有明确责任分配差异的正向例子。相反，在 chunks→chunks、种子 100601723 中，SFT 直接实现成功，B3 虽然生成合法 `prepare` 子任务、且计划把 `AGGREGATE` 归给 `self`，父节点却直接返回预处理字符串，漏做切块和长度统计。这是仍然存在的计划—执行接口问题。见 [intervals 的 SFT](../runs/refinement/B3_sft_replication/evaluation/SUCCESS_SFT_92000713/natural/train_transfer_intervals_matrix_03/repeat_11/runtime/trace.jsonl)、[intervals 的 B3](../runs/refinement/B3_first_credit/evaluation/B3_92000713/natural/train_transfer_intervals_matrix_03/repeat_11/runtime/trace.jsonl)、[chunks 的 B3](../runs/refinement/B3_first_credit/evaluation/B3_92000713/natural/train_transfer_chunks_chunks_03/repeat_10/runtime/trace.jsonl)。

### 决策与完成检查

本轮不满足“两批 B3 在新种子上仍领先”的条件，**暂停增加训练量，不启动任务池扩展或递归训练**。保持现有检查点和原 B3 的回报配置；本结果不支持换回此前未通过对照的归零目标，也不意味着 RL 在所有条件下无效。当前要解决的具体问题是：父节点如何可靠履行保留职责、预处理结果如何与子组件接口正确衔接，以及如何区分计划差异与下游执行差异。

本轮交付了 **144 条新增执行、96 条复用执行、0 次更新、0 条训练续跑**。五模型检查点哈希不变，三个新评测的策略和执行器前后指纹一致，复用的原 B3 结果哈希不变；所有运行 COMPLETE，任务、种子与执行配置匹配。11 项定向测试及最终配对、冻结、预算和无训练检查通过。没有追加采样、调参或修改奖励。

完整结果见 [result.json](../runs/refinement/B3_sft_replication/result.json)，既有检查点清单见 [evaluation_checkpoints.json](../runs/refinement/B3_sft_replication/evaluation_checkpoints.json)，补评测入口见 [refinement_sft_replication.py](../src/audittrainer/refinement_sft_replication.py)。
