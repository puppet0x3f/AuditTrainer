# 冻结父组合摘要的新状态验证

状态：COMPLETE（2026-09-16）。16 个新父组合状态 × 4 个新执行种子 × 2 组，共 128 次续跑、0 次训练更新。

**冻结摘要的净正收益在新状态上复现：原输入 55/64，摘要 61/64，配对 8 胜、2 负、54 平，净增 6 次成功（9.38 个百分点）。** 收益分布于 3 个状态、2 个与设计轮不重叠的任务根，达到本轮事先固定的标准，支持下一阶段对各规划策略统一采用这一冻结输入条件。字典层两组均 24/24，未观察到键值混淆复现；列表层有两次新退步，全部保留。总模型 token 成本增加 20.03%，包含摘要组两次协议重试。上一轮结论保持不变，本轮没有启动 RL 回报重采或训练。

## 固定设计与新判据

本轮直接复用上一轮的 `composition_summary`、`summary_prompt` 和基础提示渲染器，不改字段、措辞或模型权重。记录摘要函数及基础提示代码哈希，评测前后检查；上一轮结果和报告哈希也固定，不回改其“未通过统一采用门槛”的结论。

主要问题：上一轮的 7 次新增成功集中在两个预选失败状态，固定摘要能否在其他父组合状态产生分布更广的净收益？

新标准在生成之前固定：**全部 64 对的净收益至少为 +1，且至少 2 个状态、来自至少 2 个不同任务根，分别具有正净收益**，才支持更广泛采用这一执行输入条件。允许个别配对退步，不沿用上一轮“原成功状态无净退步”的门槛。所有状态与字典/列表/字符串分层均报告胜负，键值混淆是否反复发生及 token 成本作为解释指标，不根据结果临时增设或删除门槛。这里的通过标准是有限实验的采用信号，不等于统计显著性结论。

## 状态选择

按 **字典 6、列表 6、字符串 4** 分层，类型指父子间 prepare 输出／aggregate 输入所传递的中间表示，不按 JSON 文本外壳或最终序列化返回值分类。

16 个状态来自 12 个任务根，全部与上一轮的 11 个任务根不重叠；同一任务根重复入选时使用不同责任分配（prepare 单子组件、prepare→aggregate 流水线或 aggregate 单子组件），不是只换历史 seed 或预算。每个子组件都已被消费、通过至少一个绑定的局部检查，恢复后证据仍有效；父节点均尚未开始实现。

字典层从既有 PR1C reference 轨迹中，固定 inventory/ledger、参数 3/4 的四个父保留聚合状态，加 inventory/ledger 参数 3 的两个流水线状态。列表层覆盖 calendar→chunks、calendar→intervals、intervals→calendar、words→chunks、words→paths、paths→chunks；字符串层为 chunks→paths 的 prepare/aggregate 分配及 chunks→words 的 prepare/流水线分配。

槽位按任务与接口结构确定。每槽位只依据“全部消费子组件局部 PASS”选首个可用历史状态，不使用历史父结果或本轮结果筛选；自然轨迹按 SFT、original 三个种子、optimized 三个种子的顺序，再按 repetition 顺序取首个合格状态。不存在已通过局部检查的图→词字典状态，因此没有从该任务强行纳入未经验证的子代码，也没有额外生成子组件。

其中字典层 6 状态来自 4 个既有 TRAIN 根，另外 10 状态来自 8 个既有 VALIDATION 组合根。它们均未参与摘要设计，但前一批 TRAIN 任务曾用于项目其他实验，VALIDATION 组合也曾评测过模型；本轮检验对摘要设计新状态的迁移，不声称全新未接触任务或新组件算法泛化。全部运行仅评测，不生成训练奖励或使用验证数据训练。

具体状态清单与冻结身份见 [request.json](../runs/refinement/B3_parent_validation/request.json)，配置见 [refinement_parent_validation.yaml](../configs/refinement_parent_validation.yaml)。

## 执行条件与检查

复用上一轮父快照恢复和续跑函数。计划、子实现及消费绑定冻结，原 observation、预算上限和已用预算完全恢复；不导入历史父代码或后续结果。唯一变动仍是附加同一摘要，额外输入 tokens 正常计入原预算。冻结 F01 执行器，不调用规划器、不修复、不更新模型。

独立执行种子为 `120600713 + state_index*1000 + repetition`，与上一轮不同；每组均在恢复后、父执行前重设同一个配对种子，交替组别执行顺序。固定 128 次，不根据中间结果补采或改提示。

7 项定向测试通过，覆盖全部新旧状态恢复、局部证据有效、新旧任务隔离、摘要公开信息边界、配对统计、新标准允许单点损失但要求多个任务获益、新种子无重叠。32 种初始输入均完成上下文预算检查。代码见 [refinement_parent_validation.py](../src/audittrainer/refinement_parent_validation.py)；上一轮评测器仅增加可显式允许 VALIDATION 快照的参数，默认仍仅允许 TRAIN，摘要实现未改。

## 实际结果

### 总体与接口分层

| 范围 | 配对数 | 原输入通过 | 摘要通过 | 摘要胜/负/平 |
| --- | ---: | ---: | ---: | ---: |
| 全部新状态 | 64 | 55/64（85.94%） | 61/64（95.31%） | 8/2/54 |
| 字典 | 24 | 24/24 | 24/24 | 0/0/24 |
| 列表 | 24 | 21/24 | 21/24 | 2/2/20 |
| 字符串 | 16 | 10/16 | 16/16 | 6/0/10 |

3 个状态净收益为正，1 个状态为负，其余 12 个持平；按任务根合并后，2 个获益、1 个退步、9 个持平。字符串层的 6 次收益来自 chunks→paths 的两种不同责任分配；列表层的 2 次收益来自 intervals→calendar，2 次退步来自 calendar→intervals。类型分层是描述性统计；字典层来自 TRAIN、另两层来自既有 VALIDATION，层间差异不能单独归因于数据类型。

收益已经超出设计轮的 chunks→chunks 和 intervals→matrix 状态。每状态四个执行种子仍是同一任务的重复，且多个状态共享任务根，不将 64 对描述为 64 个独立新任务，也不宣称所有接口类型都获得提升。

### 各状态结果

| 状态 | 类型 | 任务 | 固定子接口 | 原输入 | 摘要 | 胜/负/平 |
| --- | --- | --- | --- | ---: | ---: | ---: |
| S00 | 字典 | inventory，参数 3 | prepare | 4/4 | 4/4 | 0/0/4 |
| S01 | 字典 | inventory，参数 4 | prepare | 4/4 | 4/4 | 0/0/4 |
| S02 | 字典 | ledger，参数 3 | prepare | 4/4 | 4/4 | 0/0/4 |
| S03 | 字典 | ledger，参数 4 | prepare | 4/4 | 4/4 | 0/0/4 |
| S04 | 字典 | inventory，参数 3 | prepare, aggregate | 4/4 | 4/4 | 0/0/4 |
| S05 | 字典 | ledger，参数 3 | prepare, aggregate | 4/4 | 4/4 | 0/0/4 |
| S06 | 列表 | calendar→chunks | prepare | 4/4 | 4/4 | 0/0/4 |
| S07 | 列表 | calendar→intervals | prepare | 4/4 | 2/4 | 0/2/2 |
| S08 | 列表 | intervals→calendar | prepare | 2/4 | 4/4 | 2/0/2 |
| S09 | 列表 | words→chunks | prepare | 4/4 | 4/4 | 0/0/4 |
| S10 | 列表 | words→paths | aggregate | 3/4 | 3/4 | 0/0/4 |
| S11 | 列表 | paths→chunks | prepare, aggregate | 4/4 | 4/4 | 0/0/4 |
| S12 | 字符串 | chunks→paths | prepare | 2/4 | 4/4 | 2/0/2 |
| S13 | 字符串 | chunks→paths | aggregate | 0/4 | 4/4 | 4/0/0 |
| S14 | 字符串 | chunks→words | prepare | 4/4 | 4/4 | 0/0/4 |
| S15 | 字符串 | chunks→words | prepare, aggregate | 4/4 | 4/4 | 0/0/4 |

### 收益、退步与残留错误

逐项审查了全部 12 条终局失败代码、配对成功代码和两次协议拒绝；诊断标签可重叠，不替代正式责任证明。记录见 [diagnostics.json](../runs/refinement/B3_parent_validation/diagnostics.json)。

**S08，父聚合义务遗漏得到修复（2 次新增成功）。** 原输入直接序列化 prepare 返回的占用槽列表，遗漏相邻间隔、breaks 和 max_gap 的聚合。摘要组使用同一 prepare 子组件，随后在父节点计算间隔与阈值筛选，完成原本保留的职责。例见 [原输入](../runs/refinement/B3_parent_validation/evaluation/S08/repeat_01/control/runtime/artifacts/N0/v1.py)、[配对摘要](../runs/refinement/B3_parent_validation/evaluation/S08/repeat_01/summary/runtime/artifacts/N0/v1.py)。

**S13，错误的子接口调用得到修复（4 次新增成功）。** 原输入把导入的 aggregate 函数当成模块，调用不存在的 `_aggregate_child._prepare` 与 `.aggregate`，同时没有在父节点实现字符串展开。摘要组四次均自行完成展开，再直接调用冻结的 aggregate 函数。子组件代码与计划完全未变。例见 [原输入](../runs/refinement/B3_parent_validation/evaluation/S13/repeat_00/control/runtime/artifacts/N0/v1.py)、[配对摘要](../runs/refinement/B3_parent_validation/evaluation/S13/repeat_00/summary/runtime/artifacts/N0/v1.py)。

**S12，返回结构及格式得到修复，但两次均经过协议重试。** 原输入一条返回了错误结构并手工拼出无效 JSON，另一条输出含多余空格的非紧凑 JSON。对应摘要轨迹第一次 `write_code` 的动作 JSON 本身非法，被宿主拒绝；模型在原定一次重试预算内自行生成正确实现，随后通过。主指标按原预算内最终结果计数，因此保留这两次收益，同时计入额外调用和 tokens。例见 [第一条配对摘要轨迹](../runs/refinement/B3_parent_validation/evaluation/S12/repeat_02/summary/runtime/trace.jsonl)、[第二条配对摘要轨迹](../runs/refinement/B3_parent_validation/evaluation/S12/repeat_03/summary/runtime/trace.jsonl)。

为说明收益来源，额外统计“首次动作被接受且最终成功”：原输入 **55/64**，摘要 **59/64**，配对 **6 胜、2 负、56 平**。这只是解释指标，不替换预定主标准；其中仍有 S08、S13 两个不同任务根的获益，全部新增成功并非都依赖重试。

**S07，出现新的父聚合语义退步（2 次）。** 摘要实现把区间的端点和长度混用，用 `[value, 1]` 构造区间却把第二项与后续槽值比较，还只返回起点而非半开区间。例如应返回 `[[0,3],[1437,1440]]`，实际只返回 `[0]`，覆盖数也错误。对应原输入均通过。两次退步计入净收益，没有通过改提示或追加种子消除。例见 [退步实现](../runs/refinement/B3_parent_validation/evaluation/S07/repeat_00/summary/runtime/artifacts/N0/v1.py)。

**S10，输入预处理遗漏仍有残留。** 两组在种子 `120610715` 均把未经 strip/lower/空串过滤的原始 items 交给 aggregate，父 PREPARE 义务未落实；摘要中相应义务存在，但此次没有转化为行为修正。该状态两组均 3/4。见 [原输入](../runs/refinement/B3_parent_validation/evaluation/S10/repeat_02/control/runtime/artifacts/N0/v1.py)、[摘要](../runs/refinement/B3_parent_validation/evaluation/S10/repeat_02/summary/runtime/artifacts/N0/v1.py)。

**本轮没有观察到字典键值混淆反复发生。** 字典层两组各 24 条均通过；新退步属于区间表示和返回结构错误，不标为键值混淆。这支持继续保持当前摘要，不因上一轮一次键值错误临时加字段；同时不将有限用例的全通过解释为该类错误已被彻底消除。

### 实际成本与冻结检查

| 成本 | 原输入 | 摘要 |
| --- | ---: | ---: |
| 父组合续跑 | 64 | 64 |
| 真实父模型调用 | 64 | 66 |
| 协议拒绝／重试 | 0 | 2 |
| 模型输入 tokens | 151,668 | 184,240 |
| 模型输出 tokens | 13,969 | 14,582 |
| 模型 tokens 合计 | 165,637 | 198,822 |
| 每次成功的模型 tokens | 3,011.58 | 3,259.38 |

输入 tokens 增加 **21.48%**，总模型 tokens 增加 **20.03%**，每次成功的模型 tokens 增加 **8.23%**，均包含两次重试。128 次续跑对应 130 次父模型调用；没有把重试隐藏在预算外。两组宿主验证、最终私有评分和预算上限相同，成功率为本轮预定主指标，成本作为明确取舍报告。

三个分片分别完成 48、40、40 条，128 条全部 COMPLETE，无基础设施 ERROR；模型、计划、子代码及其绑定保持冻结。摘要函数和基础提示哈希不变，上一轮结果及报告哈希不变。7 项定向测试、全部实际快照一致性和配对预算检查完成。32 种初始提示均在原上下文预算内，见 [prompt_sizes.json](../runs/refinement/B3_parent_validation/prompt_sizes.json)。未新增训练更新，也未将 VALIDATION 执行转为训练奖励。

### 决策

本轮净收益 **+6**、正收益状态 **3 个**、正收益任务根 **2 个**，均达到生成前固定的新标准，`supports_broader_summary=true`。**支持将现有冻结摘要作为下一阶段各规划策略共同的父组合输入条件。** 这项决定依据本轮新状态结果，上一轮依其旧标准“不统一采用”的结论不变。

下一阶段应在同一摘要条件下比较 B0、成功计划 SFT 与 B3，并在 TRAIN 上重新采集候选及独立基线回报；执行条件改变后，旧回报不混入新一轮更新。本轮 VALIDATION 轨迹保留为评测数据。摘要尚不能保证父聚合完全正确，但已有跨任务的净收益证据，无需围绕单次键值错误继续修改提示才向前推进。

本次交付止于固定的 128 次验证、结果分析和采用依据；未自动修改既有训练配置、重采 RL 反馈或新增优化器更新。完整结果见 [result.json](../runs/refinement/B3_parent_validation/result.json)。复现入口：`python -m audittrainer.refinement_parent_validation prepare`、`evaluate --shard 0 --shards 3`（另运行 shard 1、2）、`report --shards 3`，使用 audittrainer Conda 环境。
