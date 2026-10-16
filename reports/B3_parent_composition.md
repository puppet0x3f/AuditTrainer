# 父节点组合执行输入对照

状态：COMPLETE（2026-09-16）。固定 12 个 TRAIN 组合状态，每状态 4 个新种子、2 种输入，共 96 次父组合续跑、0 次训练更新。

**父组合输入呈现确实影响了合理分解的执行收益：原输入通过 41/48，职责/接口摘要通过 47/48，配对 7 胜、1 负、40 平。** 摘要修复了预处理遗漏、子输入误用和父输出格式遗漏，但引入一次 inventory→ledger 父聚合错误。原成功状态从 40/40 降为 39/40，未满足预定的无净退步条件，因此本轮不统一切换默认执行输入、不追加 RL 训练。结果支持继续解决父节点履约问题，不能把这些冻结计划下的失败简单归为规划能力不足。

## 问题与固定条件

本轮检验：在计划和已通过局部检查的子组件完全固定时，将父节点保留职责、子组件输入输出和声明的数据流集中呈现，能否改善父组合最终功能通过率。

选择在新对照生成前完成。使用既有 TRAIN 轨迹中的 2 个父组合失败状态、10 个成功状态，覆盖 11 个任务。入选失败的历史父代码分别为：chunks→chunks 已完成聚合却直接返回 dict，遗漏父节点 FORMAT 义务；intervals→matrix 把原始区间传给期望预处理结果的 aggregate。跨模型重复出现的同一计划和子代码不作为新的失败类型重复计数。成功组包含与 chunks 失败相同计划/子代码、不同历史预算的状态，并覆盖父保留聚合、双子组件流水线和单 compute 子组件。原成功组用于检查退步，不是随机抽样的任务分布。

此前报告中 B3 第二批 chunks/repeat_10 的子契约缺少 ensures，局部状态为 UNKNOWN；它不满足本实验条件，未纳入。实际采用的是已有 bound local PASS 的同类失败。每个入选子组件均至少执行过一个局部检查用例，其版本、哈希、已消费绑定和局部证据在恢复后仍有效。局部 PASS 不等于对子组件的完整正确性证明。

恢复点为根父节点首次 IMPLEMENT/COMPOSE，所有子组件已完成并消费、父节点尚未保存代码。只复制当时的计划和子 artifact，不复制历史父代码或后续结果；恢复后的原始 observation 与保存的 observation 完全相同。沿用每个状态原预算上限和已用预算、协议、Qwen3-8B NF4 与冻结 F01 执行 adapter，关闭修复。共同 SFT 规划 adapter 仅随原双 adapter 加载器加载，不调用规划器。

## 唯一输入改动

- control：原输入和原提示渲染器。
- summary：在现有 root_spec 后附加一个紧凑摘要，逐项提取父契约中 obligation_map 归属 self 的义务原文、retained_responsibilities、各子接口 signature/requires/ensures/uses、消费绑定和已有 dataflow/obligation_map。

摘要不推断缺失的数据流、不补全或纠正计划、不生成组合代码、不加入用例答案或私有评分信息。原 observation 其余字段及所有模型采样参数不变；默认提示代码不修改，仅本实验显式选择摘要包装器。额外输入 tokens 照常计入同一预算。

每个状态和种子的 control/summary 从相同保存状态开始，恢复完成后、首次父执行调用前单独设置 RNG。种子为 `110600713 + state_index*1000 + repetition`，与此前评测不同；交替两组执行顺序，全部失败保留。共 48 对、96 次续跑，无新子组件实现、规划、模型更新或奖励修改。

## 预定指标与判断

主指标为全部 48 对的最终私有功能通过数及 summary 相对 control 的胜/负/平，另按原成功/失败状态分层。解释指标为父职责遗漏、输入误用、原成功状态退步、协议拒绝和输入成本。

只有摘要组总通过数严格提高，且原成功状态上的通过数不低于对照组，才支持将这一输入条件用于后续各规划器的统一比较。逐对损失仍逐项记录和检查；这一规则不是统计显著性或未见任务泛化结论。本轮最多完成 96 次父组合续跑，不依据中间结果追加采样或改提示。

如满足条件，之后须在统一执行输入下重新采集规划 RL 回报，旧执行回报不能混用；如不满足，保留当前默认执行输入，再考虑有针对性的父职责执行器训练。本轮不自动启动后续训练。

配置：[refinement_parent_composition.yaml](../configs/refinement_parent_composition.yaml)；代码：[refinement_parent_composition.py](../src/audittrainer/refinement_parent_composition.py)；固定请求：[request.json](../runs/refinement/B3_parent_composition/request.json)。

## 实现验证

4 项定向测试通过：恢复全部 12 个原 observation 和有效子证据、无未来父 source；摘要只重述公开职责/接口/数据流且不改变原状态；UNKNOWN 子组件排除；配对统计保留损失并拒绝缺失或重复槽位。

## 实际结果

### 总体与分层

| 评测范围 | 配对数 | 原输入 | 摘要输入 | 摘要胜/负/平 |
| --- | ---: | ---: | ---: | ---: |
| 全部状态 | 48 | 41/48（85.42%） | 47/48（97.92%） | 7/1/40 |
| 历史父组合失败状态 | 8 | 1/8 | 8/8 | 7/0/1 |
| 历史父组合成功状态 | 40 | 40/40 | 39/40 | 0/1/39 |

总体提升 **12.50 个百分点**。7 次新增成功均来自两个预选失败状态；另一 chunks 成功状态在两组中均为 4/4。每状态的四个种子是执行重复，不能当作四个新任务，且本轮样本按历史结果有意选取，不据此宣称总体任务分布上的提升。

| 状态 | TRAIN 组合任务 | 固定子接口 | 历史父结果 | 原输入 | 摘要输入 | 胜/负/平 |
| --- | --- | --- | --- | ---: | ---: | ---: |
| S00 | chunks→chunks | prepare | 失败 | 1/4 | 4/4 | 3/0/1 |
| S01 | intervals→matrix | aggregate | 失败 | 0/4 | 4/4 | 4/0/0 |
| S02 | chunks→chunks | prepare | 成功 | 4/4 | 4/4 | 0/0/4 |
| S03 | calendar→matrix | prepare | 成功 | 4/4 | 4/4 | 0/0/4 |
| S04 | graph→inventory | prepare, aggregate | 成功 | 4/4 | 4/4 | 0/0/4 |
| S05 | inventory→ledger | prepare | 成功 | 4/4 | 3/4 | 0/1/3 |
| S06 | ledger→inventory | prepare, aggregate | 成功 | 4/4 | 4/4 | 0/0/4 |
| S07 | inventory→graph | prepare, aggregate | 成功 | 4/4 | 4/4 | 0/0/4 |
| S08 | ledger→graph | prepare | 成功 | 4/4 | 4/4 | 0/0/4 |
| S09 | intervals→intervals | prepare, aggregate | 成功 | 4/4 | 4/4 | 0/0/4 |
| S10 | matrix→matrix | prepare | 成功 | 4/4 | 4/4 | 0/0/4 |
| S11 | paths→paths | compute | 成功 | 4/4 | 4/4 | 0/0/4 |

### 父职责与接口诊断

逐项检查了全部 8 条失败父代码及其配对结果，标签可重叠，仅作具体代码错误解释，不替代正式责任判定。记录见 [diagnostics.json](../runs/refinement/B3_parent_composition/diagnostics.json)。

**S00：父保留义务重新落实。** 原输入的两条失败在完成聚合后直接返回 dict，遗漏 FORMAT；另一条调用并未定义的 `_aggregate_child`，父聚合实现缺失，同时未完成序列化。摘要组四次均自行生成完整聚合及 JSON 格式化代码，最终通过。计划、prepare 子代码和绑定没有改变。例见 [原输入的格式遗漏](../runs/refinement/B3_parent_composition/evaluation/S00/repeat_00/control/runtime/artifacts/N0/v1.py)、[配对摘要实现](../runs/refinement/B3_parent_composition/evaluation/S00/repeat_00/summary/runtime/artifacts/N0/v1.py)、[原输入的未定义聚合调用](../runs/refinement/B3_parent_composition/evaluation/S00/repeat_02/control/runtime/artifacts/N0/v1.py)。

**S01：子输入误用四次均被修正。** 原输入四次都把原始 intervals 直接传入期望 prepared 整数序列的 aggregate。摘要组四次均先在父节点计算去重、排序后的占用整数槽，再调用完全相同的 aggregate 子组件，最终通过。这直接证明，至少这些失败回报可以通过父组合执行的输入呈现修复，无需修改计划或训练权重。例见 [原输入](../runs/refinement/B3_parent_composition/evaluation/S01/repeat_00/control/runtime/artifacts/N0/v1.py)、[配对摘要实现](../runs/refinement/B3_parent_composition/evaluation/S01/repeat_00/summary/runtime/artifacts/N0/v1.py)。

**S05：摘要不能保证父聚合正确。** 种子 `110605715` 的摘要实现对 balances 字典迭代得到的账户名执行 `b <= -3`，把键当作余额；同时将 accounts 写成映射而非要求的账户名列表。可见检查为 WRONG_OUTPUT，私有检查含 CANDIDATE_EXCEPTION。同种子的原输入按字典值筛选账户并输出正确结构，通过检查。四次中另外三次摘要通过；这是一次实际配对退步，不能因总体净增加六次成功而省略。见 [退步代码](../runs/refinement/B3_parent_composition/evaluation/S05/repeat_02/summary/runtime/artifacts/N0/v1.py)、[配对通过代码](../runs/refinement/B3_parent_composition/evaluation/S05/repeat_02/control/runtime/artifacts/N0/v1.py)。

### 成本与完成检查

两组各 48 次真实父模型调用，无协议拒绝或重试；每条续跑再使用相同的宿主 verify/finish 控制，均无规划器调用、新子实现或修复。全部 96 条为 COMPLETE，功能失败保留，无评分或运行基础设施 ERROR。

原输入/摘要输入分别使用 **123,748 / 144,104 个输入 tokens**，增加 16.45%；输出分别为 10,195 / 10,470 tokens。总模型 tokens 为 133,943 / 154,574，增加 15.40%。两组沿用相同状态预算；全部初始提示无需压缩或截断，见 [prompt_sizes.json](../runs/refinement/B3_parent_composition/prompt_sizes.json)。本次输入改动有真实 token 成本，不把它视作免费提升。

三个分片均完成 32 条，规划与执行 adapter 前后指纹相同，跨分片指纹相同；所有计划、子 source、绑定与子状态保持不变。4 项定向测试、96 条实际恢复一致性检查、配对槽位及冻结预算检查完成。未新增优化器更新或训练轨迹回报。

### 决策

**有明确修复价值，但未通过本轮统一采用门槛。** 总通过数提高这一条件成立，原成功状态无净退步这一条件不成立；按生成前固定的规则，`supports_summary=false`。默认执行输入保持现状，摘要保留为显式实验条件，本轮不追加提示变体或种子来改变结论。

下一阶段重点可以收敛到父节点的聚合履约，特别是映射的键/值与返回结构，而不继续扩大原配置的规划 RL。本轮已有证据表明，把执行错误全部反馈为分解失败会掩盖可实现的计划收益；后续如统一采用新的执行条件，必须重新采集候选续跑及基线回报，再评价 B3 与成功计划 SFT，不能直接沿用旧回报。

完整机器结果：[result.json](../runs/refinement/B3_parent_composition/result.json)。复现入口为 `python -m audittrainer.refinement_parent_composition prepare`、`evaluate --shard 0 --shards 3`（另运行 shard 1、2）、`report --shards 3`；使用 audittrainer Conda 环境。
