# PR1C：可执行训练条件与合法分解价值校准

状态：**COMPLETE / TRAIN 条件 READY（2026-09-13）**。

本轮建立固定 F01 执行器能够完成、且不同合法责任分配产生不同终局结果的 TRAIN 条件。
已完成 225 次校准运行：9 个根任务全部存在成功的分解，库存→账本任务的 pipeline 与 fused
在两个预先固定的种子块中均出现 0.5 的平均回报差。旧规划器仍缺乏足够的合法候选支持，
因此完成一轮完整计划 SFT 和新行为策略独立采样。新采样中两个均合法、均有局部 oracle、均只有一个子节点的分解
分别为 2/2 与 0/2 成功，训练池已通过现有 RL 加载器检查。总计 **252 次运行、27 次 SFT 更新、0 次新增 RL 更新**。

## 训练条件

先选择原有库存、账本、矩阵三个 TRAIN 任务族，各保留参数 3、4，共 6 根。
选择依据是 PR0 在这些族曾发现下游差异，属于利用既有 TRAIN 结果的课程设计。
原任务说明、输入输出、测试用例和分区均不修改。最初两个库存根的 32 次固定计划运行全部通过，
据此追加并在采样前固定三个组合任务：库存准备→账本汇总、账本准备→库存排序、图节点度数→库存排序。
全部根和失败结果保留，VALIDATION 与封存评测不参与。

三个组合任务复用已有准备/汇总组件、字典接口和独立 oracle，只改变配对；
公开精确的可选子契约，不提供实现或私有答案。这些任务共享已有算法组件，属于 TRAIN 组合课程，
不作为三个独立的未见算法家族或泛化评测。

课程使用已有深度限制 `max_depth=1`：根可分解，子节点只允许 ATOMIC。
保持 64 次策略调用、196608 个总 token、32 次在线验证、最多 7 个节点、3 个子节点和 8192 context 的预算。
实现与组合始终使用冻结的 A5 F01，沿用 PR0/PR1 的宿主验证与控制路由，修复关闭。
这使第一阶段的回报主要反映根责任分配的执行与组合；当前条件是浅层课程。

| 计划 | 子任务 | 父节点直接保留 |
| --- | --- | --- |
| ATOMIC | 无 | 解析、准备、汇总、输出 |
| pipeline | prepare → aggregate | 解析、输出 |
| fused | compute（准备与汇总） | 解析、输出 |
| parent_aggregate | prepare | 解析、汇总、输出 |

36 个固定根计划均通过真实运行器注册；36 个子契约绑定均有可用的可见局部验证器。
这些计划保持 incoming contract 与全部义务，SPLIT 复用已检查的精确子契约。
历史“初始注册成功”不一定意味着验证覆盖相同，例如历史样本存在 format 契约无 oracle、参数名不匹配等情况；
本轮对有收益差异的合法候选额外确认局部 oracle 覆盖。见 [历史契约诊断](../runs/refinement/PR1C/prior_contract_diagnosis.json)。

奖励沿用 `R=(1-v)y-0.2v`：y 为终局功能验证通过，v 为确认的责任违规，UNKNOWN 不作违规。
成本单独记录，节点数、深度、计划注册本身均无奖励。本轮功能通过与责任状态分别报告，
不把功能通过解释为完整语义责任证明。

## 固定计划的执行与价值差异

每根四个计划、每计划四次续跑，同根不同计划采用配对种子，共 **144 次**。
前两次与后两次是预先固定的两个种子块；有向收益差异在两个块中同向出现，才标记为重复观察。
固定计划明确标记 `AUTHORED_CALIBRATION_ONLY_NOT_POLICY_SAMPLE`，不赋予模型正文概率，不进入 RL 采样池。

| TRAIN 根任务 | ATOMIC | pipeline | fused | parent_aggregate |
| --- | ---: | ---: | ---: | ---: |
| inventory_03 | 4/4 | 4/4 | 4/4 | 4/4 |
| inventory_04 | 4/4 | 4/4 | 4/4 | 4/4 |
| ledger_03 | 4/4 | 4/4 | 4/4 | 4/4 |
| ledger_04 | 4/4 | 4/4 | 4/4 | 4/4 |
| matrix_03 | 4/4 | 4/4 | 4/4 | 3/4 |
| matrix_04 | 4/4 | 4/4 | 4/4 | 3/4 |
| graph → inventory | 4/4 | 4/4 | 4/4 | 4/4 |
| inventory → ledger | 1/4 | 2/4 | 0/4 | 2/4 |
| ledger → inventory | 4/4 | 4/4 | 4/4 | 3/4 |

固定计划合计 **130/144** 功能通过；9/9 根在两个种子块中都存在成功 SPLIT，8/9 根的 ATOMIC 至少 3/4 通过。
库存→账本的 pipeline 结果为 `[0,1,1,0]`，fused 为 `[0,0,0,0]`，两个块的平均差均为 **+0.5**。
这是合法分解的价值信号；四次续跑用于课程诊断，不作为稳定总体成功率估计。

具体失败机制可在 [fused 第一次局部验证](../runs/refinement/PR1C_transfer/reference/train_transfer_inventory_ledger_03/fused/repeat_0/runtime/evidence/E0001.json) 中复查：
生成的 compute 把库存三字段记录 `[sku, added, removed]` 当成账本二字段记录 `[account, delta]`，
在准备阶段触发 `ValueError: too many values to unpack`。拆开准备与汇总能够成功，提供了后续信用分配需要区分的结构选择。

完整结果：[原有任务批次](../runs/refinement/PR1C/result.json)、[组合批次](../runs/refinement/PR1C_transfer/result.json)。

## 旧策略的原始候选支持

另对每根采样 1 个规范 ATOMIC 和 3 个独立 SPLIT 正文，每候选续跑两次；共 36 个候选、72 次续跑。
所有非法、重复、截断候选均保留，不修正、不重采。另有每根一次独立自然基线，共 9 次。
因此上述校准实际为 **144 + 72 + 9 = 225 次**。

| 旧策略采样 | 原有 6 根 | 组合 3 根 | 合计 |
| --- | ---: | ---: | ---: |
| SPLIT 候选数 | 18 | 9 | 27 |
| 初始注册成功 | 11 | 2 | 13 |
| 初始注册且精确匹配局部 oracle | 5 | 1 | 6 |
| 原始候选续跑功能通过 | 39/48 | 14/24 | 53/72 |
| 独立自然基线功能通过 | 5/6 | 2/3 | 7/9 |
| 有合法且 oracle 覆盖的 SPLIT 间回报差异的根 | 0 | 0 | 0 |

库存→账本的原始候选 Q 为 `[0,1,0,0.5]`。其中两子任务候选已注册且有精确局部 oracle，2/2 成功；
合并候选的 1/2 成功发生在初始 `INVALID_DATAFLOW` 拒绝后的重试，不能据此认定合法 fused 的价值为 0.5。
由此区分两个结果：任务本身已有分解价值差异，但旧策略尚不能充分采到可比较的合法分解。

## 一次共同结构 SFT

[补训配置](../configs/refinement_curriculum_support.yaml) 使用旧共同 B0 初始化，仅更新规划 adapter，冻结 F01。
对全部 9 根导出四种合法计划，全部形式参与，不根据成功率或回报筛选目标，不训练代码实现。
数据含 **36 个唯一 prompt/target 对**；每个 SPLIT 形式显式复制两次，源文件共 63 行；
已有平衡选择器再重复 ATOMIC，最终 **108 次曝光（54 ATOMIC、54 SPLIT）**。

固定一轮、27 次 optimizer 更新，学习率 2e-5，梯度累积 4，完整条件正文 CE，额外前缀加权关闭。
实际 tokenizer 检查最长样本为 2842 token，未截断目标；源输入仅为采样前的公开根 observation。
同一补训 checkpoint 将作为后续方法的共同起点，此处的表示修订不计入 RL 算法收益。

数据：[plans.jsonl](../data/refinement/curriculum_support/plans.jsonl)、[来源与重复说明](../data/refinement/curriculum_support/plans.manifest.json)。
编码检查：[encoding_check.json](../runs/refinement/PR1C_support/encoding_check.json)。
实际完成 **27 次 SFT 更新、108 次曝光**，运行约 550 秒。F01 参数指纹和固定输入前向结果不变；
规划 adapter 保存重载后的 logits 与模式 log probability 最大差均为 0。
模型：[final_adapter](../runs/refinement/PR1C_support/adaptation/final_adapter)。
训练记录：[training_report.json](../runs/refinement/PR1C_support/adaptation/training_report.json)。

## 补训后的独立采样与训练入口

采样前固定为全部三个组合任务：一个非饱和结构对照及两个可执行控制任务。
原有六个根的校准证据和结构 SFT 保留。新策略已单独采集 12 个原始候选、24 次候选续跑和 3 次独立基线。
新旧行为策略的 token、概率与 Q 分开保存，固定参考计划不混入任何原始策略采样池。

[训练条件配置](../configs/refinement_curriculum_ready.yaml) 可由已有 PR1 采样与训练加载器读取，RL 更新目前关闭。
就绪检查要求原始采样里既有已注册且具备局部 oracle 的 SPLIT 价值差异，也有平均 SPLIT Q 高于 ATOMIC 的状态；
平均值保留所有原始候选，不使用 best-of-K。诊断分解差异时，对相同子契约与父职责划分的重复候选
先平均 Q，避免把同一计划的执行波动算成不同分解的差异；训练池仍保留全部重复样本。

验收结果为 **READY**，完整记录见 [result.json](../runs/refinement/PR1C_supported/result.json)。

| 组合任务 | ATOMIC Q | 三个原始 SPLIT Q（采样顺序） | 已注册且有精确局部 oracle 的 SPLIT | 不同合法职责划分的价值差异 |
| --- | ---: | --- | ---: | --- |
| graph → inventory | 1 | [1, 1, 1] | 1/3 | 无 |
| inventory → ledger | 0 | [0, 1, 0] | 2/3 | 有 |
| ledger → inventory | 1 | [1, 1, 1] | 3/3 | 无 |

新批次 9 个 SPLIT 中，8 个可解析、7 个初始注册、6 个同时具备精确局部 oracle；
24 次候选续跑有 18 次功能通过，独立自然基线为 3/3。
旧策略在相同三个组合任务上仅 1/9 个 SPLIT 同时满足注册与精确 oracle 覆盖，新批次为 6/9。
两个批次使用不同的预先固定种子，这里用于描述生成支持，不作为配对的 SFT 完成率收益估计。

库存→账本提供两类训练信号：

- **模式：** ATOMIC Q=0，全部三个原始 SPLIT 的平均 Q=1/3；非法候选也参与平均，不取最大值。
- **正文：** prepare child / parent aggregate 为 2/2，compute child 为 0/2；两者初始注册及局部 oracle 检查均通过。
  两者都只有一个子节点，观察到的价值差异对应职责放置与后续执行，而无需增加节点数奖励。

[该决策组](../runs/refinement/PR1C_supported/groups/train_transfer_inventory_ledger_03/group_result.json)
保留原始 token、旧概率、两个续跑种子和完整优势。
图→库存的一个合并候选未精确匹配局部契约，另一个候选初始被拒绝后恢复；
这些成功照常进入原始策略组，但不计入合法职责划分的价值证据。

目前的可用池是 **3 个 TRAIN 根状态、12 个原始候选、24 次续跑与 3 次独立基线**。
它支持下一轮小规模根规划 RL 对照；两次续跑的 Q 是有限样本估计，泛化与深层递归收益仍需相应实验。

## 实际成本

| 阶段 | 根轨迹数 | 推理输入 token | 推理输出 token |
| --- | ---: | ---: | ---: |
| 六个原有根校准 | 150 | 1,732,709 | 65,847 |
| 三个组合根校准 | 75 | 815,572 | 28,358 |
| 补训后新策略采样 | 27 | 321,400 | 11,916 |

总计 252 条根轨迹，推理输入 2,869,681 token、输出 106,121 token。
表中为 worker 实测模型用量，包含模式评分和候选生成；不把注入的参考计划虚记为模型生成。
SFT 的 108 次曝光与约 550 秒训练单独记录；在线验证和离线评分成本见各批次结果。

## 检查与复现

定向检查 **31 passed**，覆盖精确局部 oracle、真实原子叶调度、两块差异判定、重复划分的执行噪声、组合 oracle 手算及原 PR0/PR1 概率与训练逻辑。
日志：[pytest_focused.log](../runs/refinement/PR1C_support/pytest_focused.log)。校准五个 GPU worker 的规划器与执行器参数指纹前后一致。
补训参数更新、冻结执行器前向与保存重载检查均通过；新采样三个 worker 的权重不变、同一行为策略来源和现有训练组加载检查均已通过。

```bash
# 原有课程与组合课程。首次创建数据使用以下 prepare 命令；重采样时使用新的输出目录，已有结果保留。
python -m audittrainer.refinement_curriculum prepare --config configs/refinement_curriculum.yaml
python -m audittrainer.refinement_curriculum collect --config configs/refinement_curriculum.yaml
python -m audittrainer.refinement_curriculum report --config configs/refinement_curriculum.yaml
python -m audittrainer.refinement_curriculum prepare-transfer --config configs/refinement_curriculum_transfer.yaml
python -m audittrainer.refinement_curriculum collect --config configs/refinement_curriculum_transfer.yaml
python -m audittrainer.refinement_curriculum report --config configs/refinement_curriculum_transfer.yaml

# 共同结构补训与新行为策略采样。
python -m audittrainer.refinement_curriculum prepare-support --config configs/refinement_curriculum_support.yaml
python -m audittrainer.train_refinement --config configs/refinement_curriculum_support.yaml
python -m audittrainer.refinement_pr1 collect --config configs/refinement_curriculum_ready.yaml
python -m audittrainer.refinement_curriculum ready --config configs/refinement_curriculum_ready.yaml
```

实际运行环境为 audittrainer conda 环境，`CUDA_VISIBLE_DEVICES` 选择工作 GPU，`OMP_NUM_THREADS=4`。
旧六根采用 `--shards 2 --shard-index 0/1`，组合批次和补训后采样采用 `--shards 3 --shard-index 0/1/2`；按根分片不改变种子与动作分布。
本轮交付的是可执行、可区分且具有真实采样来源的 TRAIN 条件；未新增 RL 更新，也不声称已改善未见任务或深层递归能力。
