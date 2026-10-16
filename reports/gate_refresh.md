# 新执行器下的独立根门控校准

2026-09-18。本轮固定执行器SFT候选、原MULTI_BODY正文及内部策略，只更新独立根门控。旧F01反馈不进入本轮估值或校准。

## 执行前固定的设计

- 8个TRAIN状态：既有calendar→matrix、intervals→matrix、chunks→chunks、words→words、graph→ledger、inventory→graph，以及新增模5系数提取、路径计数状态。
- 每个状态8次ATOMIC及8份新采样SPLIT、每份1次完整续跑，共128条；包含拒绝、内部再次分解、重试及最终失败。
- 两种根动作使用相同初始执行种子，计划采样种子独立。后续不同轨迹不宣称随机流逐token相同。
- 32步固定预算校准，学习率1e-5，8个TRAIN状态每步各曝光一次，总256次状态曝光。使用最后一个检查点，无自然执行结果选模型。
- 80条配对自然执行：旧/新门控分别在8个TRAIN根×4种子及2个未参与门控估值/训练的模7参数变体×4种子执行。两种门控均使用同一新执行器和原MULTI_BODY。

## 校准目标

原门控在新增状态的SPLIT概率低至1e-8量级，直接对`p*ΔQ`求梯度容易近乎停滞。本轮将新执行价值对应的逐状态KL正则最优分布作为软监督，而不再次进行单例SFT诊断。

对成功数相差至少2/8的状态，令参考概率`p_ref`为冻结原MULTI_BODY的门控概率，β=0.1：

`p_target = sigmoid(logit(p_ref) + (Q_SPLIT - Q_ATOMIC) / β)`。

该分布是`p*Q_SPLIT + (1-p)*Q_ATOMIC - β*KL(Bern(p)||Bern(p_ref))`的逐状态最优解。训练最小化这个软目标对门控的交叉熵，是对逐状态最优分布的拟合，不是直接策略梯度；共享参数下未必精确达到每个目标。目标概率也不是把成功率直接当作SPLIT概率。

成功数相差至多1/8的状态保留旧独立门控概率作为监督目标，不赋予收益方向。此阈值、β、步数和任务清单在采集前固定。7项目标公式/不确定带测试及2项分离路由测试通过。

## 结果

### 当前执行价值（128条完成）

| TRAIN状态 | ATOMIC成功 | SPLIT尝试成功 | 监督方向 | 目标P(SPLIT) |
|---|---:|---:|---|---:|
| train_course_calendar_matrix_03 | 0/8 | 3/8 | SPLIT | 0.914901 |
| train_course_intervals_matrix_03 | 8/8 | 1/8 | ATOMIC | 4.00637e-05 |
| train_course_chunks_chunks_04 | 8/8 | 7/8 | UNCERTAIN | 7.73442e-08 |
| train_course_words_words_04 | 7/8 | 8/8 | UNCERTAIN | 6.8256e-08 |
| train_course_graph_ledger_03 | 8/8 | 5/8 | ATOMIC | 0.00669285 |
| train_course_inventory_graph_03 | 0/8 | 0/8 | UNCERTAIN | 1.76034e-06 |
| train_execution_polynomial_matrix_05 | 4/8 | 7/8 | SPLIT | 0.914901 |
| train_execution_paths_matrix_05 | 0/8 | 8/8 | SPLIT | 0.99982 |

全部128条正常完成。SPLIT包含所有初次拒绝、重试及失败：chunks→chunks的repeat_00首次JSON被拒绝，旧门控重试为ATOMIC并成功；该次计入SPLIT尝试收益，不能解释为成功注册后分解的成功率。其余127条首次根计划注册成功。这个状态处于不确定带，未生成分解收益方向标签。估值是旧门控后续策略下的单步改进信号，最终效果由新门控自然执行单独检验。

三个正向状态是calendar、多项式、路径；两个负向状态是intervals、graph→ledger。inventory两种模式均失败，不强造标签。新执行器下calendar直接实现0/8，示例失败为prepare已返回分钟列表，aggregate却再次按矩阵求列和；因此不声称新执行器普遍无回归，也不复用旧F01估值。

### 自然执行与参数保持

32步、256次状态曝光完成。仅根门控副本更新；正文与执行器参数指纹不变，4组配对正文生成逐token一致，磁盘重载logit最大差0。

| TRAIN状态 | 原P(SPLIT) | 校准后P(SPLIT) |
|---|---:|---:|
| train_course_calendar_matrix_03 | 1 | 0.851953 |
| train_course_intervals_matrix_03 | 2.06115e-09 | 8.4811e-05 |
| train_course_chunks_chunks_04 | 7.73442e-08 | 0.00026119 |
| train_course_words_words_04 | 6.8256e-08 | 0.000379985 |
| train_course_graph_ledger_03 | 1.63738e-07 | 0.00591107 |
| train_course_inventory_graph_03 | 1.76034e-06 | 0.000379985 |
| train_execution_polynomial_matrix_05 | 1.72577e-08 | 0.924142 |
| train_execution_paths_matrix_05 | 4.69114e-08 | 0.998073 |

负收益状态原本已接近零，软目标校准后概率略升但仍强烈偏向ATOMIC，不将“相对旧概率进一步下降”作为本轮结论。共享参数拟合未精确达到各目标，三个不确定状态也存在微小变化；用完整自然执行检查实际影响。

80条配对自然执行全部完成，两组使用相同新执行器、冻结MULTI_BODY正文及内部策略，仅根门控不同。

| 评估范围 | 原门控根成功 | 新门控根成功 | 原/新SPLIT次数 |
|---|---:|---:|---:|
| 8个TRAIN状态 | 18/32 | 22/32 | 4/32 → 9/32 |
| 2个未训练同家族参数变体 | 3/8 | 7/8 | 0/8 → 8/8 |

40组配对为9胜、1负、30平。根模式次数直接读取策略选定模式，初次正文拒绝仍算分解尝试。

| 状态 | 根成功：原→新 | SPLIT次数：原→新 | 新P(SPLIT) | 配对胜/负 |
|---|---:|---:|---:|---:|
| train_course_calendar_matrix_03 | 1/4 → 1/4 | 4/4 → 2/4 | 85.20% | 0/0 |
| train_course_intervals_matrix_03 | 4/4 → 4/4 | 0/4 → 0/4 | 0.01% | 0/0 |
| train_course_chunks_chunks_04 | 4/4 → 4/4 | 0/4 → 0/4 | 0.03% | 0/0 |
| train_course_words_words_04 | 2/4 → 2/4 | 0/4 → 0/4 | 0.04% | 0/0 |
| train_course_graph_ledger_03 | 4/4 → 4/4 | 0/4 → 0/4 | 0.59% | 0/0 |
| train_course_inventory_graph_03 | 0/4 → 0/4 | 0/4 → 0/4 | 0.04% | 0/0 |
| train_execution_polynomial_matrix_05 | 3/4 → 3/4 | 0/4 → 3/4 | 92.41% | 1/1 |
| train_execution_paths_matrix_05 | 0/4 → 4/4 | 0/4 → 4/4 | 99.81% | 4/0 |
| train_executor_check_polynomial_matrix_07 | 3/4 → 4/4 | 0/4 → 4/4 | 91.49% | 1/0 |
| train_executor_check_paths_matrix_07 | 0/4 → 3/4 | 0/4 → 4/4 | 99.59% | 3/0 |

### 结论与剩余执行缺口

**本轮完成了“新执行能力 → 当前策略重新估值 → 根门控调用 → 自然收益”的闭环。** 原门控下几乎从不分解的路径状态，在校准后4/4调用分解、4/4成功；未用于门控训练的路径模7变体4/4调用、3/4成功。两个未训练参数变体均出现实际分解调用并提高成功数，收益并非全部来自转向ATOMIC。六个既有TRAIN状态的自然成功数均保持；多项式训练状态3/4→3/4，有一胜一负，不能声称所有正收益任务都在本批自然执行中提高。

两条新增任务SPLIT自然失败继续计分：

- 多项式训练repeat_02：内部prepare再次分解后，根consume错误引用孙节点，两次`CHILD_NOT_PENDING`后终止。这是唯一配对退步。
- 路径模7 repeat_01：内部prepare节点写成compute并提前执行聚合，发生两次`INTERFACE_MISMATCH`；根随后错误处理产物版本并尝试消费孙节点，最终协议重试耗尽。旧门控对应种子也失败。

证据分别在 [多项式失败](../runs/refinement/gate_refresh/natural/AFTER/train_execution_polynomial_matrix_05/repeat_02/case_result.json) 和 [路径失败](../runs/refinement/gate_refresh/natural/AFTER/train_executor_check_paths_matrix_07/repeat_01/case_result.json)。

**保留新根门控＋冻结MULTI_BODY＋新执行器为下一阶段候选。下一步进入少量固定内部状态的ATOMIC/SPLIT对照，以最终根成功判断是否需要继续细分，并集中检查当前函数职责与直属子节点消费。** 内部策略改变后重新采集受影响的根价值，不复用本轮QS；本轮不追加门控预算或重开正文单例训练。

范围：单个训练种子、小样本开发实验；TRANSFER为同组合家族的新参数/输入变体，两个状态曾用于上一轮执行器的开发评估，未用于本轮门控估值、训练或检查点选择。因此支持同家族参数迁移，尚不支持全新组合家族泛化或独立封存测试结论。

### 复现与核验

- 128条估值＋80条自然执行，共208条模型执行，0基础设施错误；业务失败照常计入。
- 原根门控、原MULTI_BODY、新执行器磁盘文件均未改变；训练时正文与执行器指纹不变；4次配对正文生成逐token一致；重载差0；9项定向测试通过。
- 新根门控：`runs/refinement/gate_refresh/training/final_adapter`。必须经`RootGatePolicy`使用，正文/内部策略仍指向`runs/refinement/body_transfer/MULTI_BODY/final_adapter`；执行器槽位指向`runs/refinement/executor_sft/training/final_adapter`。
- [固定请求](../runs/refinement/gate_refresh/request.json)、[完整结果](../runs/refinement/gate_refresh/result.json)、[核验结果](../runs/refinement/gate_refresh/validation.json)、[运行实现](../src/audittrainer/refinement_gate_refresh.py)。
- CLI顺序：`python -m audittrainer.refinement_gate_refresh prepare`；`values --shard 0/1/2`分别执行；`summarize_values`；`train`；`natural --method BEFORE/AFTER --shard 0/1/2`分别执行；`report`。使用audittrainer环境，GPU按资源配置分配。
