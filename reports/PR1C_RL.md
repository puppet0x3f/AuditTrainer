# PR1C RL：从可执行价值信号到规划策略更新

状态：**COMPLETE（2026-09-13）**。

本轮执行 PR1C 训练条件上的 B0/B3/B4 有限对照，检验完整后续执行回报能否改变规划决策。
B3、B4 已各完成 2 次实际 RL optimizer 更新，执行器与参考策略保持冻结，保存重载差为 0。
新种子复测已完成每组 30 条自然轨迹和 12 个条件 SPLIT 样本，所有结果保留。
B0 为 15/30，B3、B4 均为 14/30；两种更新提高了优选正文的相对评分，本轮尚未转化为自然运行收益。

## 实验条件与预算

B0 是 [PR1C 共同结构 SFT checkpoint](../runs/refinement/PR1C_support/adaptation/final_adapter)。
B3、B4 均从 B0 初始化，使用 PR1C 同一组原始候选、旧概率和独立自然基线。
[原始训练池](../runs/refinement/PR1C_supported/result.json) 含 3 个 TRAIN 根状态、12 个候选、24 次候选续跑、3 次独立基线；
本轮通过原文件链接复用，没有重新筛选、修改候选或混入参考计划。

| 条件 | 更新 |
| --- | --- |
| B0 | 保留共同 SFT checkpoint，0 次更新 |
| B3 | 统一计划优势，2 轮遍历、2 次 optimizer 更新 |
| B4 | 模式/正文双层优势，2 轮遍历、2 次 optimizer 更新 |

每次更新累积全部 3 个根状态；学习率 1e-6，clip 0.2，KL 权重 0.02，SFT 保持项权重 0.02，
共同 objective scale 为 1/256，梯度范数上限 1。没有额外 SFT 训练阶段；RL 目标仍包含上述共同保持项。
SFT 保持样本、数据顺序、更新次数和其余设置匹配，方法间只改变优势构造。

执行器固定为 A5 F01，修复关闭，沿用 `max_depth=1` 的根规划课程与 PR1C 全部运行预算。
回报仍为 `R=(1-v)y-0.2v`，成本另记，没有节点数或注册成功奖励。

库存→账本根的训练 Q 为 `[0,0,1,0]`，独立自然基线回报为 1。
B3 的三个正文优势为 `[-1,0,-1]`，B4 为 `[-0.5,1,-0.5]`；
B4 给已成功的 prepare child / parent aggregate 正文直接提供正优势。
有收益正文与合并正文都只有一个子节点，并均通过首次注册和精确局部 oracle 检查。

训练数据与评测预算在更新前固定于 [refinement_curriculum_rl.yaml](../configs/refinement_curriculum_rl.yaml)。
旧配置与旧采样结果保持原有身份，本轮结果写入独立的 `PR1C_RL` 目录。

## 新种子复测

每个模型使用相同的根任务与种子：库存→账本 20 条，图→库存 5 条，账本→库存 5 条，共 **30 条自然轨迹**。
三个模型共 90 条。根与重复编号分别分配种子区间，避免跨根重复。
自然运行只有一次分支选择，不进行候选搜索或 best-of-K；所有成功与失败均计入平均回报。

另外在库存→账本相同的公开根状态，每模型固定采样 **12 个 SPLIT 正文**，共 36 个计划探针。
探针按原分布采样，无去重、修正或失败重采，仅用真实运行器检查首次注册，不进行后续执行、不产生训练奖励。
匹配已审定责任分配时同时比较子契约、父保留职责、义务映射与数据流，允许子节点别名和列表顺序变化。
注册成功但未匹配审定分配的计划单独保留，不直接标为有收益方案。

此外对相同三个根状态计算精确模式概率，并对训练时的固定正文 token 序列重新评分。
固定文本概率是学习方向诊断；新的条件采样频率用于检查语义责任分配是否更常出现。
两者分别报告，不把一条固定文本的似然当成整个语义类别的生成概率。

本轮是相同 TRAIN 根上的新随机种子诊断，没有使用 VALIDATION 或封存任务，评测奖励不回流到更新。

## 训练结果

B3、B4 均实际执行了固定的 2 次更新，初始 teacher-forced 评分与冻结 B0 完全一致。
沿用 PR1 的旧概率约定：原始 cached 生成概率保留，更新分母由同一冻结行为策略按 actor 的完整前缀评分路径重算。
参考策略与 F01 指纹前后一致，保存重载 logits 最大差均为 0。

| 方法 | 实际 optimizer 步数 | 训练用时 | 重载最大差 |
| --- | ---: | ---: | ---: |
| B3 | 2 | 196.45 秒 | 0 |
| B4 | 2 | 156.86 秒 | 0 |

模型与训练记录：[B3](../runs/refinement/PR1C_RL/B3/training_report.json)、[B4](../runs/refinement/PR1C_RL/B4/training_report.json)。
运行在共享 GPU 上，用时记录不作为两种优势公式的速度对照。
B3 复测曾在 GPU 3 加载权重时被系统终止（exit 137），尚未产生偏好评分、计划样本或自然轨迹；
[启动日志](../runs/refinement/PR1C_RL/evaluate_B3_gpu3.log) 保留，随后在 GPU 1 完整完成复测，未删除任何评测样本。

## 评测结果与阶段决策

**本轮未观察到自然运行收益，B4 也未显示优于 B3 的结果。** 全部 90 条自然轨迹与 36 个计划探针已完成，
结果见 [result.json](../runs/refinement/PR1C_RL/result.json)。

| 方法 | 图→库存 | 库存→账本（重点） | 账本→库存 | 合计 |
| --- | ---: | ---: | ---: | ---: |
| B0 | 5/5 | 5/20 | 5/5 | 15/30 |
| B3 | 4/5 | 5/20 | 5/5 | 14/30 |
| B4 | 4/5 | 5/20 | 5/5 | 14/30 |

相对 B0 的配对结果，B3、B4 均为 **0 胜、1 负、29 平**；失败差异出现在同一个图→库存种子 `1019717`。
B0 初始 JSON 被拒绝后恢复并通过；两种更新后的策略均注册了一个不匹配已审定完整职责分配的计划，
随后发生接口/协议错误，未产生根程序。该样本说明本轮局部偏好改善尚未带来完整运行的可靠性提升。
见 [B3 配对失败](../runs/refinement/PR1C_RL/evaluation/B3/natural/train_transfer_graph_inventory_03/repeat_04/case_result.json)
和 [B4 配对失败](../runs/refinement/PR1C_RL/evaluation/B4/natural/train_transfer_graph_inventory_03/repeat_04/case_result.json)。

| 方法 | 重点根 SPLIT 概率 | 优选/合并固定正文概率比的相对变化 | 新采样中优选职责分配 |
| --- | ---: | ---: | ---: |
| B0 | 20.1813% | +0.00% | 7/12 |
| B3 | 20.1813% | +46.50% | 8/12 |
| B4 | 20.1813% | +43.30% | 7/12 |

两种更新均提高了训练中优选正文的完整序列似然，并降低合并正文的似然；B3 的变化略大。
表中的概率比仅比较两条固定完整文本，不代表语义分配类别的总体概率。
新的条件采样中，B3 多出现 1 个优选分配，B4 与 B0 相同；12 次采样用于诊断，不能据此宣称总体频率优势。
模式概率基本没有变化（B3 的绝对变化约 -2.4e-8，B4 为 0）。

**阶段决策：本轮完成根规划的奖励更新闭环，暂不进入 PR2 递归状态训练。**
当前证据支持“固定执行回报能够参与更新并改变正文偏好”，尚未支持自然运行收益或双层优势的额外收益。
后续实验仍应留在根规划，检验多轮新采样与更充分的有效更新；本次 2 步结果不用于否定 RL 路线。
这一判断保留 B0/B3/B4 对照及固定 F01 条件，不根据本轮评测结果追加优化步骤。

## 实际复测成本

| 方法 | 模型输入 token | 输出及评分目标 token |
| --- | ---: | ---: |
| B0 | 362,280 | 11,927 |
| B3 | 364,447 | 11,784 |
| B4 | 364,271 | 11,804 |

以上包含 90 条自然轨迹、36 个条件计划探针和固定文本/模式评分，共计模型输入 1,090,998 token、输出及评分目标 35,515 token。
训练数据复用 PR1C 已有 24 次候选续跑及 3 次基线，不重复记为新采样。
在线验证、离线评分与每条轨迹的实际开销均保存在各方法结果中。


## 检查与复现

定向测试 **20 passed**，覆盖优势计算、token 求和与完整词表 KL、现有课程运行，以及探针对完整责任映射的匹配。
[日志](../runs/refinement/PR1C_RL/pytest_focused.log)。30 个自然种子互不重复，并在方法间配对复用。
两组训练使用相同数据摘要；三组评测的规划器与执行器指纹均前后一致，完整性、配对种子及汇总检查通过。

```bash
# 首次准备：从 PR1C_supported 链接 groups、independent_baseline 和 collect_B0_shard*_request.json
# 到新的 PR1C_RL 输出目录；配置指定同一共同初始化和原始数据。
python -m audittrainer.refinement_pr1 train --config configs/refinement_curriculum_rl.yaml --method B3
python -m audittrainer.refinement_pr1 train --config configs/refinement_curriculum_rl.yaml --method B4
python -m audittrainer.refinement_curriculum_rl evaluate --method B0
python -m audittrainer.refinement_curriculum_rl evaluate --method B3
python -m audittrainer.refinement_curriculum_rl evaluate --method B4
python -m audittrainer.refinement_curriculum_rl report
```

使用 audittrainer conda 环境，`OMP_NUM_THREADS=4`，以 `CUDA_VISIBLE_DEVICES` 选择 GPU；
重现实验使用新的输出目录，保留旧原始训练池和全部评测记录。
实现：[refinement_curriculum_rl.py](../src/audittrainer/refinement_curriculum_rl.py)，训练仍复用既有 [refinement_pr1.py](../src/audittrainer/refinement_pr1.py)。
