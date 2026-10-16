# PR1C 多轮根规划 RL

状态：**COMPLETE（2026-09-14）**。以下预算在本轮采样和评测前固定。

本实验检验持续获得后续执行反馈，能否将此前的固定正文偏好变化转化为真实分解决策和任务完成率的改善。
已完成两组各 5 轮、50 次连续更新，270 条新训练轨迹，以及三组共 180 条自然复测和 36 个计划探针。
**B3 为 31/60，相比 B0 的 29/60 有 2 条配对净收益；B4 为 29/60，尚未显示双层优势的额外收益。**
本轮首次在该路线的根规划对照中观察到小幅自然运行收益；证据限于 3 个 TRAIN 根的新种子诊断，尚不足以确认稳定提升。

## 固定实验条件

共同初始模型为 PR1C 结构 SFT checkpoint；B0 保持该模型，B3/B4 从该模型分别开始。
每种方法独立执行 **5 轮新采样 × 每轮 10 次 optimizer 更新，共 50 次更新**。
每轮采样其当前策略的 3 个根状态，各含 1 个 ATOMIC 与 3 个条件 SPLIT 正文，每个候选 2 次配对续跑，
另有每根 1 次独立自然基线。因此每方法 120 条候选续跑、15 条基线，两方法合计 **270 条新训练轨迹**。
无效动作和重复计划全部保留。方法间使用配对种子；第二轮起各自的数据随策略变化，不要求候选文本相同。

执行器固定为 F01；max_depth=1，修复关闭。任务、奖励 `R=(1-v)y-0.2v`、执行预算均沿用前一轮。
学习率 1e-6，clip 0.2，KL/SFT 保持项各 0.02，objective scale 1/256，梯度范数上限 1。
每次更新累积全部 3 根状态。SFT 保持示范依次遍历、跨轮衔接，B3/B4 顺序匹配。
没有新增独立 SFT 阶段；上述 SFT 保持项仍参与每次优化。B3 使用统一计划优势，B4 分别构造模式与条件正文优势。

## 多轮训练实现

每轮的 PPO 分母由冻结的本轮行为策略重算，继续采用与 actor 一致的完整前缀评分路径，并保留原始缓存生成概率。
模式和正文 KL 始终参照共同的初始 SFT 模型。Adam 状态跨轮继承，累计步数为 10、20、30、40、50。
训练单独检查模式/正文奖励项的实际 adapter 梯度，排除仅由保持项产生梯度的情况；无对应回报差异时如实记录缺少该项信号。
每轮保存冻结检查、重载检查、候选、回报、旧概率和训练记录。

## 训练结果与实际奖励信号

两组均完成全部 5 轮新采样及 50 次连续 optimizer 更新；每轮的行为策略接续上一轮模型，KL 参考始终固定。
10 轮采样和更新的冻结检查全部通过，初始 actor 与行为策略的完整前缀评分差、保存重载 logits 最大差均为 0。
模型与逐轮记录见 [B3](../runs/refinement/PR1C_multiround/B3/training_report.json) 和
[B4](../runs/refinement/PR1C_multiround/B4/training_report.json)。

| 训练信号 | B3：统一计划优势 | B4：模式/正文双层优势 |
| --- | ---: | ---: |
| 根状态采样批次（3 根 × 5 轮） | 15 | 15 |
| 有模式价值差异的批次 | 9/15 | 10/15 |
| SPLIT 候选回报不全相同的批次 | 7/15 | 8/15 |
| 已注册、局部 oracle 覆盖的不同完整分配间有回报差异 | 3/15 | 3/15 |
| 首次注册成功的 SPLIT 样本 | 39/45 | 39/45 |
| 首次注册且局部 oracle 覆盖的 SPLIT 样本 | 37/45 | 36/45 |
| 最终训练链 optimizer 步数 | 50 | 50 |
| 完成的更新阶段累计用时 | 2,201.36 秒 | 2,505.90 秒 |

“15”是重复采样的根状态批次，共涉及 3 个 TRAIN 根。完整分配的匹配同时考虑子契约、父保留职责、
义务映射与数据流，允许别名和列表顺序变化；重复抽到同一分配时先平均回报，再统计不同分配的差异。
注册和 oracle 覆盖不要求匹配某个手写计划模板；这些计数描述本轮观测信号，不把两次续跑的差异当成真实价值的精确估计。

奖励项单独反传的梯度检查确认：B3 第 1–4 轮、B4 全部 5 轮存在模式奖励梯度；
B3 第 1–4 轮及 B4 第 1、3、4、5 轮存在正文奖励梯度。
B4 第 2 轮所有条件 SPLIT 回报相同，其正文相对优势为零，符合双层信用定义；
B3 同轮仍有相对自然基线的正文梯度，但它不能作为不同 SPLIT 正文之间存在收益差异的证据。
B3 第 5 轮全部奖励优势为零，10 次更新只包含 KL/SFT 保持项及继承的 Adam 状态作用。
因此，**50 次 optimizer 更新不等于每次都获得了新的分解比较信号**。

## 资源恢复与成本口径

第二轮 B4 更新时 GPU 1 被其他任务占用，剩余显存约 461 MiB、训练显著变慢，随后迁至 GPU 0。
中断尝试已执行 1 次未保存进最终训练链的 optimizer 更新；迁移后从第一轮检查点及其 Adam 状态重新开始第二轮，复用全部原始采样。
最终模型仍按累计 50 次更新验收；实际计算成本需另计这 1 次丢弃的临时更新及其未完成的后续梯度累积。
中断日志和输出保留于 [train_gpu1_interrupted.log](../runs/refinement/PR1C_multiround/B4/round_02/train_gpu1_interrupted.log)
及 [B4_gpu1_interrupted](../runs/refinement/PR1C_multiround/B4/round_02/B4_gpu1_interrupted)。
第三轮 GPU 0 又有新进程占用约 15 GiB，B4 在首次 optimizer 更新前显存不足退出，已保留
[失败日志](../runs/refinement/PR1C_multiround/B4/round_03/train_gpu0_interrupted.log)，随后从第 20 步检查点继续。
GPU 3 的一次恢复尝试在加载后收到 SIGKILL，同样未执行 optimizer 更新，
[日志](../runs/refinement/PR1C_multiround/B4/round_03/train_gpu3_interrupted.log) 保留。
随后暂时错开两组更新阶段，并为恢复进程启用 PyTorch expandable segments 以减少显存碎片。
第 30 步检查点完整保存后，利用重新释放的 GPU 1 恢复两组并行；此次切换没有丢弃更新。训练配置、采样与最终步数保持原定义。
合计实际执行 **101 次 optimizer 更新**，其中 100 次属于两条最终训练链，1 次为第二轮中断后丢弃的临时更新。
表中更新阶段用时仅累计成功完成的阶段，不含这些中断和重新加载成本；共享 GPU 下的用时不作为算法速度对照。

## 固定最终评测

使用第 5 轮结束的模型，不按中途结果选择 checkpoint。
每模型在库存→账本上评测 40 条自然轨迹，另两个控制根各 10 条，共 60 条；三模型共 **180 条**。
评测种子在模型间配对，独立于本轮训练及此前自然评测。评测回报不用于训练；本轮仍为 TRAIN 根上的新种子诊断。

补充每模型 12 个条件 SPLIT 样本，检查真实注册与完整责任分配匹配；另外重评旧训练池中的相同固定文本用于方向比较。
该旧池仅用于固定诊断，不进入本轮训练。最终报告以自然通过率为主，并报告合法责任分配、协议失败及模式概率。

## 自然运行结果

全部结果见 [result.json](../runs/refinement/PR1C_multiround/result.json)。自然运行沿用协议内的有限重试，
没有额外候选搜索或 best-of-K，所有成功、失败和弃权均纳入分母。

| 方法 | 图→库存 | 库存→账本（重点） | 账本→库存 | 合计 | 相对 B0 的配对胜/负/平 |
| --- | ---: | ---: | ---: | ---: | ---: |
| B0：共同 SFT | 9/10 | 11/40 | 9/10 | 29/60（48.33%） | — |
| B3：统一优势 | 10/10 | 12/40 | 9/10 | 31/60（51.67%） | 2 / 0 / 58 |
| B4：双层优势 | 10/10 | 9/40 | 10/10 | 29/60（48.33%） | 2 / 2 / 56 |

B3 总体提高 3.33 个百分点，重点根提高 2.5 个百分点。两条新增通过均对应根计划正文的变化：

- 图→库存种子 `1139722`：B0 注册了未匹配完整审定分配的计划并失败；B3 改为 prepare child / parent aggregate 后通过。
  见 [B0](../runs/refinement/PR1C_multiround/evaluation/B0/natural/train_transfer_graph_inventory_03/repeat_09/case_result.json)、
  [B3](../runs/refinement/PR1C_multiround/evaluation/B3/natural/train_transfer_graph_inventory_03/repeat_09/case_result.json)。
- 库存→账本种子 `1140736`：B0 的 SPLIT 因无效数据流被拒绝，随后 ATOMIC 执行失败；
  B3 首次注册完整 parent_aggregate 分配并通过。见
  [B0](../runs/refinement/PR1C_multiround/evaluation/B0/natural/train_transfer_inventory_ledger_03/repeat_23/case_result.json)、
  [B3](../runs/refinement/PR1C_multiround/evaluation/B3/natural/train_transfer_inventory_ledger_03/repeat_23/case_result.json)。

B4 在两个控制根各增加 1 次通过，同时在重点根减少 2 次通过。重点根种子 `1140747` 中，
B0 选择 SPLIT 并通过，B4 改为 ATOMIC 后失败；种子 `1140726` 中，两组均注册 SPLIT，B4 后续弃权。
见 [B4 模式变化样本](../runs/refinement/PR1C_multiround/evaluation/B4/natural/train_transfer_inventory_ledger_03/repeat_34/case_result.json)
和 [B4 弃权样本](../runs/refinement/PR1C_multiround/evaluation/B4/natural/train_transfer_inventory_ledger_03/repeat_13/case_result.json)。
因此，模式发生变化并不自动意味着粒度选择更好。

| 运行诊断 | B0 | B3 | B4 |
| --- | ---: | ---: | ---: |
| 根首次计划尝试注册成功 | 56/60 | 56/60 | 58/60 |
| 根最终接受的计划为 SPLIT | 11/60 | 10/60 | 11/60 |
| 协议重试耗尽 | 0 | 1 | 0 |
| 弃权 | 0 | 0 | 3 |
| 重点根接受 parent_aggregate 完整分配 | 6/40 | 6/40 | 5/40 |

上述模式计数只取根节点首先接受的计划，不混入叶节点的 ATOMIC 动作；B3 有 1 条轨迹未注册根计划。
没有独立确认的责任违规，所有根的语义责任状态仍为 UNKNOWN；通过数衡量功能执行结果，不能替代完整语义责任证明。

## 计划分布诊断

| 方法 | 重点根 SPLIT 概率 | parent_aggregate / fused 固定文本概率比相对 B0 | 条件计划首次注册 | 局部 oracle 覆盖 | parent_aggregate 样本 |
| --- | ---: | ---: | ---: | ---: | ---: |
| B0 | 20.1813% | 1.0000 倍 | 9/12 | 9/12 | 7/12 |
| B3 | 20.1813% | 12.5008 倍 | 10/12 | 9/12 | 7/12 |
| B4 | 18.2426% | 0.2910 倍 | 11/12 | 10/12 | 6/12 |

比值只比较旧诊断池的两条完整 token 序列。B3 的 parent_aggregate 对数似然变化为 -0.0835、fused 为 -2.6093，
相对比值大幅提高主要来自 fused 文本似然降低；新采样中的 parent_aggregate 数量仍为 7/12。
B4 的对应变化为 -0.3120 和 +0.9223。**固定文本偏好变化不能替代责任分配的实际采样与执行评测。**

完整分配匹配中，B3 另外出现 pipeline 1 次、fused 1 次；其余未匹配样本数为 B0 5、B3 3、B4 6。
未匹配包含注册失败或其他责任分配，不等同于全部非法。36 个探针不续跑、不产生优化奖励；12 次/模型用于行为诊断，
不足以据此确认语义类别频率的稳定变化。增加 SPLIT 概率或树深度本身不是本实验的成功指标。

## 阶段结论

多轮根规划训练已把当前行为策略采样、完整后续回报、连续参数更新和新种子执行评测连接起来。
B3 的两条配对净收益提供了计划正文变化改善下游运行的初步证据；B4 的重点根退步和总体持平，
则说明目前尚未建立双层信用优于统一优势的证据。

当前主要经验问题是**已注册完整分配之间的有效比较信号仍少，以及这些信号能否稳定改变实际责任分配与执行收益**。
该判断来自每组仅 3/15 批次有这类回报差异，以及固定文本变化与真实采样频率未同步的观测。
本轮按固定预算结束，保留根规划研究范围，PR2 中间节点训练尚未启动。
3 个训练根、单条训练随机链及少量配对差异不支持跨任务泛化或稳定优势结论；
由于优化保留共同 KL/SFT 项，B3 对 B0 的变化也不能单独归因于奖励项。B3/B4 的对照保持了这些共同设置。

## 实际采样与评测成本

| 阶段/方法 | 新轨迹数 | 模型输入 token | 输出及评分目标 token |
| --- | ---: | ---: | ---: |
| B3 训练采样 | 135 | 1,451,284 | 52,923 |
| B4 训练采样 | 135 | 1,461,165 | 53,488 |
| B0 最终评测 | 60 | 664,656 | 20,569 |
| B3 最终评测 | 60 | 636,336 | 19,918 |
| B4 最终评测 | 60 | 623,382 | 19,262 |

最终评测的模型成本另包含每组 12 个不续跑的条件计划探针和固定文本/模式评分；这些探针不计为完整执行轨迹。
表中采样 token 不包含训练反向传播的 token 曝光，更新次数及更新阶段时间另见前文。
旧诊断池仅重评分，不重复记为新训练轨迹。在线验证、离线评分和逐轨迹开销保存在原始结果中。

## 检查与复现

定向测试 **22 passed**，覆盖当前行为分母与固定 KL 参考的分离、奖励梯度方向、跨轮配置/种子，
以及已有优势、KL、课程运行和完整责任分配匹配。见 [测试日志](../runs/refinement/PR1C_multiround/pytest_final.log)。
10 个轮次的行为模型衔接、连续 Adam 步数、冻结与保存重载检查通过；
三组各 60 个唯一根/种子对一致，训练与自然评测种子分离，评测未产生任何 optimizer 更新。
270 条训练轨迹、180 条自然轨迹、36 个探针和 101 次实际 optimizer 更新均已核对，
见 [聚合检查](../runs/refinement/PR1C_multiround/integrity_checks.log)。

配置：[refinement_multiround.yaml](../configs/refinement_multiround.yaml)。
实现：[refinement_multiround.py](../src/audittrainer/refinement_multiround.py)，复用原有采样、训练、评测器。

```bash
python -m audittrainer.refinement_multiround run --method B3
python -m audittrainer.refinement_multiround run --method B4
python -m audittrainer.refinement_multiround evaluate --method B0
python -m audittrainer.refinement_multiround evaluate --method B3
python -m audittrainer.refinement_multiround evaluate --method B4
python -m audittrainer.refinement_multiround report
```

使用 audittrainer conda 环境及 `OMP_NUM_THREADS=4`；以 `CUDA_VISIBLE_DEVICES` 选择 GPU。
输出：[PR1C_multiround](../runs/refinement/PR1C_multiround)。
