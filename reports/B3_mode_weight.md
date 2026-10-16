# B3 模式梯度诊断与单变量权重对照

状态：**COMPLETE（2026-09-15）**。加权梯度、输出精度诊断、权重 4 的十次更新及固定评测全部完成。

**本轮保持模式奖励权重 1。权重 4 的自然通过数为 48/60，低于原重复组的 49/60，配对结果为 0 胜、1 负、59 平。**
唯一退步来自多项式任务：两组都选择合法 SPLIT，但权重 4 丢失了 prepare→aggregate 的责任细化，重新出现父节点组合错误。
已注册根 SPLIT 数量反而从 10/60 增至 11/60，说明提高分解频率不能替代有效责任分配。

## 诊断结论

**首个完整累积窗口中，模式奖励梯度受到 SFT 保持梯度的明显反向作用；正文奖励梯度虽更大，却与模式奖励梯度近乎正交。**
此外，BF16 末层投影会掩盖细小的模式概率变化。因此不能仅凭固定观察的 P(SPLIT) 不变，认定模式没有学习。
这些发现支持一次有限的模式奖励权重对照，不足以预先认定提高权重会增加任务成功率。

复用 B3 重复组的六个 TRAIN 状态、24 个候选、四次续跑均值、四次独立自然基线均值，
在原采样 checkpoint（Adam 第 50 步对应权重）重放第一个完整累积窗口。
顺序和 SFT 保持槽位 150–155 与原训练一致，不采样新轨迹，不执行 optimizer 更新。
评分所用的 frozen behavior teacher-forced 概率直接读取原训练记录；模式和正文初始评分差均为 0。
首窗口保持示范为 4 个 SPLIT、2 个 ATOMIC；十次更新的全部 60 个槽位则为 24 个 SPLIT、36 个 ATOMIC。
以下梯度方向结论限定于首个窗口，不将其外推为每一步都具有相同的梯度关系。

测量包括真实的 `1/256` 全局缩放、六状态梯度累积、SPLIT 概率权重、三个正文候选平均，以及 KL/SFT 权重。
每项独立反传后对完整 43,646,976 个可训练参数累加，并额外执行原组合损失反传进行数值比较。

| 完整批次中的损失项 | 加权梯度范数 |
| --- | ---: |
| 模式奖励 | 0.00038112 |
| 正文奖励 | 0.01368784 |
| 模式 KL | 0.00001560 |
| 正文 KL | 0.00020164 |
| SFT 保持 | 0.00051752 |

正文/模式奖励梯度范数比为 **35.91**，两者余弦为 **0.00250**；模式奖励与 SFT 的余弦为 **−0.64982**。
实际组合梯度范数为 0.01363926，未触发阈值为 1 的梯度裁剪。
BF16 下独立分项反传之和与组合反传存在 **1.11%** 相对范数残差；报告保留该差异，不将数值分解称为逐位相等。
初次诊断和按原组合表达式重放的诊断方向一致，以下采用后者。

进一步对各状态的 SPLIT/ATOMIC 对数概率比求导，检查负损失梯度方向：

| TRAIN 状态 | Q(SPLIT 均值) − Q(ATOMIC) | 原组合方向 | 模式奖励权重 4 的局部方向 |
| --- | ---: | --- | --- |
| calendar → matrix | +0.3333 | 增加分解 | 减少分解 |
| chunks → chunks | −0.5000 | 增加分解 | 减少分解 |
| intervals → matrix | −0.4167 | 增加分解 | 减少分解 |
| inventory → graph | 0 | 增加分解 | 减少分解 |
| inventory → ledger | +0.4167 | 增加分解 | 减少分解 |
| ledger → graph | 0 | 增加分解 | 减少分解 |

这是**无穷小 SGD 方向诊断**，不是 Adam 实际更新量，也不是更新后的实测概率。
权重 4 的方向由原组合梯度加三倍模式奖励梯度估计。
它能够纠正两个直接实现更优状态的方向，同时可能损害两个分解更优状态；必须以实际训练和自然执行判定取舍。

## 模式概率的精度检查

在相同模型隐藏状态上，仅将最后的语言模型输出投影改用 FP32 重新计算，比较原采样权重与重复组第 60 步权重。
这是只读诊断；已有策略、续跑和评测仍使用原 BF16 投影。

六个状态在原采样权重下，首个模式 token 的 BF16 logits 全部为 `[22.5, 21.125]`，
其差均为 1.375，归一化分解概率约 20.18%。FP32 投影揭示了被舍入掩盖的状态差异。

| 固定状态 | 原 BF16 P(SPLIT)，更新前→后 | FP32 末层诊断，更新前→后 |
| --- | --- | --- |
| calendar → matrix | 20.1813% → 22.2700% | 20.5406% → 20.8263% |
| chunks → chunks | 20.1813% → 20.1813% | 20.6319% → 20.6978% |
| intervals → matrix | 20.1813% → 20.1813% | 19.7537% → 20.0301% |
| inventory → graph | 20.1813% → 20.1813% | 20.6917% → 20.8466% |
| inventory → ledger | 20.1813% → 20.1813% | 19.4502% → 18.9859% |
| ledger → graph | 20.1813% → 20.1813% | 19.8441% → 19.6314% |

FP32 末层结果不是新的自然评测通过率，也不能替代当前部署策略的概率。
它说明“BF16 评分未变”与“模型参数没有学到变化”是不同判断。

## 固定的权重对照

保持 B3 统一优势 `A = Q_hat − V_hat_independent`，只改变模式奖励项的系数 α：

`L = scale × [−α × J_mode + β × KL_mode − J_body + β × KL_body + η × L_SFT]`

其中正文项继续包含原有的 SPLIT 概率和候选平均权重，所有状态继续按原规则累积。
KL、SFT、正文 token 求和、学习率、裁剪、奖励、执行器、预算及 BF16 精度均不改变。

| 条件 | 权重 1：复用 B3_repeat | 权重 4：新增分支 |
| --- | --- | --- |
| 起点 | 相同原采样权重、Adam 第 50 步 | 相同 |
| TRAIN 数据 | 同一批 120 条完整轨迹 | 复用同一批 |
| 每候选续跑 / 独立基线 | 4 / 4 | 4 / 4 |
| optimizer 更新 | 10 步，50→60 | 10 步，50→60 |
| SFT 保持槽位 | 150–209 | 150–209 |
| 自然评测 | 已完成 12 根 × 5 个种子 | 相同根、相同种子 |
| 条件计划探针 | 已完成 12 个 | 相同观察和种子，12 个 |

权重 4 从原行为策略重放这批数据，没有把第 50 步的旧轨迹用于继续更新第 60 步模型。
这与重新采样一轮是不同实验：它直接隔离当前批次的模式奖励权重影响，且节省重复采样和对照训练成本。
权重 4 在新训练前固定；不根据评测结果再搜索更多权重或挑选中间 checkpoint。
主指标为最终通过数与配对胜负，分解比例和固定模式概率作为解释指标。
所有任务属于既有 TRAIN，评测种子也来自既有诊断集；这是单训练种子的机制对照，不是独立泛化确认。

权重 4 已完成十次更新，实际 Adam 状态为 50→60，训练循环耗时 782.08 秒。
行为策略初始评分差、保存后重载差均为 0，参考和执行器的权重指纹与原对照保持一致。

### 实际训练后的模式变化

| 固定 TRAIN 状态 | 权重 1：BF16 P(SPLIT) | 权重 4：BF16 P(SPLIT) | FP32 末层诊断：权重 1→4 |
| --- | ---: | ---: | --- |
| calendar → matrix | 22.2700% | 20.1813% | 20.8263% → 20.2418% |
| chunks → chunks | 20.1813% | 20.1813% | 20.6978% → 20.4862% |
| intervals → matrix | 20.1813% | 20.1813% | 20.0301% → 19.7543% |
| inventory → graph | 20.1813% | 22.2700% | 20.8466% → 20.4288% |
| inventory → ledger | 20.1813% | 18.2426% | 18.9859% → 18.9151% |
| ledger → graph | 20.1813% | 20.1813% | 19.6314% → 19.5451% |

FP32 末层诊断下，权重 4 对六个状态均降低了分解概率，包括两个分解更优的状态。
BF16 的变化不总与该诊断同向：inventory→graph 的两项 logits 分别舍入后，模式概率反而上升。
这进一步说明统一放大模式项不能自动获得针对不同状态的粒度选择，且现有输出精度会影响细小更新如何转化为行为。

## 自然执行结果

| 条件 | 更新的六根 | 其余六根 | 总体通过 | 已注册根 SPLIT |
| --- | ---: | ---: | ---: | ---: |
| 权重 1：原 B3_repeat | 19/30 | 30/30 | **49/60** | 10/60 |
| 权重 4 | 19/30 | 29/30 | **48/60** | 11/60 |

两组在相同 60 个根/种子槽位上为 **0 胜、1 负、59 平**。原对照的 11 条失败轨迹均未被修复。
两组各 12 个条件计划探针均注册成功、均对应已有局部检查，类别频数同为 parent_aggregate 9、fused 1、其他 2。
首次计划注册后的成功数为 48→47，首次计划被拒后恢复的成功数均为 1。
唯一已注册根模式变化发生于 inventory→graph、种子 1544713：从 ATOMIC 改为 SPLIT，两者均成功。

### 唯一退步：丢失已获得的责任细化

任务为 `train_transfer_polynomial_polynomial_03`，种子 `1550717`。
它未参加本轮六根更新，但属于此前已训练的十二根 TRAIN 池。

| 项目 | 权重 1 | 权重 4 |
| --- | --- | --- |
| 根模式及首次注册 | SPLIT，成功 | SPLIT，成功 |
| 子责任 | prepare → aggregate | 仅 aggregate |
| 父节点保留 | parse、format | parse、prepare、format |
| 根节点加子节点数 | 3 | 2 |
| 最终检查 | 通过 | 可见和私有检查均因程序异常失败 |

权重 1 正确地先调用 prepare，再把结果交给 aggregate。
权重 4 的父程序直接调用 `aggregate(items)`，将返回的字典误当成预处理列表，再次求值并访问次数，导致组合异常。
该父程序与上一轮起点 B0 的失败代码逐字节相同；两组 aggregate 子组件源码也逐字节相同。
因此本条退步对应责任分配及父组合的退化，没有涉及执行器更新或协议重试恢复差异。

原始证据：[权重 1 计划](../runs/refinement/B3_variance/evaluation/B3_repeat/natural/train_transfer_polynomial_polynomial_03/repeat_04/runtime/plans/N0.v1.json)、
[权重 1 组合代码](../runs/refinement/B3_variance/evaluation/B3_repeat/natural/train_transfer_polynomial_polynomial_03/repeat_04/runtime/artifacts/N0/v1.py)、
[权重 4 计划](../runs/refinement/B3_mode_weight/evaluation/B3_mode4/natural/train_transfer_polynomial_polynomial_03/repeat_04/runtime/plans/N0.v1.json)、
[权重 4 组合代码](../runs/refinement/B3_mode_weight/evaluation/B3_mode4/natural/train_transfer_polynomial_polynomial_03/repeat_04/runtime/artifacts/N0/v1.py)。

### 决策

**继续使用原 B3_repeat checkpoint 和模式权重 1，不采用权重 4 作为默认。**
本轮完成了“诊断→条件触发的小对照→自然执行检验”；结果不支持通过统一放大模式奖励改善当前策略。
共享参数下，调整模式权重也改变了计划正文，本例丢失了此前对固定执行器有帮助的责任细化。
这仍是单训练种子、既有 TRAIN 和固定诊断种子上的结果，不据此断言其他权重或任务分布均无效。
后续模式优化应同时关注状态间相反的粒度偏好，以及采样、训练和评测一致的数值精度；本轮没有扩大权重搜索或改动精度协议。

## 实现、验证与复现

新增 [梯度诊断](../src/audittrainer/refinement_gradients.py)、[精度检查](../scripts/check_b3_gate_precision.py)、
[有限权重对照驱动](../src/audittrainer/refinement_mode_weight.py)和[配置](../configs/refinement_mode_weight.yaml)。
训练器新增可选 `mode_reward_weight`，缺省值为 1，仅作用于模式奖励项。
16 项定向测试通过，涵盖分项梯度累积、B3 credit/clip、正文目标和已有重复续跑数据逻辑。
首个完整累积窗口中，两组正文损失、正文 KL、SFT 保持损失及优势逐项完全相同；
六个模式损失均符合仅将奖励项乘以 4 的计算，KL 没有随之放大。
实际 Adam、同批数据及保持顺序、冻结、重载和 60 条配对评测检查均通过。
本轮新增 **10 次 optimizer 更新、60 条自然轨迹、12 个条件计划探针**，新增 TRAIN 续跑为 0；
梯度和末层精度检查均未更新参数。权重 1 的训练与评测完整复用已有记录。
新自然评测累计墙钟时间为 1221.14 秒；共享硬件下的耗时不用于两组效率优劣结论。

保留的主线模型：[B3_repeat](../runs/refinement/B3_variance/B3_repeat/final_adapter)。
实验模型：[模式权重 4](../runs/refinement/B3_mode_weight/B3_mode4/final_adapter)。

```bash
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 conda run --no-capture-output -n audittrainer \
  python -m audittrainer.refinement_gradients --output runs/refinement/B3_gradients/diagnostic_exact
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 conda run --no-capture-output -n audittrainer \
  python scripts/check_b3_gate_precision.py
conda run --no-capture-output -n audittrainer python -m audittrainer.refinement_mode_weight prepare
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 conda run --no-capture-output -n audittrainer \
  python -m audittrainer.refinement_mode_weight train
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 conda run --no-capture-output -n audittrainer \
  python -m audittrainer.refinement_mode_weight evaluate
conda run --no-capture-output -n audittrainer python -m audittrainer.refinement_mode_weight report
```

诊断和训练输出要求新目录；重现时更换输出位置以保留已有证据。

原始记录：[加权梯度](../runs/refinement/B3_gradients/diagnostic_exact/result.json)、
[末层精度](../runs/refinement/B3_gradients/precision.json)、[预定对照](../runs/refinement/B3_mode_weight/design.json)、
[权重 4 精度检查](../runs/refinement/B3_mode_weight/precision.json)、
[单变量检查](../runs/refinement/B3_mode_weight/control_checks.log)、[训练检查](../runs/refinement/B3_mode_weight/training_checks.log)、
[评测检查](../runs/refinement/B3_mode_weight/evaluation_checks.log)、[完整聚合](../runs/refinement/B3_mode_weight/result.json)、
[测试](../runs/refinement/B3_gradients/pytest.log)。
