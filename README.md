# AuditTrainer

从公开 Qwen3-8B 底座重新建立责任分解、契约履行与组合、有界修复的研究实现。
实施依据为仓库中的 v3.0-R1 从零实现路线。

> 训练集、评测集、运行轨迹、生成数据和 Qwen3-8B 权重不随代码上传；固定版本及下载位置见 [`data/README.md`](data/README.md)。

A0、F1–F9、A2、A3、A4 pilot 与 A5 已完成。正式训练集含 50 个 TRAIN root、2,240 条通过检查的动作；
A5 已完成四组各 140 步训练与 180 个开发评测运行；A6 的一次父职责覆盖修订、140 步训练与
45 个开发运行也已完成。PRRPO 支线的 PR0、PR1 单批根规划 RL pilot、PR1C 训练条件建设、有限及多轮 RL 对照，以及 B3 任务覆盖、回报估计和模式权重对照已完成；原路线的 A7 仍待开展。
验收状态见 [STATUS.md](STATUS.md)。

## calendar 明确目标正对照：完成，完整收益尚未形成

**calendar 明确好方案的 SFT 正对照**已完成：复用 36 条既有 TRAIN 校准，新增 60 次更新、128 条条件计划探针、192 条自然评测；无新增训练续跑、无 RL 更新。
共同起点 / 仅门控 / 仅正文 / 联合监督的既定测试通过分别为 **24/48、25/48、22/48、22/48**；首次完整好方案为 **10、12、7、10/48**。
精确分解概率为 22.27%、32.08%、22.27%、29.42%；正文拟合改善未形成稳定的条件分配增益。门控组两块净收益为 +1、0，联合组为 -1、-1，未打通预定学习到执行的完整过程。
另发现门控组一条“拒绝后恢复通过”的代码在公开合法输入上出错，既定评分存在具体盲点；保留主指标原值，补充 CPU 反例证据。
**保留共同 SFT＋冻结摘要，不替换新检查点、不扩大 RL。** 下一次采集回报前先修补明确评分盲点；本轮不能证明 SFT 不可学或 RL 必需。
8 项定向测试以及冻结、重载和配对检查通过。见 [calendar 正对照报告](reports/B3_calendar_target.md)。

## 固定计划验证：完成，定位到可学习的分解收益

**固定计划的分解收益验证**已完成：两个 TRAIN 根、三种固定合法计划、12 个配对新种子，共 72 条续跑、0 次更新；执行种子在根计划注册后重置。
calendar→matrix 的 ATOMIC / 父聚合 / 双子计划分别为 **4/12、12/12、5/12**；父聚合配对 8 胜 0 负，两块均净胜 4 条，但五个现有模型只在 3–4/32 次首次规划中选择该完整方案。
intervals→matrix 分别为 **9/12、9/12、11/12**；双子计划两块各净胜 1 条，证据较弱；父聚合的 3 次失败均遗漏 JSON 序列化。
**下一笔小规模预算优先投入 calendar 的规划学习与选择机制，不扩大当前 B3 训练规模，也不将父聚合作为所有任务的统一最优标签。** 分解的 token 成本明显更高，收益结论限定于当前预算内完成率。
复用已有 480 条轨迹中的 320 条相关轨迹完成选择诊断；6 项定向测试及计划、摘要、模型冻结、种子和配对检查通过。
见 [固定计划报告](reports/B3_fixed_plan.md)。

## 统一摘要下的规划训练：完成，B3 额外收益未复现

**冻结摘要下的成功计划 SFT 与 B3 公平对照**已完成：240 条新 TRAIN 续跑、80 次更新、480 条自然评测。
两批分别有 3/3、2/3 个状态存在合法且有局部检查覆盖的分配收益差异。共同 SFT 在两组新种子上为 28/48、34/48；
成功 SFT 两批为 27/48、34/48 和 26/48、35/48；B3 两批为 28/48、34/48 和 26/48、35/48。
B3 相对成功 SFT 的四组净收益为 +1、0、0、0，没有达到跨批次、跨种子重复获益标准；首次接受且成功合计为 118 对 119。
**保留共同结构 SFT＋冻结摘要作为参考，不扩大当前 B3 训练或递归深度。** 成功计划 SFT 同样没有证明优于共同起点。
四模型空 Adam 起步、冻结及重载、同批共享反馈、全部配对与摘要输入检查通过；12 项定向测试通过。
见 [公平对照报告](reports/B3_summary_training.md)。

## 冻结摘要的新状态验证：完成，净收益跨任务复现

16 个新父状态、12 个与设计轮不重叠的任务根，完成 128 次配对续跑、0 次更新，摘要保持不变。
原输入 **55/64**，摘要 **61/64**，配对 **8 胜、2 负、54 平**；3 个状态、2 个新任务根获益，达到事先固定的新标准。
字典层两组均 24/24，未见键值混淆复现；两次新退步来自区间聚合。总模型 tokens 增加 **20.03%**，含两次协议重试。
支持下一阶段统一采用冻结摘要，再在 TRAIN 重采规划回报、比较成功 SFT 与 B3；本轮仅完成验证，上一轮结论保留。
7 项定向测试及完整冻结、配对检查通过。见 [新状态验证报告](reports/B3_parent_validation.md)。

## 父节点组合输入对照：完成，有修复价值但未统一采用

冻结 12 个 TRAIN 父状态的计划、已通过局部检查的子组件和模型权重，完成 96 次配对父组合续跑、0 次更新。
原输入 **41/48**，职责/接口摘要 **47/48**，配对 **7 胜、1 负、40 平**。
摘要修复了父职责遗漏和子输入误用，但原成功状态 **40/40→39/40**，新增一次父聚合的键/值与返回结构错误。
未达到预定的无净退步条件，保留默认输入，不扩大规划 RL；摘要作为实验条件保留。4 项定向测试及实际冻结、配对检查通过。
见 [父组合实验报告](reports/B3_parent_composition.md)。

## 新执行种子的 SFT 基线：完成，暂停增加训练量

新增三个 SFT 模型共 144 条自然执行，复用原 B3 的 96 条同任务、同种子结果，未训练或修改奖励。
共同 SFT 为 **30/48**；两批成功计划 SFT 均为 **31/48**，原 B3 为 **30/48、33/48**。
B3 相对成功 SFT 为 **1 胜 2 负 45 平、3 胜 1 负 44 平**，额外终局收益没有在两批中一致复现。
按既定标准暂停增加训练量，不扩任务池或递归训练；轨迹已定位预处理/聚合接口误用、父职责遗漏及下游执行差异。
11 项定向测试及完整性检查通过。见 [补评测报告](reports/B3_sft_replication.md)。

## 首次规划信用分配：完成，不采用归零方案

复用已有 240 条 TRAIN 执行数据，候选和独立基线一致重计分，完成 40 次新增更新、192 条新种子自然评测。
首次接受且最终成功：两批分别 **29→29、31→31**；总体成功：**30→30、33→32**（每模型 48 条）。
两批根规划重试均从 2 次降至 1 次，但主指标没有净提升，第二批损失一次恢复成功，未达到采用条件。
保留原 B3 的完整后续回报，本轮不扩大任务覆盖或递归训练。24 项定向测试及完整性检查通过。
见 [实验报告与轨迹分析](reports/B3_first_credit.md)。

## 成功计划 SFT 与 B3：完成

三个预选 TRAIN 状态、两个独立共享反馈批次的可学习性对照已完成。共同 SFT 为 35/48；
成功计划 SFT 为 36/48、35/48，B3 均为 **37/48**，相对同批成功 SFT 为 1 胜、2 胜，均无退步。
B3 两批的平均优选概率估计均更高，实际初始优选次数分别与成功 SFT 持平、领先。
结果支持继续以 B3 为研究主线；第一批额外成功来自初始拒绝后的重试，不能将全部收益解释为首次分解质量提升。
完成 48 条独立校准、240 条共享训练续跑、80 次更新、240 条自然评测及 360 个计划探针；23 项定向测试和完整性检查通过。
本轮是既有 TRAIN 状态的可学习性对照。见 [实验报告](reports/B3_success_sft.md)。

## B3 独立组合泛化验证：完成

六个 RL 模型从共同 SFT 起点完成三个训练种子的等预算比较，12 个未训练新组合上的 420 条独立评测已完成。
SFT 为 51/60；原 B3 等预算广采样版为 51、51、52，优化 B3 为 52、52、51。
优化组相对原组平均仅多通过 0.33/60 条，尚未显示稳定优势；相对 SFT 的新增成功集中在同一个 chunks→words 任务/执行种子。
完成 1,440 条 TRAIN 续跑、240 次更新，288 次参考答案检查及 34 项定向测试通过。见 [独立验证报告](reports/B3_generalization.md)。

## B3 模式权重：完成诊断后保留权重 1

在原重复组首个完整累积窗口中，正文/模式奖励梯度范数比为 35.91，两者近乎正交；
模式奖励与 SFT 保持梯度余弦为 −0.64982。末层检查还发现 BF16 会掩盖细小的模式概率变化。
据此完成模式奖励权重 1 对 4 的单变量实验：复用相同 120 条 TRAIN 轨迹及原对照，
权重 4 从相同 Adam 第 50 步起点新增 10 次更新，并完成 60 条配对自然评测和 12 个计划探针。

| 模式奖励权重 | 自然通过 | 已注册根 SPLIT |
| --- | ---: | ---: |
| 1：原 B3_repeat | **49/60** | 10/60 |
| 4 | **48/60** | 11/60 |

权重 4 为 0 胜、1 负、59 平。唯一退步中，两组都选择合法 SPLIT，但权重 4 丢失 prepare→aggregate 的责任细化，
复现了先前的父组合错误；aggregate 子组件源码相同。增加分解频率没有改善本轮任务结果。
继续使用 B3_repeat 和模式权重 1；这是单训练种子、既有 TRAIN 诊断集上的结论。
16 项定向测试及单变量、Adam 衔接、冻结、重载和评测检查通过。见 [完整报告](reports/B3_mode_weight.md)。

## B3 回报估计：相同执行预算下的广采样与重复续跑

从上一轮轮换 B3 的相同权重和 Adam 第 50 步继续，固定六个 TRAIN 根。
广采样组每根采 8 或 9 个 SPLIT、每候选续跑 2 次、自然基线 1 次；重复组为 3 个 SPLIT、每候选续跑 4 次、基线 4 次取平均。
两组均为 120 条训练轨迹、10 次新增更新；另完成 120 条独立回报检查。

相对独立四次检查均值，重复组的统一优势均方误差从 0.359375 降至 0.135417，下降 **62.3%**。
新的 12 根、每根 5 个配对种子的自然评测结果为：

| 方法 | 通过 |
| --- | ---: |
| 起点 B3（内部 B0） | 48/60 |
| 广采样 B3_wide | 48/60 |
| 重复续跑 B3_repeat | **49/60** |

重复组对广采样为 1 胜、0 负、59 平。新增成功来自多项式任务的首次合法计划：
将预处理下放为 prepare 子任务，并明确连接到 aggregate，帮助固定执行器正确组合。
该根未参与本轮六根更新，但已在之前的 TRAIN 池中出现；本轮仍是单训练随机链的 pilot。
广采样覆盖了更多有回报差异的合法、局部检查覆盖的责任划分（5/6 对 3/6），重复组则改善了共同候选的估计。
合计完成 540 条轨迹、36 个探针和 20 次新增更新；44 项定向测试及采样、衔接、冻结、重载、评测检查通过。
配置、模型、误差诊断和新增成功的代码证据见 [B3 回报估计报告](reports/B3_variance.md)。

## B3 任务覆盖：固定 3 根与轮换 12 根

完成 192 条 TRAIN 执行校准，从共同 SFT 初始化新训轮换 B3，并复用已有固定 B3 作为对照。
两组均为 135 条训练轨迹、50 次连续更新；奖励、执行器、学习率及候选/基线续跑设置一致。
新的 12 根均衡评测集每根使用 5 个相同的新种子：

| 方法 | 原有 3 根 | 新增 9 根 | 总体通过 |
| --- | ---: | ---: | ---: |
| B0：共同 SFT | 11/15 | 40/45 | 51/60 |
| 固定 3 根 B3 | 11/15 | 40/45 | 51/60 |
| 轮换 12 根 B3 | 11/15 | 41/45 | 52/60 |

轮换组相对固定组为 1 胜、0 负、59 平；这条新增成功来自分解被拒后重试原子实现。
有效责任分配比较从 3/15 批次降至 2/15，扩大覆盖尚未增加该目标信号。
本轮观察到小幅自然运行收益，不能据此认定初始合法分解或粒度选择能力提升。
28 项定向测试及训练衔接、冻结、重载、配对汇总检查通过。
完整数据、解释及复现见 [B3 任务覆盖报告](reports/B3_coverage.md)。

## PR1C 多轮 RL：持续采样与连续更新

B3/B4 从共同结构 SFT 模型各完成 5 轮新采样、50 次连续 optimizer 更新，共 270 条新训练轨迹。
每轮刷新行为策略，KL 参考与 F01 执行器固定，Adam 状态跨轮继承；没有新增独立 SFT 阶段。
最终三组完成 180 条配对新种子自然复测和 36 个条件计划探针：

| 方法 | 库存→账本（重点） | 总体通过 |
| --- | ---: | ---: |
| B0：共同 SFT | 11/40 | 29/60 |
| B3：统一优势 | 12/40 | 31/60 |
| B4：双层优势 | 9/40 | 29/60 |

B3 相对 B0 为 2 胜、0 负、58 平，新增通过对应更完整的责任分配正文；B4 为 2 胜、2 负、56 平，尚无额外收益。
每组只有 3/15 个采样批次在已注册、局部 oracle 覆盖的不同完整分配间出现回报差异。
本轮观察到小幅样本收益，证据仍限于 3 个 TRAIN 根的新种子诊断；PR2 中间节点训练尚未启动。
22 项定向测试及训练衔接、冻结、重载、配对汇总检查通过。配置、模型与复现见 [多轮 RL 报告](reports/PR1C_multiround.md)。

## PR1C RL：执行回报驱动的根规划对照

复用 PR1C 的共同初始化与原始训练池，B3、B4 各完成 2 次真实 RL 更新，继续冻结 F01。
三组完成相同新种子的 90 条自然轨迹和 36 个条件计划探针：B0 为 15/30，B3、B4 均为 14/30；
重点库存→账本任务三组均为 5/20。优选正文相对合并正文的固定文本概率比提高了 46.5%（B3）和 43.3%（B4），
模式概率基本不变，本轮尚未观察到自然运行收益或双层优势的额外收益。
该有限对照促成了上述根规划多轮采样与连续更新实验。

20 项定向测试以及冻结、重载、配对汇总检查通过；无新增 SFT 阶段，无 VALIDATION 或封存任务参与。
配置、模型、结果与复现命令见 [PR1C RL 报告](reports/PR1C_RL.md)。

## PR1C：可执行且有分解价值差异的训练条件

完成 9 个 TRAIN 根的课程校准与一次共同结构 SFT，共 252 次运行、27 次 SFT 更新。
固定 F01 执行器，根可分解、叶节点原子执行；固定计划 130/144 功能通过，存在两个种子块均出现的合法分解收益差异。
补训后的原始采样中，库存→账本的 prepare child / parent aggregate 为 2/2，compute child 为 0/2；
两者均初始注册成功且有精确局部 oracle，均只有一个子节点。

[训练条件配置](configs/refinement_curriculum_ready.yaml) 对应 3 个根、12 个原始候选、24 次续跑与 3 次独立基线，
已通过现有 RL 加载器检查，状态 READY；非法和重复候选全部保留。本轮未新增 RL 更新，未使用 VALIDATION 或封存任务。
31 项定向测试及权重冻结、模型重载与聚合检查通过。数据、模型、结果与复现命令见 [PR1C 报告](reports/PR1C.md)。

## PR1：共同格式补训与真实根规划 RL

已完成 25 次共同格式 SFT 更新、80 次候选续跑、10 次独立自然基线，以及 B3/B4 各 10 次实际 RL 更新。
同状态格式复测的可解析 SPLIT 从 6/30 提高到 22/30；两种 RL 方法使用相同原始候选、冻结 F01 执行器及匹配更新设置。
B0/B3/B4 各完成 18 条自然评估，均为 TRAIN 0/10、VALIDATION 0/8；本轮未观察到 RL 带来的完成率提升。
只有一个训练状态提供数据流合法性相关的回报差异，尚无已注册分解之间的粒度收益信号。
284 项测试、权重冻结、保存重载与最终聚合通过。实现、配置、模型与结果见 [PR1 报告](reports/PR1.md)。

## PR0：规划表示适配与零更新诊断

依据 [PRRPO 设计](docs/决策分解改进设计_PRRPO_2026-09-10.md)，使用 A6 F11_C 初始化规划器，
固定 A5 F01 执行器。模式采用完整多 token 候选评分，SPLIT 正文按条件分布采样，
ATOMIC 正文由宿主规范化。新协议检查责任映射的结构覆盖，语义履约仍由执行证据判断。

[refinement.yaml](configs/refinement.yaml) 固定 200 条 TRAIN 表示适配示范、25 次 SFT 更新，
以及 50 个 TRAIN 根状态 × 4 个候选 × 2 次续跑。PR0 不执行 RL 更新；保留非法、重复和截断候选。
实际完成 400 次续跑：ATOMIC 100/100 通过，SPLIT 候选起始的续跑 72/300 通过，其中 19 次来自初始计划被拒绝后的重试恢复。
SPLIT 初始注册 34/150，3 个状态有已注册分解间的回报差异；平均 SPLIT 选择概率为 92.64%。
273 项测试、权重冻结、保存重载和全量聚合检查通过。
这些结果促成了 PR1 的共同格式修订、TRAIN 条件调整及有限 RL 对照。
完整结果与复现命令见 [PR0 报告](reports/PR0.md)。

## A6 有限数据修订

[a6.yaml](configs/a6.yaml) 固定一次主要因素变化：在既有 50 个 TRAIN root 上新增
prepare child / parent aggregate 的职责分配，实际执行并审查后，只替换 A5 F11 采样表中
96 个父组合 `write_code` 槽位。其余 2,144 个槽位、训练超参数和 A5 评测协议不变。
F11_C 从公开底座新训 140 步，以 A5 F11 为原始对照；DPO 本轮不启用。

单 child 职责重分配通过数 1/8 → 2/8，固定组件 12/16 → 14/16；双 child 组合
7/8 → 6/8，自由运行仍为 1/5。全量测试 261 项通过，权重重载、运行回接与最终聚合验收通过。
本轮为单 seed 开发比较，按一次修订规则结束 A6，保留原四组及独立的 F11_C 权重。

```bash
conda run --no-capture-output -n audittrainer python -m audittrainer.a6 collect
conda run --no-capture-output -n audittrainer python -m audittrainer.a6 prepare
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 conda run --no-capture-output -n audittrainer python -m audittrainer.train_sft --config configs/a6_training.yaml
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 conda run --no-capture-output -n audittrainer python -m audittrainer.a6 evaluate
conda run --no-capture-output -n audittrainer python -m audittrainer.a6 report
```

已有目录不覆盖，完整实验约定和实际结果见 [A6 报告](reports/A6.md)。

## A5 核心训练与能力隔离

四组 F00/F10/F01/F11 使用同一公开 theta0、seed 42、50 个 TRAIN root、2,240 个匹配训练槽位和
140 次更新，分别关闭/开启责任分解 D 与模块履约 I 的针对性示范。未开启的槽位替换为同 root
新执行的普通功能示范；consume 和控制动作共享。输入与目标 token、不同样本曝光量的实际差异
见 [数据对照](runs/a5/data_comparison.json)，不把该比较解释为仅改标签的消融。

```bash
# 数据只需首次创建；已有目录不会覆盖。
conda run --no-capture-output -n audittrainer python -m audittrainer.a5_data collect-functional
conda run --no-capture-output -n audittrainer python -m audittrainer.a5_data prepare-groups
# group 依次为 F00、F10、F01、F11，均从 theta0 新训。
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 conda run --no-capture-output -n audittrainer python -m audittrainer.train_sft --config configs/a5/F11.yaml
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 conda run --no-capture-output -n audittrainer python -m audittrainer.a5_evaluate --group F11 --adapter runs/a5/train/F11_seed42/final_adapter
# 四组训练与评测完成后聚合。
conda run --no-capture-output -n audittrainer python -m audittrainer.a5_report
```

正式四组统一使用 CPU 暂存反向所需张量，保留完整 action、8,192 token 上限和相同损失。
F00/F11 的 5 步 pilot 只验证训练链路，正式训练不继承 pilot 更新。
[评测配置](configs/a5_evaluation.yaml) 固定 8 个 DEV_MECHANISM root 的 40 次能力隔离运行，
以及全部 5 个 DEV_ENGINEERING root 的自由递归运行。局部 plan 之后统一由公开 theta0 执行；
固定契约实验包括两 child 与单 child 的合法责任重分配，子产物只读。完整证据见 [A5 报告](reports/A5.md)。

F00/F10/F01/F11 固定组件履约通过数为 0/16、3/16、13/16、12/16，自由递归为
1/5、1/5、2/5、1/5。履约监督改善了固定契约执行与双 child 组合；父职责重分配和计划格式
仍是明确瓶颈。本轮为单 seed 开发比较，实际 token/不同样本曝光差异见报告。

## A4 TRAIN pilot

[data_pilot.yaml](configs/data_pilot.yaml) 选择 50 个新 TRAIN root，覆盖账目、区间、图、库存、文本、
分块、矩阵、时间、多项式和路径 10 个任务族，每族 5 个参数配置。同族全部留在 TRAIN，
不把 50 个参数化任务称作 50 个独立任务族。

每个 root 用同一固定 `E_ref`、同一预算运行 ATOMIC、两 child pipeline、单 child fused 三种计划。
示范策略是明确标记的可审查参考执行器。另从每族固定选参数 3 的 root，以公开底座采集 K=1 的自然轨迹。
受控对照覆盖遗漏输出、迁移算法、调用后弃用、真实旧 artifact 版本、错误参数映射、扩大 owns 六类。

```bash
# 初次构造；已有目录/导出保留，重现应另设配置中的输出路径。
conda run --no-capture-output -n audittrainer python -m audittrainer.cli import-roots
conda run --no-capture-output -n audittrainer python -m audittrainer.cli build-demos
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 conda run --no-capture-output -n audittrainer python -m audittrainer.cli collect
conda run --no-capture-output -n audittrainer python -m audittrainer.cli make-contrasts
conda run --no-capture-output -n audittrainer python -m audittrainer.cli validate-labels
conda run --no-capture-output -n audittrainer python -m audittrainer.cli export-sft --config configs/frozen_mechanism.yaml --trace-dir data/raw_traces/a4 --split TRAIN --max-length 8192 --out data/sft/train.jsonl
conda run --no-capture-output -n audittrainer python -m audittrainer.cli make-schedule
```

前六个数据命令接受 `--config`；示范/采集/对照可用 `--limit` 做小批调试。
`import-roots --input tasks.jsonl` 接受规范化任务包，每行包含 `definition`（task、contract、
local_contracts、可见 suites）、`private`（私有 suites）、`source`、`original_split`，
可选参考字段 `reference`。本轮默认导入新建的封闭任务目录，不重新标记已消费开发题。

独立 oracle 检查真实执行；责任审查只对已审定的组件/组合或证据明确的单因素变换给出强标签，
自由模型代码无法确认时保留 UNKNOWN。分层表区分计划、实现、消费、父组合、未知与受控负例，
记录程序审查来源及真人复核状态。原始失败、拒收、离线检查和旧协议诊断均保留。

A4 修复了隐藏的每节点 child 上限：当前协议 `at-v3r1-a3.2` 展示 `budget_remaining.child_slots`，
并在拒绝时说明具体限制；预算值不变。旧 3.1 输入归档并重跑，影响清单见
`runs/a4/protocol_bug_impact.json`。完整产物与验收见 [A4 报告](reports/A4.md)。

当前 [train.jsonl](data/sft/train.jsonl) 含 D_PLAN=460、D_IMPL=320、D_COMPOSE=400、D_CONTROL=1,060。
[采样清单](runs/a4/sampling_schedule.jsonl) 已按主 SFT 配置生成，计划 140 个 optimizer step；
A4 阶段不执行参数更新，四组训练在 A5 进行。20 对功能匹配责任违规见 [对照标签](runs/a4/contrasts.jsonl)，
用户已报告人工复核无问题，Codex 二次复核通过；数据验收已完成。
补查了原抽查表未覆盖的真实计划及其余负例类型，结果与复核命令见 [A4 报告](reports/A4.md#数据验收与二次复核2026-09-08)。

## A3 冻结主机制

[frozen_mechanism.yaml](configs/frozen_mechanism.yaml) 是后续主线的运行配置：逐节点层次契约、
函数/模块所有权、显式消费，使用 R1 责任筛选定位与 F4 冻结补集修复。
动作与 observation renderer 保持原有格式；首次冻结协议为 `at-v3r1-a3.1`，A4 预算可见性修订为 `at-v3r1-a3.2`。
候选最多 3 个、补丁最多 3 次、父边界最多扩大一次；根共享 64 次策略调用、32 次验证/重放预算。

[数据与检查协议 v1](docs/data_check_protocol_v1.md) 固定 oracle/UNKNOWN、责任正例与功能对照、
负例判据、分区、抽样及训练规则。导出 manifest 和样本记录 `at-data-check-v1`，训练入口检查版本。
`sft_main.yaml` 已指向冻结配置；旧 `sft_smoke.yaml` 仍读取 F6 协议，旧 smoke 数据可继续用于连通验收。

```bash
# 验证冻结配置的运行链路；输出目录应为新路径。
conda run --no-capture-output -n audittrainer python -m audittrainer.cli evaluate --config configs/evaluation_main.yaml --policy script --split SYN_FIXTURE --limit 6 --output-dir runs/a3_fixture_new

# 主机制的开发评测（默认 DEV_ENGINEERING，基础模型或后续正式 adapter）。
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 conda run --no-capture-output -n audittrainer python -m audittrainer.cli evaluate --config configs/evaluation_main.yaml --policy base --output-dir runs/main_dev_new

# A4 完成 TRAIN 轨迹与独立审查后导出；A3 不生成正式训练数据。
conda run --no-capture-output -n audittrainer python -m audittrainer.cli export-sft --config configs/frozen_mechanism.yaml --trace-dir runs/train_traces --split TRAIN --max-length 8192 --out data/sft/train.jsonl

# A5 四组配置继承主 SFT 配置，并绑定各自的数据与匹配采样清单。
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 conda run --no-capture-output -n audittrainer python -m audittrainer.train_sft --config configs/a5/F11.yaml
```

A2 支持采用 R1 作为定位起点；R1 与 F4 强冻结组合的学习收益仍由后续实验检验。
本次选择理由、配置入口和实际验收见 [A3 报告](reports/A3.md)。

## A2 有限机制比较

`mechanism` 在同一批 20 个 `DEV_MECHANISM` root 上比较责任粒度、事后链接/静态计划/逐节点承诺，
以及日志定位、责任筛选与有限重放。共五种生成条件、三种修复条件；P1/P2 共享 hierarchical 轨迹。
模型固定为无 adapter 的 Qwen3-8B，使用真实推理、现有 runtime、Bubblewrap 和可见测试选择。
私有测试只评分已经选定的产物，开发数据不进入 SFT。

```bash
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 conda run --no-capture-output -n audittrainer \
  python -m audittrainer.cli mechanism --config configs/mechanism.yaml
```

重现时先复制配置并设置尚不存在的 `output_dir`；驱动不会覆盖既有结果。
固定任务已经登记，`mechanism --build-tasks` 仅用于首次构造空数据目录。
输出包括 `mechanism_p1.json`、`mechanism_p2.json`、`mechanism_p3.json`、逐 root/arm 的完整轨迹、
模型原始输出、候选源码、冻结快照与成本。实验结果、条件遵守情况和选择理由见 [A2 报告](reports/A2.md)。

## F9 统一评测与 APPS 导入

`evaluate` 对 script、base、SFT 输出相同的逐 root 字段。先在第一次完整 root 代码被接受时，
保存该版本及其全部依赖；结束时再保存当前 root 产物。两个快照独立，后续修复不能覆盖首个程序。
在线 verifier 仅持有 VISIBLE suite；候选选定并结束模型决策后，才进行离线可见/私有评分。

```bash
conda run --no-capture-output -n audittrainer python -m audittrainer.cli evaluate --policy script --split SYN_FIXTURE --limit 6 --max-policy-calls 64 --output-dir runs/eval_script_new
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 conda run --no-capture-output -n audittrainer python -m audittrainer.cli evaluate --policy base --load-in-4bit --output-dir runs/eval_base_new
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 conda run --no-capture-output -n audittrainer python -m audittrainer.cli evaluate --policy sft --adapter runs/sft_smoke/final_adapter --load-in-4bit --output-dir runs/eval_sft_new
```

使用 `configs/evaluation.yaml`；默认固定 5 个 DEV_ENGINEERING 任务、每 root 24 次模型调用，
script 示例使用 64 次以覆盖已有长修复 fixture。输出目录须为新路径。
`roots.jsonl` 保留所有选中任务，包括无候选、弃权、预算耗尽、协议错误和加载/执行错误。
root 结果保存 `first_program` / `final_program`、可见/私有评分、责任状态、失败原因和全部成本；
`result.json` 提供全 root 分母的汇总。

- `FirstProgramSuccess`：首个完整程序的私有评分成功率，包含构造程序时的子节点调用/修复成本；不等于单次模型调用 Pass@1。
- `BudgetedSuccess`：终止时当前 root 产物的私有评分成功率。预算耗尽或弃权时仍评分其已存在的产物，并单独记录终止原因。
- 在线调用/token/验证/重放、离线评分验证和实际执行用例分别计数；没有成功时 `cost_per_budgeted_success` 为 null，即 N/A。

APPS 导入支持原始 JSONL 的 `id` 或 HF `problem_id`，通过 `fn_name` 区分 stdin 与 call-based，
只导入 stdin。保留原始题目/测试、来源 revision/ID/hash、原始 split 和 root/family；
官方 test 范围仅能分配到 APPS_SEALED。输入/输出按统一 `solve(text) -> str` 包装适配。

```bash
conda run --no-capture-output -n audittrainer python -m audittrainer.cli import-apps --input data/apps_source/train_first100.jsonl --output-dir data/apps_import_new --root-manifest data/manifests/new_roots.jsonl --test-manifest data/manifests/new_tests.jsonl --original-split train --split DEV_ENGINEERING --visible-cases 0 --limit 3 --source https://huggingface.co/datasets/codeparrot/apps/tree/21e74ddf8de1a21436da12e3e653065c5213e9d1
```

root manifest 需已存在（可为空）；重复 root 与跨 split family 会被拒收。
默认 `--visible-cases 1` 将前一例设为在线可见，其余保留离线。当前下载的前 100 条训练记录只有单例，
本次显式设为 0：原始用例全部离线评分，在线无可用 suite 则 UNKNOWN。题面中的公开样例照常保留，
这些离线样例不声称对模型未见。导入报告保留所有未收录记录的原因。

APPS suite 使用单独的 `apps_stdio_legacy` 比较规则，保留上游去空白、近似数字、集合/舍入回退；
后者可能忽略顺序和重复，因此不能解释为严格输出等价。原有 fixture 使用 exact JSON 比较。
比较依据为 [APPS 官方 testing_util.py](https://github.com/hendrycks/apps/blob/main/eval/testing_util.py)。

本次实际数据、评测与验收结果见 [reports/F9.md](reports/F9.md)。

## F8 新 adapter 回接

在 `audittrainer` 环境重新加载公开底座和 F7 保存的 adapter，使用一个明确可见的 GPU：

```bash
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 conda run --no-capture-output -n audittrainer python -m audittrainer.cli rollout --policy sft --adapter runs/sft_smoke/final_adapter --load-in-4bit --split DEV_ENGINEERING --limit 2 --max-policy-calls 24
```

`--policy sft` 必须指定 `--adapter`；`--policy base` 使用公开底座，不接受 adapter。
`--load-in-4bit` 显式启用与 F7 相同的 NF4 双重量化、BF16 计算和冻结参数 dtype；不指定时沿用常规加载。
SFT 加载器核对 base revision、adapter 权重/配置摘要与已完成的训练报告，读取磁盘 tokenizer，
并核对其编码器、特殊 token 与 chat template 的语义一致性。

开始第一个任务前，对同一个初始 observation 执行 adapter 开启、关闭、恢复三次前向。
全部 logits 必须有限，开关之间须有可检测差异，恢复后须与开启时一致，才继续自然 rollout。
诊断保存为 `adapter_probe.json` 与 `adapter_probe_logits.pt`，单独记录 3 次 forward；
其后同一 policy/model 服务所有任务和递归节点，仍使用共享 renderer、action schema、RepairRuntime 与 verifier。
每次模型调用保存 adapter 路径/hash、模型 revision、tokenizer 身份和真实输入输出。

F8 新增两个独立 DEV_ENGINEERING 任务：按首次出现顺序去重、括号配对。
任务与可见用例在运行前固定，family 与 SYN_FIXTURE 训练数据分开。
本次实际运行保存在 `runs/f8_sft_rollout/`：adapter 开关 logits 最大差值 **28.625**，恢复差值 **0**；
共 7 次真实生成，词序去重通过 6 个用例，括号配对在 JSON 格式错误后弃权，未进入功能验证。
两任务责任状态均为 UNKNOWN。详细证据见 [reports/F8.md](reports/F8.md)。

这些结果完成权重回接验收；SMOKE_ONLY adapter 和两个开发任务不用于声称训练带来能力提升。

## F7 QLoRA SFT

在项目根目录使用 `audittrainer` 环境，显式选择一个有足够空闲显存的 GPU：

```bash
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 conda run --no-capture-output -n audittrainer python -m audittrainer.train_sft --config configs/sft_smoke.yaml
```

输出目录须尚不存在；再次运行时在配置中指定新目录。`extends` 相对 YAML 文件解析，
模型、数据与输出路径相对项目根目录。smoke 从固定公开 Qwen3-8B 重新加载 NF4 双重量化底座，
使用 BF16 计算、梯度检查点和 `all-linear` LoRA（r=16、alpha=32、dropout=0.05）。
只有 adapter 参数可训练；每个 microbatch 一条动作，累积 4 次，共执行 10 个 optimizer step。

分词直接读取 F6 保存的 `messages`，采用同一 no-thinking chat template；prompt 全部标记为 -100，
完整 action JSON 和 EOS 参与训练。超长样本整体拒收；右侧 padding 根据长度屏蔽，保留与 pad 同 ID 的真实 EOS。
只计算目标预测位置的 logits，每条样本先按目标 token 平均 loss，再按样本平均。
梯度累积使用当前实际样本数，覆盖最后一个不足量 batch。

前 20% 更新使用 D_IMPL / D_CONTROL = 80% / 20%，其余使用
D_PLAN / D_IMPL / D_COMPOSE / D_CONTROL = 30% / 40% / 25% / 5%。
每次先选 bucket，再均匀选 root，最后选 root 内的动作。抽样有放回且预算有限，固定 seed 并保存完整顺序。
当前 32 条 smoke 数据产生 40 次抽样、19 条不同样本；D_COMPOSE 实际含 5 次 consume 和 6 次父节点 write_code。
默认 TRAIN 配置的抽样总量上限为一个数据集大小。A5 对照组使用显式匹配的 2,240 槽位清单，
以主组预算为共同上限，去重后的控制池虽较小也执行相同更新数；重复曝光次数单独报告。

运行目录保存：

- `resolved_config.yaml`、`run_manifest.json`：固定模型 revision、数据/tokenizer/模板/代码摘要及环境版本。
- `tokenized.jsonl`、`sampling_schedule.jsonl`：实际学习位置与实际抽样顺序。
- `before_update.json`、`training_report.json`：更新前后参数摘要和差值、逐步 loss/梯度/目标 token/LR、重载前向对照。
- `final_adapter/`：adapter、tokenizer 与训练参数。
- `checkpoint-10/`：adapter、tokenizer、optimizer、scheduler、RNG 和 Trainer 状态。
- `reload_probe.pt`：固定输入及保存前 logits，用于核对从磁盘重载的模型。

训练入口只接受单独的 SYN_FIXTURE 或 TRAIN 导出，校验数据摘要与 root/family 分区，
并拒绝 FUNCTION_ONLY 对照混入核心 SFT。32 条 fixture 数据用于参数更新与加载验收；
其 loss 变化不作为能力提升或泛化证据。adapter 回接责任 runtime 的实际验收见 F8。

本次 GPU 1 上实际完成 10 步训练，43,646,976 个 LoRA 参数可训练，平均训练 loss 为 0.20257，
四个跟踪张量全部变化；重新加载底座与 adapter 后，固定输入 logits 最大差值为 0。
已保存的 adapter 位于 `runs/sft_smoke/final_adapter/`，完整证据与测试结果见 [reports/F7.md](reports/F7.md)。

## F6 一步 SFT 数据导出

在 `audittrainer` 环境运行新 fixture suite，然后导出 32 条完整动作：

```bash
conda run --no-capture-output -n audittrainer python -m audittrainer.cli smoke --stage f6 --run-dir runs/fixture_suite
conda run --no-capture-output -n audittrainer python -m audittrainer.cli export-sft --trace-dir runs/fixture_suite --out data/sft/smoke.jsonl --limit 32
```

运行目录和导出文件须使用尚不存在的路径。本次已生成的正式 F6 smoke 输入位于
`data/sft/smoke.jsonl`，其实际轨迹位于 `runs/f6_fixture_suite/`，可直接供 F7 使用。
F6 只加载 tokenizer，不加载模型权重或更新参数。

fixture suite 使用 FX01、FX02，以及新建的 FX05（正确 child、错误 parent）和 FX06
（parent 调用 child 后丢弃结果并重算）。四条轨迹全部经同一 RepairRuntime 与真实隔离 verifier 执行。
`data/fixtures/f6_reviews.json` 是运行前固定的源码/接口责任审查；绑定具体动作和契约，
执行后写入各 run 的 `supervision.jsonl`，不会提供给在线 policy。

导出按事件选择，保留 `observation_before → 一条完整 action`。输入复用 `render_messages` 和 no-thinking 模板，
模型轨迹还核对 F5 保存的实际输入；脚本轨迹必须匹配采集时保存的 renderer 身份。
样本带 root/family/split、mode、bucket、事件来源、契约/计划/artifact 版本、证据引用和监督范围。

- `RESPONSIBILITY_SUPPORTED`：独立责任审查与相应执行证据支持；不把测试 PASS 或 child 被调用当作充分语义证明。
- `PROTOCOL_ONLY`：正确验证、据实提交和有依据的弃权等控制行为；不声称候选代码正确。
- `FUNCTION_ONLY`：只有功能证据。默认不进入主导出；`--allow-function-only` 可显式导出普通功能对照。

错误代码、已知责任违规、UNKNOWN 实现、错版本证据和被拒动作不进入正确代码 target。
失败 root 的正确 child 可以独立保留。未来结果只用于离线筛选，不能补进当时输入；
`evidence_refs`、监督标签和 `meta` 都是离线元数据，不属于训练 prompt/target。

默认 `--max-length 4096` 与 F7 smoke 配置一致，超长样本整体拒收，不截断契约、源码或 target。
输出同时产生 `smoke.rejected.jsonl` 和 `smoke.manifest.json`，记录所有未纳入事件的原因、
输入轨迹及摘要、模型/tokenizer/template 身份、分区和计数。相同 run/event 的拷贝不会重复加权，raw traces 不改写。
仅允许分别导出 SYN_FIXTURE smoke 或 TRAIN；开发、验证、holdout 和 SEALED 分区不能混入 SFT。

当前数据：32 条样本，PLAN 12 / IMPLEMENT 6 / COMPOSE 4 / CONTROL 10。
完整验证结果、拒收统计及 F5 模型轨迹的功能对照导出见 [reports/F6.md](reports/F6.md)。

## F5 真实模型 rollout

`rollout --policy base` 加载本地公开 Qwen3-8B，一份权重顺序服务全部 root 和递归节点。
当前 manifest 登记六个 SYN_FIXTURE 与五个 DEV_ENGINEERING 任务；SYN_FIXTURE 前两个为整数求和与区间求和任务。
使用 Conda `audittrainer`，选择有足够可用显存的 GPU：

```bash
CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=4 conda run --no-capture-output -n audittrainer python -m audittrainer.cli rollout --policy base --split SYN_FIXTURE --limit 2 --max-policy-calls 24
```

也可显式使用 CPU，加载同一份完整 BF16 权重：

```bash
OMP_NUM_THREADS=16 conda run --no-capture-output -n audittrainer python -m audittrainer.cli rollout --policy base --split SYN_FIXTURE --limit 2 --max-policy-calls 24 --device cpu
```

`--run-dir` 可指定新目录，`--config` 指定配置；`--policy sft --adapter` 加载匹配底座的本地 PEFT adapter。
F5 使用 base；F7 产出的新 adapter 已在 F8 验收 runtime 回接。`--max-policy-calls` 是每个 root 共享的总调用上限，
覆盖递归、错误重试和修复；覆盖后的实际配置会保存。未指定时使用 framework.yaml 的 64 次。

`prompts.render_messages` 从同一份决策前 observation 构造输入，供后续 SFT 复用。
系统协议固定，当前可用 action 的 JSON Schema 从执行 schema 生成；
Qwen 模板固定 `enable_thinking=False`。不读取 fixture script/reference 作为模型答案，不做 JSON 提取或自动补写。
非法或截断输出由 host 拒绝；下一次 observation 携带结构化反馈，最多重试一次。

每次调用在 8192 上下文中，为所有当前允许动作的**最大输出配额**预留空间。
先按固定规则省略重复证据绑定/重复内容；仍放不下则保存 `CONTEXT_OVERFLOW`。
完整契约、当前源码、失败用例和 ticket 不截断；实际 prompt 与精确 token ID 都保留。
root 剩余 token 不足时在生成前终止，已消耗用量不会返还。

运行目录保存 `request.json`、`model_identity.json`、batch `result.json`，
以及每个 root 的源码、证据、状态和 `trace.jsonl`。POLICY 事件新增 `model_input` 与 `output_token_ids`，
包括真实 messages、no-thinking prefix、压缩方式、生成设置和 token 用量；终止 EOS 单独保留在 token ID 中。
host 仍以严格 schema 重解析原始响应，所有生成代码通过原有 Bubblewrap 执行器检查。

模型失败、弃权、协议预算终止同样产生结果。rollout 的退出码 0 表示真实生成后的执行已完成，
不代表所有 root 为 PASS；模型加载、模型执行或验证基础设施错误返回非零。Scripted smoke 保持原有 PASS 门禁。
F5 是接线验收，功能状态、责任状态、模型来源和测试范围分别报告。
本机实际验收使用 CPU：FX01 为 PASS，FX02 经真实失败、修复拒绝和反馈后弃权，保留 FAIL。
共 10 次真实模型调用；完整结果与 GPU 加载情况见 [reports/F5.md](reports/F5.md)。

## F4 有界修复与计划修订 smoke

```bash
conda run --no-capture-output -n audittrainer python -m audittrainer.cli smoke --stage f4
conda run --no-capture-output -n audittrainer python -m audittrainer.cli smoke --stage f4 --fixture data/fixtures/fx04_plan_revision.json
conda run --no-capture-output -n audittrainer python -m pytest -q
```

默认 FX03 在 root 中注入答案顺序错误：先产生真实 FAIL，再执行
`localize → ticket → repair → 可见重放 → finish(v2)`。补丁仅修改 root，三个后代的源码保持不变。
FX04 则执行 `localize(CONTRACT, PLAN_SUBTREE) → plan(v2) → 重建子树 → 验证 → finish`，
新 child 使用 `N0_p2_*` 身份，重新生成、验证和显式消费，incoming contract 保持不变。

`repair.py` 提供模块/函数所有者、artifact/契约/consumer、证据目标三个索引，最多给出三个责任候选。
localize 必须引用当前可见证据；先无修改重放检查可复现性，再生成 ticket。
ticket 分别声明 `editable_files`、`editable_plan_nodes`、`rebind_edges`、`revalidate_nodes` 和 `frozen_hashes`。
代码替换的 `file` 使用 ticket 中的逻辑路径，例如 `nodes/N0.py`；`base_hash` 必须匹配当前版本。

候选在 `repairs/T*/attempt_*/` 中执行，活跃目录保留原程序。
只有修复目标及受影响节点的固定可见检查通过、完整 root 的全部固定可见 suite 通过，且冻结补集保持不变，
候选才被提交。额外 suite 的回归也会导致回滚；UNKNOWN 不算通过，私有检查不参与补丁选择。
接受后的证据绑定新版本，finish 直接使用它，不重复验证。

依赖更新由 `repair.dependency_updates` 明确指定，且只能使用 ticket 授权的边。
`EXISTING` 选择已存在的版本/hash，`CANDIDATE` 选择本次事务生成的节点版本；host 不自动更新旧消费决定。
因此一个 frozen consumer 可以按授权仅更新绑定元数据，其源码 hash 仍须保持不变。
child 局部修复也已接通：`request_child_repair` 后 child 自己 localize/repair，再返回新版本供 parent 消费。

源码补丁与计划重建共享 root 的 **3 次尝试**，失败及拒绝的补丁也计数；每节点最多修订一次计划。
自动重放使用同一验证预算；局部尝试失败后至多扩大一次父边界。
计划重建会暂存新子树，验证失败、弃权或预算耗尽时恢复旧状态，已消耗的调用和节点创建预算不会返还。
已形成的外部 consumer 必须纳入计划边界；子树重建期间不嵌套开启第二个修复事务。

运行目录保留 ticket、源码/计划快照、可见 suite 与 runner 身份、无修改重放、候选版本、接受/回滚记录，
完整动作仍写入同一 `trace.jsonl`。`result.json` 的 `repair_history` 给出结果与候选目录。
实际验收见 [reports/F4.md](reports/F4.md)。

## F3 递归与显式消费 smoke

```bash
conda run --no-capture-output -n audittrainer python -m audittrainer.cli smoke --stage f3
conda run --no-capture-output -n audittrainer python -m pytest -q
```

默认读取 `data/fixtures/fx02_range_sums.json`：root 解析 JSON 并组合前缀和与区间查询，
查询节点进一步委托孙节点计算端点差。共 4 个节点、19 次脚本决策、4 次验证调用、12 个用例。
所有层级复用同一个 Policy、root 预算、verifier 与 trace；调度采用顺序深度优先。

```text
N0 solve
├── N0_p1_prefix prefix_sums
└── N0_p1_query range_sums
    └── N0_p1_query_p1_difference range_sum
```

`plan(SPLIT)` 登记完整子契约，host 根据 parent、plan version 和 local_key 分配 ID。
child 结束后 parent 进入 `RESOLVE_CHILD` 并再次调用 policy，选择 `consume`、`reject`、
`request_child_repair` 或 `abstain`。全部 child 处理后 parent 才进入 IMPLEMENT。
达到深度、节点或 child 数量预算时仍可 ATOMIC 或弃权；不会偷偷替模型改写计划。

`consume` 记录 child 的当前版本、hash、alias 与可见证据。FAIL/UNKNOWN 需要显式
`accept_unverified=true`，已知失败另记 `known_failed_consumed`；基础设施 ERROR 不可消费。
`write_code.declared_dependencies` 必须与消费记录一致。源码使用明确的导入格式，例如：

```python
from nodes.N0_p1_prefix import prefix_sums as prefix
from nodes.N0_p1_query import range_sums as query
```

parent 自己编写参数传递和业务组合。host 仅物化选定版本的依赖闭包和固定 I/O wrapper，
保存到 `workspaces/<node_id>/v<version>/`；不会根据计划添加调用或补写算法。
静态检查和沙箱导入检查约束直接 child 依赖；兄弟节点通过 parent 传值。

证据中的 `dependency_usage` 分别记录 `selected`、`linked`、`called` 和调用次数。
消费但未导入/未调用不会算实际调用；调用后丢弃返回值也不会获得消费语义证明，
`dependency_supported` 保持 UNKNOWN。调用观测不完整（如超时）时，缺失调用不会直接记为 false。
fingerprint 覆盖完整传递依赖；后代发布新版本会使旧 consumer 证据失效，历史源码和证据保留。

`reject` 保留委托义务与未完成 child，即使 parent 自行重做后功能通过，仍记录责任不完整。
`request_child_repair` 保留契约和 artifact，将 child 重新激活到 REVIEW，传入请求和可见证据，
每条边最多请求一次。F3 入口可重新验证、提交或弃权；F4 入口进一步开放 localize/ticket/repair。

每次运行创建 `runs/f3_<UTC时间>/`，包含逐节点源码版本、工作区、`consumptions.jsonl`、
证据、25 条事件与各节点的 NodeResult。验收报告见 [reports/F3.md](reports/F3.md)。

## F2 原子完成 smoke

在仓库根目录使用 `audittrainer` 环境执行，无需 GPU：

```bash
conda run --no-capture-output -n audittrainer python -m audittrainer.cli smoke --stage f2
conda run --no-capture-output -n audittrainer python -m pytest -q
```

默认运行 FX01：`plan(ATOMIC) → write_code → verify → finish`。6 个手工审查的求和用例
通过后，节点返回 `NodeResult`，`functional_status=PASS`，责任状态仍为 `UNKNOWN`。
`finish` 只提交当前版本及宿主证据；失败或无法判定的候选同样可以提交，未生成代码时可 `abstain`。

运行产物位于 `runs/f2_<UTC时间>/`：源码、计划、状态快照、动作日志、
`evidence/E0001.json` 和 `result.json`。支持 `--fixture`、`--config`、`--run-dir`；
指定的运行目录必须尚不存在。CLI 成功退出要求 F2 提交为 PASS。

执行依赖 Linux 用户命名空间、Bubblewrap 和系统 `libseccomp.so.2`；本机验证版本为
Bubblewrap 0.4.0。沙箱提供当前 `audittrainer` Python 3.11 标准库、只读候选源码与单个测试输入，
隔离网络、进程视图和宿主目录，清空继承环境，不提供 GPU 或第三方 Python 包。
临时目录可写；禁止创建子进程/线程，限制每例 2 秒、进程地址空间 1024 MiB、
stdout/stderr 合计 1 MiB，同时设置 CPU、单文件大小和文件描述符上限。
隔离设施无法启动时返回 ERROR，没有宿主直接执行的回退路径。

`Suite` 由宿主登记，绑定 root、完整审查契约、契约版本、覆盖义务和 oracle 来源。
支持 `solve(text)` 和 JSON 参数/返回值的同步 callable；此阶段使用精确 JSON 比较。
找不到匹配的可见 suite 时，observation 提供 `NO_APPLICABLE_SUITE`，验证产生 UNKNOWN；
空 suite 或义务覆盖不完整也不会通过。模型不能自行指定可信 oracle 路径或调用私有 suite。
预期答案留在宿主比较，可见检查完成后其反馈可进入 observation；私有检查仅支持显式宿主离线调用。

证据 fingerprint 绑定契约、源码、依赖、suite、runner 与执行配置。
过期证据保留历史记录，不能用于 `finish` 或当前候选的 PASS。用量记录验证调用、
测试调用次数（含启动失败的尝试）与执行耗时；`verification_performed` 表示已产生宿主检查证据，
因此 UNKNOWN 的空 suite 可为 true，但 `executed_cases=0`。
实际结果与适用范围见 [reports/F2.md](reports/F2.md)。

## F1 原子 smoke

在仓库根目录使用 `audittrainer` 环境执行，无需 GPU：

```bash
conda run --no-capture-output -n audittrainer python -m audittrainer.cli smoke --stage f1
conda run --no-capture-output -n audittrainer python -m pytest -q
```

默认读取 `data/fixtures/fx01_integer_sum.json` 和 `configs/framework.yaml`。
每次 smoke 新建 `runs/f1_<UTC时间>/`，也可用 `--run-dir` 指定一个尚不存在的目录。
其中保存 `trace.jsonl`、决策前后 `states/`、host 登记的 `plans/`、
`artifacts/N0/v1.py` 及版本元数据，最终摘要为 `result.json`。

F1 定义全部 10 类 action schema，执行 `plan(ATOMIC)` 与首次 `write_code`；
源码经过编译检查后保存，节点停在待验证状态。`functional_status` 为 `null`，
`verification_performed` 为 `false`。脚本调用 token 用量为 0，明确标记 scripted 来源。
协议错误保留原始输出和反馈；调用、输入/输出 token 及连续重试共用 root 预算。
实际阶段报告见 [reports/F1.md](reports/F1.md)。

## 环境

使用 Python 3.11.14，环境统一为名为 `audittrainer` 的独立 Conda 环境。
从仓库根目录执行：

```bash
conda activate audittrainer
python -c "import audittrainer; print(audittrainer.__version__)"
```

在另一台 Linux x86_64 / CUDA 12.8 兼容机器上重建：

```bash
conda create -n audittrainer python=3.11.14 pip -y
conda activate audittrainer
python -m pip install -r requirements.lock
python -m pip install --no-build-isolation --no-deps -e .
```

锁文件记录实际验证环境的全部 Python 依赖；`pyproject.toml` 定义直接依赖。
实际版本为 PyTorch 2.10.0（CUDA 12.8）、Transformers 4.57.6、PEFT 0.18.1、
Accelerate 1.14.0、bitsandbytes 0.48.2。Torch 和 bitsandbytes 的锁条目保留
公开 wheel URL 与 SHA256，以固定本次验证的具体构建。
GPU 通过调用时的 `CUDA_VISIBLE_DEVICES` 选择；单卡运行时，程序内设备为 `cuda:0`。
硬件与轻量检查结果记录在 `data/manifests/environment.json`。

本机 A0 检查选择物理 GPU 3。按当前设备占用调整后，可复查环境：

```bash
CUDA_VISIBLE_DEVICES=3 conda run -n audittrainer python scripts/check_a0.py
```

此命令检查真实训练 API 导入、tokenizer 与 non-thinking 前缀、BF16 矩阵运算、
NF4 双重量化层前向和依赖兼容性，成功后更新环境记录；不加载 8B 权重或更新参数。

## 模型与配置

公开模型：`Qwen/Qwen3-8B`，revision 固定为 `b968826d9c46dd6066d109eabc6255188de91218`。
当前本地权重目录为 `Qwen3-8B/`，不纳入 Git。文件身份与 tokenizer/chat-template hash
记录在 `data/manifests/model.json`。F5 在运行目录记录实际加载身份，参数更新属于 F7。

`configs/framework.yaml` 定义共享递归协议、预算、执行和修复设置。
`configs/sft_main.yaml` 与 `sft_smoke.yaml` 定义正式训练及短程接线参数；
`configs/evaluation.yaml` 定义初始开发评测口径。
`configs/mechanism.yaml` 定义 A2 的固定开发池、比较条件、解码与总预算。
这些是起始规格，消费这些配置的业务入口随对应 F 阶段实现；不把示例命令当成已完成接口。
路径相对仓库根目录，smoke 的 `extends` 相对所在配置文件目录。

## 数据

八个分区及使用规则见 `data/manifests/splits.yaml`；`roots.jsonl` 当前登记六个
`SYN_FIXTURE`：整数求和 FX01、递归区间求和 FX02、代码修复 FX03、计划修订 FX04、
失败 parent FX05、绕过 child FX06，均保留来源 hash、原分区和 I/O 类型。
FX03–FX06 派生自 FX02，保持相同 family 和分区。正式数据导入在后续阶段实现。
另有 F8/F9 的五个 `DEV_ENGINEERING` root，以及 A2 的二十个 `DEV_MECHANISM` root；
A2 共用一个预先保留的组合任务 family，详见 [开发池说明](data/mechanism/README.md)。
按 root/family 分区，全部派生动作和修复继承分区。正式训练不使用 fixture 或开发题。
训练与推理 observation 只包含决策前可见信息，SEALED 私有测试只评分选定输出。

## 技术依据

- [Qwen3-8B 官方模型卡](https://huggingface.co/Qwen/Qwen3-8B)：模型与 non-thinking 模板。
- [PyTorch 安装版本](https://pytorch.org/get-started/previous-versions/)：CUDA 构建。
- [PEFT 量化训练](https://huggingface.co/docs/peft/en/developer_guides/quantization)：QLoRA 接口。

- [Bubblewrap 官方说明](https://github.com/containers/bubblewrap)：命名空间、挂载和权限隔离。
- [Python resource 文档](https://docs.python.org/3.11/library/resource.html)：进程资源上限。

A0 的 CUDA/量化检查与 F2 候选执行相互独立；F5 模型生成的代码仍使用 F2 隔离执行器。
