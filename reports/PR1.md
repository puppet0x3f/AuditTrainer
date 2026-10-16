# PR1：共同格式修订、任务条件与根规划 RL 对照

状态：**COMPLETE（2026-09-12，单批次 PR1 pilot）**。
已按共同 SFT → 候选执行 → B0/B3/B4 对照的顺序完成：25 次格式 SFT 更新、80 次候选续跑、
10 次独立自然基线、B3/B4 各 10 次真实 RL 更新，以及 54 次自然评估。

**共同补训改善了已检查状态的计划格式支持，但本轮没有观察到 RL 带来的任务完成率提升。**
只有 1/10 个训练状态提供回报差异，且来自数据流合法性；已注册的 SPLIT 之间没有可区分的价值信号。
这轮实验证明采样、双层信用、实际参数更新及评测已接通，尚不支持细粒度分解收益的科学结论。

| 方法 | RL 更新 | TRAIN 自然通过 | VALIDATION 自然通过 |
| --- | ---: | ---: | ---: |
| B0：共同格式 SFT 起点 | 0 | 0/10 | 0/8 |
| B3：统一长期信用 | 10 | 0/10 | 0/8 |
| B4：模式/正文双层信用 | 10 | 0/10 | 0/8 |

VALIDATION 为 4 个根任务各 2 次运行，来自 2 个未参与训练的任务族。首个完整程序与终局的通过数相同，
全部失败、弃权和没有程序的运行均计入分母；54 条运行均正常结束并留有评分，基础设施错误为 0。
完整机器汇总及最终检查：[result.json](../runs/refinement/PR1/result.json)。

## 已固定的实验

- 共同起点：PR0 适配后的 planner，已完成一次 200 条 TRAIN 示范、25 次更新的格式 SFT（924.37 秒）。
  根状态 ATOMIC/SPLIT 各 100 条；来自 150 条不同既有示范，ATOMIC 明确重复两次。
  正文均值 CE 额外加上前 8 个 token 的均值 CE（系数 1），模式 CE 不变。
- 格式检查：原 PR0 每族首个根状态，共 10 根、30 个原始 SPLIT 候选；前后使用相同 observation 和种子。
  旧数据中 111 个解析错误含 51 个开头重复逗号、8 个开头缺引号；完整训练前缀和目标的解码拼接检查通过。
  同状态可解析候选从 6/30 提高到 22/30，复测平均 P(SPLIT)=0.38052；这项改善仅在该旧任务样本上验证。
- TRAIN：8 个预先固定的组合任务（FIFO 清算、分段工时计费、版本价格关联、带禁用区间的依赖调度）及 2 个原有简单任务。
  全部保留，不根据 rollout 成败挑选。新任务说明公开可选子契约，供使用已有局部检查；不暴露实现和私有用例。
- VALIDATION：4 个根任务，来自独立的预留/发货状态机与分支文本修订两个任务族；仅做最终评测。
- 候选：每根 1 ATOMIC、3 个独立原始 SPLIT，各 2 次完整续跑，共 80 次。
  另采集每根一个独立自然后缀，共 10 次，作为 B3 的状态基线；两组共同计入该采样开销。
- 更新：B3/B4 从共同 checkpoint 开始，在同一批原始候选上各运行 2 个 epoch；每 2 个状态累积一次梯度，共各 10 次更新。
  学习率 1e-6、clip 0.2、KL 0.02、SFT 保持 0.02；整个目标共同乘 1/256，正文仍按 token 求和。
  B3 对每个 SPLIT 的统一优势分别裁剪再平均；B4 分别使用模式优势和正文留一优势。
  KL 为二元模式 KL 和旧采样前缀上的全词表正文 KL。
- 执行器始终为冻结 F01；修复关闭。候选执行时后续规划为冻结 B0；训练只更新根规划 LoRA。
- 本轮沿用诊断奖励 `R = (1-v)y - 0.2v`，成本系数为 0；没有发现明确责任违规，实际反馈为终局通过与否。
  所有成本另行记录，本轮不声称优化了成本加权效用，也不以分解数量或深度提供奖励。
- 正常评测：每种方法各 10 条 TRAIN、8 条 VALIDATION 自然轨迹，共 54 次；无候选搜索，种子一致。
  新 planner 在各递归层共享使用；本轮仅在根状态训练，尚未对中间节点做 RL 更新。
  VALIDATION 不生成训练 reward，不进入 optimizer。

配置：[refinement_support.yaml](../configs/refinement_support.yaml)、[refinement_pr1.yaml](../configs/refinement_pr1.yaml)。
任务：[selection.json](../data/refinement/pr1/selection.json)。格式诊断：[format_diagnosis.json](../runs/refinement/PR1_support/format_diagnosis.json)。

## 执行记录

共同 SFT 首次在 GPU 3 的权重加载阶段被 SIGKILL，没有参数更新；日志和请求保留在 PR1_support。
改用 GPU 0，参数和训练计划不变。SFT 已完成，F01 参数及同输入前向不变，保存重载的 logits 和 mode log-prob 差异均为 0。
新旧概率、裁剪、格式监督、任务 oracle 边界与数据分区的 23 项定向检查及 284 项全量回归通过。

早期新任务采集在完成 9 条草稿续跑后停止：任务说明的行模板没有明确标示列表层次，部分局部接口未精确定义中间结果形状。
已明确输入为多行数组、局部输出的字典键及每行字段，修订后的 11 项定向检查通过，最长根 prompt 为 3,359 token。
任务族、参数、测试用例及种子不变；所有 TRAIN 根均保留。草稿任务、原始候选、完整/未完成轨迹与日志保存在
`runs/refinement/PR1/input_contract_draft/`，不计入正式的 80 次续跑，也不进入 RL。VALIDATION 当时尚未运行。

正式候选采集已完成 80 次，独立自然基线 10 次。ATOMIC 4/20 通过，SPLIT 8/60 通过；
SPLIT 解析有效 19/30、初始注册 13/30，8 次成功均来自初始已注册计划。
8 个新增复杂任务的 64 次候选续跑均未通过。账本的四个候选 Q 均为 1；库存 Q 为 [1,0,0,1]。
因此仅 1/10 个状态有模式或正文价值差异。库存中两个失败计划被 INVALID_DATAFLOW 拒绝，
当前没有两个初始已注册 SPLIT 之间的价值差异；不能将此次信号称为已发现更优合法分解粒度。
B3/B4 均已完成实际更新，各 10 次 AdamW step，43,646,976 个规划 LoRA 参数可训练，跟踪参数有非零变化。
两组的冻结 B0 参考及 F01 执行器完整参数指纹前后相同；更新前评分路径的 log-prob 差异为 0，
磁盘 adapter 重载后的固定输入 logits 差异为 0。正常评测已在独立新进程中加载这些已保存的权重并全部完成。
最终聚合核对了 80 次候选续跑、10 次独立基线、20 次 RL 更新、54 条自然评测及 6 个冻结 worker，全部通过。

真实 NF4 模型概率检查发现：缓存生成与完整序列 teacher forcing 在少数 token 上存在 BF16 数值差异，
三个正文最大 log-prob 差异分别为 0.2471、0.3693、0.1751，前两个各有一个 token 会在参数未更新时触发 0.2 裁剪。
模式评分差异为 0。记录见 `runs/refinement/PR1/probability_check.json`。
因此本轮单批更新以冻结 B0 在相同完整序列评分路径下重算的概率作为正文比率分母；它同时是本轮行为策略和 KL 参考。
原始 token、缓存生成概率和回报全部保留；重算概率另存，各方法采用相同数值约定。

## 实际训练信号

两组共享 40 个原始候选、80 次续跑、10 次独立基线以及相同保留示范和更新顺序。
本轮只有库存状态具有非零回报优势，其他 9 个状态仍参与 KL/SFT 保持，不能把 10 次 optimizer 更新称为 10 个有效奖励状态。

| 库存状态 | B3：统一信用 | B4：模式/正文双层信用 |
| --- | --- | --- |
| 模式优势（ATOMIC，SPLIT） | `(0, -0.666667)` | `(0.251694, -0.414973)` |
| 三个 SPLIT 正文优势 | `(-1, -1, 0)` | `(-0.5, -0.5, 1)` |
| 实际 optimizer 更新 | 10 | 10 |
| 更新前新旧评分 log-prob 最大差异 | 0 | 0 |
| 保存重载 logits 最大差异 | 0 | 0 |

该状态的 ATOMIC Q 为 1，三个 SPLIT Q 为 `(0, 0, 1)`，旧策略 P(SPLIT)=0.377541，独立自然基线为 1。
前两个 SPLIT 被初始 `INVALID_DATAFLOW` 检查拒绝，第三个成功注册。
因此这里验证了双层信用的实际计算与更新，但还没有提供合法责任分配之间的粒度优化证据。
训练记录见 [B3](../runs/refinement/PR1/B3/training_report.json) 与 [B4](../runs/refinement/PR1/B4/training_report.json)。

新增复杂任务的 64 次终局检查中，44 次为程序异常、10 次输出错误、9 次没有根程序、1 次超时。
这是终局检查记录的首个失败类别，不能据此给每条轨迹作唯一因果归因。
已有具体异常包括把明确的发票行列表当作字典调用 `.keys()`，以及将四字段工时记录解包为三个变量。
这些记录表明新任务上的基本接口理解和执行仍限制终局奖励，不能仅用增加 RL 更新数解决信号不足。

两组训练进程耗时分别为 441.35 秒和 282.84 秒，运行期间各 GPU 上的其他工作负载不同，不能据此声称算法加速。
评测首次加载 B3/B4 时发现训练报告缺少既有加载器要求的 `adapter_config_sha256` 字段；
已补全报告输出与实际配置摘要，并通过正式加载器校验。失败发生在权重加载和采样之前，参数及评测种子未变；
原始错误日志保留为 `evaluate_B3_metadata_missing.log` 和 `evaluate_B4_metadata_missing.log`。

## 自然评估与实际成本

候选执行的 ATOMIC 4/20、SPLIT 8/60 是分层采样的结果，不能替代自然策略完成率。
独立基线为 2/10，使用事先固定的另一组种子，只用于训练基线；最终对照统一使用评测种子，B0 为 0/10。
本轮没有从这两组运行中挑选较好的结果。简单任务的候选成功也未稳定转化为模型自行选择计划时的成功。

以下为记录的模型输入/输出 token、生成调用及模式评分批次；模式评分不冒充生成调用。

| 工作负载 | 输入 token | 输出 token | 生成调用 | 模式评分批次 |
| --- | ---: | ---: | ---: | ---: |
| 共同格式复测：30 个 SPLIT | 97,110 | 8,091 | 30 | 10 |
| 共同候选采样 + 80 次续跑 + 10 次基线 | 1,548,710 | 66,733 | 271 | 100 |
| B0：18 条自然评估 | 600,890 | 21,598 | 98 | 34 |
| B3：18 条自然评估 | 598,611 | 23,623 | 97 | 36 |
| B4：18 条自然评估 | 540,832 | 19,172 | 87 | 36 |

共同采样仅实际执行一次，在 B3/B4 方法成本比较时分别计入同一份开销。
B4 的自然运行使用较少 token，但三组成功率均为零，这不能作为提高完成效率的证据。
训练前后向、概率诊断、加载失败与停止的草稿不在此表内；训练步数及耗时、独立诊断和草稿日志均分别保留，
因此此表不称作整个实验的总计算量。离线评分调用、执行用例数和各轨迹预算开销保存在机器汇总中。

本轮到此结束。后续进入中间节点 RL 前，优先建立固定执行器能够完成、且合法分解之间出现终局价值差异的 TRAIN 条件；
现有结果更支持先解决任务难度与执行能力的匹配，而非直接扩大本批数据的更新次数。

## 复现入口

在 `audittrainer` 环境执行。新训练应将配置中的输出路径及相应 adapter 路径一起改为新位置，保留本轮结果；
任务包直接使用已固定的 `data/refinement/pr1/`。以下命令按阶段顺序运行，B3/B4 均从相同 B0 初始化。

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python -m audittrainer.train_refinement --config configs/refinement_support.yaml
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python -m audittrainer.refinement_pr1 support
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python -m audittrainer.refinement_pr1 collect
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python -m audittrainer.refinement_pr1 train --method B3
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python -m audittrainer.refinement_pr1 train --method B4
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python -m audittrainer.refinement_pr1 evaluate --method B0
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python -m audittrainer.refinement_pr1 evaluate --method B3
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python -m audittrainer.refinement_pr1 evaluate --method B4
python -m audittrainer.refinement_pr1 report
```

正式采集采用 `--shards 2 --shard-index 0/1` 分配至两个进程，最终评测每个方法一个进程；
上面的顺序命令覆盖同一批任务和固定种子。最后一次全量回归为 **284 passed**，
日志见 [pytest_final.log](../runs/refinement/PR1/pytest_final.log)。
