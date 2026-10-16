# AuditTrainer PPT 逐页讲解稿

对应 25 页 PPT：第 1–20 页为正文，第 21–25 页为备份。建议正文讲解约 20 分钟；短汇报可略讲第 5、9、10、12 页。

每页讲解已写入 PPT 演讲者备注。图中均保留原始分子/分母；无显著性或最终泛化声明。

## 第 01 页｜AuditTrainer

开场：目前已完成从运行框架到核心训练与开发评测的闭环，阶段推进到 A6。研究围绕责任分解、契约履行与组合、有界修复展开。
最清楚的结果来自 A5：F01 相对普通功能 SFT 对照 F00，固定组件通过数从 0/16 提高到 13/16；双子模块组合从 1/8 提高到 8/8。随后改变父子职责分配，暴露出父节点重新承担汇总职责时的不足。
今天汇报重点是实验为什么这样设计、这些结果支持什么判断，以及下一步如何补齐结构对修复的价值证据。所有结果截至 2026-09-09，属于单 seed 开发比较。

证据：[STATUS.md](/home/lc/AuditTrainer/STATUS.md)、[reports/A5.md](/home/lc/AuditTrainer/reports/A5.md)、[reports/A6.md](/home/lc/AuditTrainer/reports/A6.md)

## 第 02 页｜目前最清楚的成果：局部履约与组合可被训练改善

目的：先让导师建立对当前进展的整体判断。这里把工程完成、科学正向结果和能力瓶颈分开表达。
结果：50 个 TRAIN 根任务形成 2,240 条训练动作；四个 A5 模型加一个 A6 修订模型均正式训练 140 步并回接运行时。F01 对比 F00 的组件和双子模块结果是当前最明确的正向证据。
解释：履约训练让模型更稳定地生成符合局部接口的代码，并把固定组件串联起来。改变父节点职责后成功下降，说明常见组合行为尚未形成稳定的职责适应能力。
边界：0/16→13/16 是两种训练组之间的开发对照，F00 也经过 SFT；不能写成公开底座直接提升到 81.25%，也不是多 seed 显著性结论。

证据：[reports/A4.md](/home/lc/AuditTrainer/reports/A4.md)、[runs/a5/result.json](/home/lc/AuditTrainer/runs/a5/result.json)、[runs/a6/result.json](/home/lc/AuditTrainer/runs/a6/result.json)

## 第 03 页｜责任是否真正改变了模型的实现行为？

目的：解释为什么要单独测“责任”，以及为什么职责重分配是关键实验。
设计：双子模块条件中，transform 负责变换，summarize 负责汇总，父节点负责解析、数据传递与格式化。单子模块条件保留同一个 transform 子契约，把汇总职责合法地留给父节点；根任务与要求的最终结果不变。
解释：父节点需要根据当前 obligation map 和 retained responsibilities 改变代码。若只是复现训练中的传递模式，就可能直接返回中间列表、遗漏汇总或沿用错误输出包装。
另一个区别是父节点可能调用 child 后丢弃返回值，自己重算。调用记录只能证明执行过，不能单独证明遵守了责任分配。A4 用受控对照明确展示这种差异。
本页为机制示意，不是额外实验结果。

证据：[reports/A4.md](/home/lc/AuditTrainer/reports/A4.md)、[reports/A5.md](/home/lc/AuditTrainer/reports/A5.md)、[src/audittrainer/schema.py](/home/lc/AuditTrainer/src/audittrainer/schema.py)

## 第 04 页｜一个共享策略，贯通计划、执行、证据与修复

目的：说明已经实现的系统如何支持三条研究主线。
流程：模型先输出 ATOMIC 或 SPLIT 计划，约定子契约、义务归属和父节点保留责任。子节点递归执行，父节点通过 consume 选择精确版本，再由模型自己编写组合代码。可见测试返回与当前契约和产物绑定的证据。
修复：R1 责任筛选提供候选，模型输出 localize 和 repair；ticket 定义真实可编辑边界，边界外源码冻结。支持代码补丁和内部计划子树重建，以可见检查全过且无已知回归为接受条件，否则回滚。
主配置：深度从根 0 到最大 2，最多 7 个累计节点、每节点 3 个 child，根共享 64 次策略调用、32 次验证/重放、3 次补丁。框架支持递归与修复，并不代表当前训练模型已经主动使用这些能力。
私有评分只在程序快照选定、模型决策结束后执行。

证据：[configs/frozen_mechanism.yaml](/home/lc/AuditTrainer/configs/frozen_mechanism.yaml)、[reports/F3.md](/home/lc/AuditTrainer/reports/F3.md)、[reports/F4.md](/home/lc/AuditTrainer/reports/F4.md)

## 第 05 页｜每个阶段回答一个明确的问题

目的：把后续结果放回研究逻辑，避免仅按工程阶段流水介绍。
A1 验证运行、训练和回接能否真实执行；A2 用 20 题比较责任粒度、承诺时机和修复候选机制；A3 根据 A2 冻结唯一主方案。
A4 构造 50 个训练任务及可追踪示范、对照，解决监督来源问题；A5 在同样训练预算下做四组两因素比较，并拆开测局部计划、组件、组合和自由运行；A6 针对父职责覆盖做一次有限干预。
下一步 A7 固定失败和修复器，检验责任结构的下游价值。160、180、45 都是条件运行次数，不是独立新任务数，A5/A6 的开发任务重复使用。

证据：[STATUS.md](/home/lc/AuditTrainer/STATUS.md)、[reports/A2.md](/home/lc/AuditTrainer/reports/A2.md)、[reports/A5.md](/home/lc/AuditTrainer/reports/A5.md)、[reports/A6.md](/home/lc/AuditTrainer/reports/A6.md)

## 第 06 页｜局部组件可通过，完整组合仍会失败

目的：比较责任表示粒度，判断函数/模块级结构能否作为后续学习载体。
设计：coarse 在一个 solve 函数里完成任务；functions 提供函数接口但不要求父义务映射；hierarchical 提供完整子契约、义务分配与显式消费。模型负责代码、非平凡计划和消费，共用固定执行控制。
结果：三组私有成功分别 8/20、2/20、1/20。functions 与 hierarchical 的局部检查都达到 29/40，但分别有 7/20 和 9/20 个任务在两个 child 都通过后仍未形成根程序。
解释：完整运行瓶颈集中在消费、组合和协议履行，不能把低根成功率简单归为所有子模块算法都不会。此结果支持优先补履约与消费数据。
边界：functions 仍生成了不符合条件的义务映射，严格遵守率为 0/20；hierarchical 为 20/20。该实验改变表示、信息和交互结构，不能估计单字段因果效应，也没有显示层次契约的零样本生成优势。

证据：[reports/A2.md](/home/lc/AuditTrainer/reports/A2.md)、[runs/a2/mechanism_p1.json](/home/lc/AuditTrainer/runs/a2/mechanism_p1.json)

## 第 07 页｜来源链接完整，不能代替功能或责任正确

目的：检验生成后补链接与生成期承诺之间的差异，避免把“有责任标注”直接当作履约。
设计：posthoc 先冻结完整代码，再给实际函数补来源链接；static 先声明 helper 和义务关系，再生成单模块代码；hierarchical 逐节点提交契约、实现和消费。hierarchical 与 P1 共享同一批轨迹，统计总运行数时不重复计数。
结果：功能通过依次为 9/20、6/20、1/20；来源链接完整为 1/20、20/20、2/20。静态计划即使链接完整，功能通过也只有 6/20。
解释：链接完整仅说明义务指向实际源码对象，并不验证算法正确、分工语义或父模块使用行为。当前数据未支持逐节点承诺的零样本生成优势；保留这一表示用于后续训练检验。
边界：static 对预定 helper/映射的严格遵守为 0/20，因此不是完全实施的纯静态机制对照。不能从柱状图得出责任标签无用，也不能把静态链接 20/20 称为责任正确 20/20。

证据：[reports/A2.md](/home/lc/AuditTrainer/reports/A2.md)、[runs/a2/mechanism_p2.json](/home/lc/AuditTrainer/runs/a2/mechanism_p2.json)

## 第 08 页｜责任筛选在共同故障上取得较好的修复结果

目的：在共同失败起点下比较日志定位、责任筛选和有限候选重放。
设计：R0 读取日志与源码；R1 增加责任索引和由可见失败筛得的候选；R2 增加无修改重放、独立候选比较与 ticket 冻结。每题最多 3 个补丁、24 次验证、10 次模型调用和 65,536 输入加输出 token。
结果：R0/R1/R2 首选注入位置命中为 14/20、20/20、20/20；私有修复为 17/20、19/20、18/20。R1 相对 R0 多成功 2 题，补丁尝试从 31 降到 26；在线验证从 93 降到 78。R2 在线验证 138 次，没有额外私有成功。
解释：R1 是本次有限机制比较中更合适的定位起点；候选轮换重放在当前任务与策略下没有显示新增价值。责任索引增加输入长度，因此不能声称所有成本都下降。
关键边界：R0/R1 虽声明首选 owner，实际授权三个模块；R2 每 ticket 只授权一个。A2 的 R1 结果不能直接当作 A3 冻结后的“R1 + 强制冻结补集”组合性能。R2 有一次越界补丁被拒绝；三组均无已观测用例回归。

证据：[reports/A2.md](/home/lc/AuditTrainer/reports/A2.md)、[runs/a2/mechanism_p3.json](/home/lc/AuditTrainer/runs/a2/mechanism_p3.json)

## 第 09 页｜把责任计划、局部实现和父组合变成训练样本

目的：为 D、I/C 提供可追踪、可审查的一步动作监督。
设计：每个 root 在同一父状态下执行 ATOMIC、两个 child 流水组合、单个 fused child 三种合法计划，使用同一个参考执行器。参考执行器是审查过的程序构造，不是真实模型采样。候选代码经隔离执行，以另一份算法实现的 oracle 检查。TRAIN 私有检查只在终止后用于标签。
结果：150 条参考示范全部通过；另外每族固定一个 root 采集 K=1 的底座自然轨迹，共 10 条，均提出超出已显示 child_slots=3 的四 child 计划，在反馈后弃权，没有完整根程序。自然正例不足，所以正式训练主体来自参考示范。
导出：2,240 条动作包括 D_PLAN=460、D_IMPL=320、D_COMPOSE=400、D_CONTROL=1,060。责任支持 1,180 条，协议监督 1,060 条；控制动作正确不意味着代码责任已被证明。
解释：本阶段证明数据和训练链路有可信来源；150/150 不能解释为底座模型达到 100% 成功率。50 个参数化任务也不是 50 个独立任务族。

证据：[reports/A4.md](/home/lc/AuditTrainer/reports/A4.md)、[runs/a4/result.json](/home/lc/AuditTrainer/runs/a4/result.json)、[data/sft/train.manifest.json](/home/lc/AuditTrainer/data/sft/train.manifest.json)

## 第 10 页｜功能匹配的责任违规，检验了独立的责任维度

目的：检验责任标签是否能区分功能错误、责任违规和协议违规，尤其是输出正确但分工被破坏的情况。
设计与结果：遗漏输出、错误参数传递形成 20 对功能失败；算法迁移到 parent、调用 child 后丢弃结果并重算形成 20 对功能匹配责任违规；选择真实旧 artifact、扩大 child owns 形成 20 对协议/契约违反。每类各 10 对，全部真实执行。
解释：其中最重要的对照是“调用后弃用”：called=true 不足以支持责任正确。独立源码审查显示 parent 重算了本来委托给 child 的算法，即使根输出匹配也属于责任问题。这为后续责任监督提供了具体判据。
质量：用户报告人工复核通过，Codex 二次复核覆盖真实计划和全部负例；没有把总体复核反馈回填成逐事件真人标注。
边界：这些是受控构造，不是对任意程序语义的通用自动证明；不同执行 observation 不完全相同，preference_ready=true 为 0，所以不能直接作为同输入 DPO 偏好对。

证据：[reports/A4.md](/home/lc/AuditTrainer/reports/A4.md)、[runs/a4/contrasts.jsonl](/home/lc/AuditTrainer/runs/a4/contrasts.jsonl)

## 第 11 页｜四组训练分开检验 D 与 I 的针对性监督

目的：分离责任分解示范 D 与局部履约/父组合示范 I 的训练影响。
设计：F00 关闭 D 和 I，F10 仅开启 D，F01 仅开启 I，F11 同时开启。关闭 D 的槽位替换为基础 ATOMIC 计划；关闭 I 的 write_code 槽位用同 root 的普通整题功能示范替换。consume 与控制动作完全共享，不加入故意违反责任的坏目标。
匹配：四组使用同一公开 Qwen3-8B revision、seed 42、50 个 TRAIN root、2,240 个共同槽位和 140 次更新；均从底座新训，取最终 adapter，不继承 pilot 或挑中间 checkpoint。
解释：该设计检验训练覆盖因素的贡献。它不是仅改标签的消融：不同组的输入/目标 token、去重数据池、重复曝光次数不同。这些实际剂量差异在备份页给出。
训练：NF4/BF16，all-linear LoRA r16/alpha32/dropout0.05，约 43.65M 可训练参数，完整 action + EOS 损失。训练 loss 只验证优化过程，不据组间 loss 比较能力。

证据：[reports/A5.md](/home/lc/AuditTrainer/reports/A5.md)、[runs/a5/data_comparison.json](/home/lc/AuditTrainer/runs/a5/data_comparison.json)、[configs/a5/F11.yaml](/home/lc/AuditTrainer/configs/a5/F11.yaml)

## 第 12 页｜把“不会分解”与“不会实现 / 组合”分开测

目的：整条自由轨迹失败可能由计划格式、子算法、消费或父组合造成，必须通过固定输入条件定位能力来源。
五个条件：① local_plan 由被评模型生成根计划，再交由固定公开底座 E_ref 实现；② transform 固定局部契约，只测组件实现；③ summarize 同理；④ parent_both_children 固定两个只读子产物，只测 parent 消费和组合；⑤ parent_transform_child 只提供 transform 子产物，parent 需要自行履行 summarize 责任。
自由运行：全部决策由被评模型作出，使用 5 个 DEV_ENGINEERING root，即两个自建任务与三个 APPS 开发题。
解释：局部计划+E_ref 的成功率仍包含固定执行器的执行能力，并不是纯分解正确率。固定组件的 16 个条件是 8 个 root 各两种组件，不是 16 个独立任务。
注意 A5 的 E_ref 是公开未微调模型；A4 的参考示范执行器是已审查的程序构造，两者不能混称。所有私有分数在产物选择后计算，失败和无程序都保留分母。

证据：[configs/a5_evaluation.yaml](/home/lc/AuditTrainer/configs/a5_evaluation.yaml)、[reports/A5.md](/home/lc/AuditTrainer/reports/A5.md)

## 第 13 页｜I 监督带来最清楚的局部实现收益

目的：在责任契约已给定时，单独测模型能否写出符合接口和语义要求的局部实现。
结果：F00 的 transform/summarize 为 0/8、0/8；F10 为 3/8、0/8；F01 为 7/8、6/8；F11 为 6/8、6/8。两种组件合计依次为 0/16、3/16、13/16、12/16。
解释：开启 I 的两组都出现明显改善，支持履约监督对固定契约实现有作用。F00 的 transform 有 7 个任务因为公开函数接口不匹配而无程序，说明提升涉及按契约生成接口，不仅是同接口下的算法准确率。
描述性因素效应：I 的平均差为 ((13/16-0/16)+(12/16-3/16))/2=68.75 个百分点。此处只描述当前 seed 和开发条件，不报告统计显著性。
剩余错误：F11 的 prefix_span 曾调用未定义 helper，局部算法和接口错误并未完全消除。F11 没有超过 F01，不能声称 D 与 I 联合一定更好。

证据：[reports/A5.md](/home/lc/AuditTrainer/reports/A5.md)、[runs/a5/result.json](/home/lc/AuditTrainer/runs/a5/result.json)

## 第 14 页｜常见组合可以成功，职责变化后仍不可靠

目的：区分常见父组合模式与随当前契约重新承担责任的能力。
结果：F00/F10/F01/F11 双子模块功能通过为 1/8、1/8、8/8、7/8；只有 transform 子模块、父节点保留汇总时为 0/8、0/8、0/8、1/8。F01 的 8/8 对 0/8、F11 的 7/8 对 1/8 构成最明显的落差。
责任证据：F01/F11 的 15 个双子模块成功程序均有直接消费、传递返回值和只做解析/格式化的源码证据。F00/F10 虽各有一个功能通过，未满足该保守责任判据，不能把它们也算责任成功。UNKNOWN 不等于违规。
解释：履约示范帮助模型组合固定 child，但当前训练没有充分覆盖父节点保留 aggregate 的状态。单子模块失败表现为直接返回 transform 的列表、错误汇总，以及沿用训练题中的对象包装而不是当前所需 JSON 数值。
这支持一个具体覆盖缺口，尚不能仅由这组结果判定所有错误都来自模板记忆。A6 用一次数据干预检验该解释。

证据：[reports/A5.md](/home/lc/AuditTrainer/reports/A5.md)、[runs/a5/result.json](/home/lc/AuditTrainer/runs/a5/result.json)

## 第 15 页｜当前没有观察到主动递归分解的学习收益

目的：测分解训练是否改善计划形成，并用固定执行器减少后续执行模型差异。
结果：F00/F10/F01 的 8 个计划全部被接受，F11 只有 4 个被接受；之后由固定 E_ref 执行，私有成功分别为 5/8、5/8、5/8、3/8。所有被接受的计划均为 ATOMIC，没有主动形成 SPLIT。
失败解释：F11 四次失败发生在注册前，例如 obligation_map 的值写成字符串 self，而 schema 要求数组 [self]，一次反馈重试后仍未修正。有效计划之后 E_ref 的失败数分别为 3、3、3、1，需要单列，不能统称分解语义失败。
结论：D 监督没有在该隔离实验中形成可见递归分解收益，联合 F11 还存在格式稳定性问题。局部计划通过率的变化不能直接代表语义分解提升。
后续 A6 中注册失败消失，但计划仍全为 ATOMIC；因此 A6 局部计划成绩改善也不等于模型学会分解。

证据：[reports/A5.md](/home/lc/AuditTrainer/reports/A5.md)、[runs/a5/result.json](/home/lc/AuditTrainer/runs/a5/result.json)

## 第 16 页｜隔离条件的提升尚未形成端到端递归收益

目的：检查各模型在全部决策自由的情况下，局部收益是否转化为预算内整题成功。
结果：F00/F10/F01/F11 首个程序私有成功与最终成功均分别为 1/5、1/5、2/5、1/5。四组共 20 个运行全部保留，评分没有基础设施 ERROR。
行为：所有自由轨迹只形成 ATOMIC，没有实际修复尝试，所以 FirstProgramSuccess 与 BudgetedSuccess 相同。这说明模型没有主动调用框架已具备的递归和修复机制，不能据此评价修复是否有效。
解释：F01 比其他组多通过一个任务是描述性结果，只有 5 个开发 root，不应报告为稳定端到端优势。训练对固定组件的收益尚未充分转化成自由策略能力。
范围：三个 APPS 题来自已消费 train 开发记录，不代表整个 APPS 基准；其离线评分也不保证对预训练未见。早期 F9 的 base 2/5 使用不同阶段设置，不能拿来充当这轮严格匹配的额外基线。

证据：[reports/A5.md](/home/lc/AuditTrainer/reports/A5.md)、[configs/a5_evaluation.yaml](/home/lc/AuditTrainer/configs/a5_evaluation.yaml)、[reports/F9.md](/home/lc/AuditTrainer/reports/F9.md)

## 第 17 页｜只补父节点保留汇总的示范覆盖

目的：用最小的数据干预检验父职责覆盖不足这一解释。
原缺口：A4 的单 child fused 示范把 prepare 与 aggregate 一起委托，父节点主要做解析与格式化，未覆盖只委托 prepare、父节点承担 aggregate 的状态。
干预：在既有 50 个 TRAIN root 上新执行合法重分配示范，所有可见与 TRAIN 私有检查通过。导出 450 条合法动作，但只把其中 50 条 parent write_code 正例加入修订池。参考实现来自 TRAIN，未用开发题参考代码构造训练目标。
匹配：在原 F11 2,240 个采样槽位的 165 个父代码槽位中，按每 root 隔一个替换一个的规则替换 96 个，覆盖全部 50 个 root；其余 2,144 个槽位保持原样。种子、计划/组件/consume/control 示例、训练超参数及 evaluator 均保持。
实际剂量：数据池 2,240→2,290，不同曝光样本 953→970；目标 token 258,254→260,403，约 +0.83%。这是数据覆盖的单一主要修订，不是每个 token 完全相同的实验。

证据：[reports/A6.md](/home/lc/AuditTrainer/reports/A6.md)、[runs/a6/data_comparison.json](/home/lc/AuditTrainer/runs/a6/data_comparison.json)、[configs/a6.yaml](/home/lc/AuditTrainer/configs/a6.yaml)

## 第 18 页｜父职责重分配增加 1 个成功，同时存在回归

目的：在原封不动的 A5 evaluator 上检验一次覆盖修订的效果，主要指标预先固定为单 child、parent 保留汇总的通过数。
结果：主指标 1/8→2/8，增加 absolute_ranks，保留 digit_sums。固定组件 12/16→14/16；局部计划+E_ref 3/8→5/8；双 child 组合 7/8→6/8；自由运行 1/5→1/5。
解释：新增父职责示范改变了部分代码行为并带来一个新成功，支持有限覆盖收益。局部计划改善伴随原先 schema 注册失败消失，8 个计划仍全部 ATOMIC；不能声称分解能力提升。
回归：双 child 中 sorted_squares 新成功，但 reverse_negate 和 absolute_ranks 因消费状态或 alias 错误无父程序，净少一个成功。自由运行成功任务也从词序去重换成括号配对，三个 APPS 仍 0/3，不能用总数相同掩盖样本变化。
结论：保留正向变化和回归，一次修订按预定规则结束；F11_C 不是所有指标更好的新最优模型。

证据：[reports/A6.md](/home/lc/AuditTrainer/reports/A6.md)、[runs/a6/result.json](/home/lc/AuditTrainer/runs/a6/result.json)

## 第 19 页｜源码案例说明：开始承担职责，仍需完整履约

目的：解释分数变化背后模型究竟做了什么，避免只报告平均值。
新增成功：absolute_ranks 中，F11_C 使用 transform 返回的有序去重列表，计算加权绝对值之和 sum(i * abs(v) ...)，并按根契约直接返回 JSON 数值。它完成了先前欠缺的父汇总职责。
仍失败：sorted_median 和 sorted_squares 已写出所需汇总计算，却返回带 value 或 sum 字段的对象，当前契约要求直接 JSON 结果。因此即便算法步骤有所改善，最终仍失败；这属于输出契约不符，不是无关紧要的空格问题。
回归：双 child 的 reverse_negate 和 absolute_ranks 遇到消费状态/alias 错误，没能产生父程序。算法实现能力和协议稳定性是不同问题。
结论：完整履约包含正确算法、精确输出契约和稳定使用依赖。该组案例支持有限行为变化；没有把单 child 功能通过自动提升成所有责任语义已证明。

证据：[reports/A6.md](/home/lc/AuditTrainer/reports/A6.md)、[runs/a6/result.json](/home/lc/AuditTrainer/runs/a6/result.json)

## 第 20 页｜下一步补齐“责任结构 → 修复价值”的证据

总结：框架闭环和可追踪训练数据已完成；A5 清楚显示固定契约履约与双子模块组合收益；职责重分配暴露了更深的能力缺口，A6 一次干预带来有限改善。A2 还提供了责任筛选支持受控修复的初步依据。
A7 轨道一固定失败源码、证据、修复器和预算，比较定位和修改边界方法，包含受控故障与自然失败。只有有可靠标签的注入故障报告定位命中，自然失败不制造唯一根因标签。
轨道二比较 F00/F10/F01/F11 生成的结构，共用同一定位器与修复器；受控注入尽量对应同一外部义务，并报告全 root 的预算内成功。不能只比较各模型自己的失败修复率，因为失败难度可能不同。
最小对照：无修复、整程序修复/重试、仅提示局部区域、责任证据+ticket+冻结补集+可见重放。指标包括私有成功、边界大小、越界、回归和总成本。
当前停在一次修订之后，继续既定 A7；DPO/RL 不是前置条件。正式封存评测与多 seed 稳定性尚未完成，今天的结果属于开发性证据。

证据：[STATUS.md](/home/lc/AuditTrainer/STATUS.md)、[reports/A2.md](/home/lc/AuditTrainer/reports/A2.md)、[reports/A6.md](/home/lc/AuditTrainer/reports/A6.md)

## 第 21 页｜A5 四组结果：保留全部条件与分母

用途：导师追问某组或某条件时直接展示完整结果。
本表来自 runs/a5/result.json 与 reports/A5.md。四组分别为功能对照、仅 D、仅 I、D+I。每个单条件有 8 个 root；自由运行有 5 个 root。所有无程序、协议失败、弃权均保留在分母。
双子模块“功能+责任证据”使用保守源码规则：parent 只解析输入、调用已选择 transform、把其返回值传给已选择 summarize、格式化结果，且没有额外算法或别名覆盖。其他写法保留 UNKNOWN，不能把 UNKNOWN 算成责任违规。
各组首个与最终自由成功相同，是因为没有修复尝试。不要把固定组件合计、两种父组合合计当作新的独立任务集合，也不要拿同一 root 的多个条件做未经设计的独立样本显著性检验。

证据：[reports/A5.md](/home/lc/AuditTrainer/reports/A5.md)、[runs/a5/result.json](/home/lc/AuditTrainer/runs/a5/result.json)

## 第 22 页｜匹配更新预算；如实报告 token 与重复曝光差异

用途：回答四组是否公平、训练是否真实完成，以及为什么不根据 loss 选模型。
匹配项：固定公开底座 revision，NF4/BF16、all-linear LoRA r16/alpha32/dropout0.05、43,646,976 个训练参数、8,192 长度、batch1×累积16、学习率 1e-4、cosine、3% LR warmup；共同采样槽位与 seed42。前 20% 更新使用履约/控制课程。
不匹配项：数据池去重大小、实际曝光的不同事件、目标 token 与输入 token。F00/F10/F01/F11 的不同曝光数为 404/646/711/953。A6 为 970。因监督内容改变，这些并不是仅标签变化或 token 严格等量的消融。
完成证据：五组均完成 140 步，LoRA 参数真实变化，梯度检查与磁盘重载、运行回接通过；固定输入重载 logits 最大差为 0。训练 loss 因目标不同不能用于组间能力排名。共享服务器墙钟受负载影响，不用训练时长比较速度。
A6 保持相同初始参数和槽位顺序，仅替换 96 个父组合槽位；目标 token 约增加 0.83%。

证据：[reports/A5.md](/home/lc/AuditTrainer/reports/A5.md)、[reports/A6.md](/home/lc/AuditTrainer/reports/A6.md)、[runs/a5/data_comparison.json](/home/lc/AuditTrainer/runs/a5/data_comparison.json)

## 第 23 页｜每个分数在测什么，哪些结论不能直接推导

FirstProgramSuccess：第一次完整根程序及其依赖闭包的私有通过率，分母为全部选中 root。生成根程序之前可能已做子节点验证和修复，因此不是一次模型调用的 Pass@1。
BudgetedSuccess：终止时当前根产物的私有通过率，即使预算耗尽或弃权，已有产物仍被评分，终止原因另报。首个和最终程序分别冻结，后续修复不能改写首个快照。
功能与责任：功能 PASS 是通过适用测试；责任支持需要明确的契约、源码和依赖证据。UNKNOWN 表示证据不足，不是违约，也不能当责任正例。没有适用局部 oracle 时不能继承根 PASS。
成本：模型调用、输入/输出 token、在线验证与重放、离线评分单独记录；失败补丁成本保留。无成功时每成功成本为 N/A。
数据范围：A2 的20题来自同一 family；A5/A6 隔离用其中8题，自由用5题。私有测试不参与在线选择，但开发结果已被用于诊断和一次修订，所以不能称最终未见泛化测试。

证据：[reports/F9.md](/home/lc/AuditTrainer/reports/F9.md)、[docs/data_check_protocol_v1.md](/home/lc/AuditTrainer/docs/data_check_protocol_v1.md)、[reports/A5.md](/home/lc/AuditTrainer/reports/A5.md)

## 第 24 页｜早期 smoke 验证了执行闭环，结果不作为训练收益

用途：回答递归、修复、训练与私有隔离是否真正运行过。
F1/F2 完成原子动作状态机及隔离验证，FX01 六个用例通过；F3 三层递归 fixture 为4节点、最大深度2，四节点均功能通过，19次脚本策略动作，责任仍保留 UNKNOWN。
F4 两个 fixture 分别完成只修根代码和计划子树重建，均从失败变为通过；前者冻结 child，后者保持 incoming contract 并重建、重验、重新消费子树。这是执行机制验收，不是真实模型已经学会修复的证据。
F5 接入真实 Qwen3-8B，F6 导出32条一步动作，F7 进行10步 QLoRA smoke、参数变化和重载最大差0；F8 开关 adapter logits 最大差28.625、恢复差0，证明权重生效。
F9 统一评价：脚本6个fixture首个3/6、最终5/6；base五个开发题2/5，smoke adapter为0/5。0/5 是保留的负结果，连通通过不等于能力提升。脚本与模型任务不同，不能跨组公平比较；F9 与 A5 预算和训练设置也不同。
A3 冻结协议后六个fixture最终5/6，其中一个为预设错误父程序；最新A6全量261项测试通过。各阶段测试数是累计套件规模，不能逐阶段相加。

证据：[reports/F3.md](/home/lc/AuditTrainer/reports/F3.md)、[reports/F4.md](/home/lc/AuditTrainer/reports/F4.md)、[reports/F7.md](/home/lc/AuditTrainer/reports/F7.md)、[reports/F9.md](/home/lc/AuditTrainer/reports/F9.md)、[reports/A3.md](/home/lc/AuditTrainer/reports/A3.md)

## 第 25 页｜当前结论与实验边界可以这样回答

问：为什么层次契约在 A2 成功率低，还继续这条路线？
答：A2 没证明零样本生成更强，但揭示了 child 已通过而 parent 仍失败的明确缺口；契约提供可绑定局部检查的学习对象。后续 A5 观察到固定契约履约和双 child 组合的训练响应，支持继续检验这条方向。
问：最强的阶段结果是什么？
答：F01 相对 F00 的固定组件 0/16→13/16、双 child 组合1/8→8/8；同时职责重分配0/8暴露了常见组合之外的能力缺口。前者是正向证据，后者是具体研究问题。
问：F11_C 是否已经全面优于 F11？
答：没有。主指标1/8→2/8、组件12/16→14/16，但双 child7/8→6/8，自由成功总数不变且通过任务替换，因此报告有限改善和回归。
问：论文核心还缺什么证据？
答：生成期责任结构对固定修复器的下游价值仍缺 A7 的决定性比较；多 seed 稳定性及正式封存评测还未完成。下一步不扩大工程范围，按两轨实验补这个环节。
问：是否有数据泄漏或挑选 checkpoint？
答：TRAIN 与开发按 root/family 隔离，私有评分不参与在线选择，五组都取140步最终权重。A5/A6 使用已消费开发集，修订也基于开发诊断，因此如实称开发比较，不能称最终未见测试。

证据：[reports/A2.md](/home/lc/AuditTrainer/reports/A2.md)、[reports/A4.md](/home/lc/AuditTrainer/reports/A4.md)、[reports/A5.md](/home/lc/AuditTrainer/reports/A5.md)、[reports/A6.md](/home/lc/AuditTrainer/reports/A6.md)
