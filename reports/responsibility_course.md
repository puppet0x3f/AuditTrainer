# 责任课程与受控深度 2

2026-09-17。COMPLETE。参考课程、80 步课程训练、48 条深度 2 评测及两个 CPU 局部反事实均已完成。

**本轮承接纯正文 27/64 → 64/64 的可学习性结果，进入多任务责任结构和内部节点。** 训练从原共同结构 SFT 独立开始，calendar 过拟合适配器只用于诊断。

## 课程与监督边界

24 个 TRAIN 根覆盖 6 类结构，每类 4 根；涉及 20 个组合族，不能计为 24 个独立任务族。另固定 12 个未进入训练的开发验证根；这些来自已经用于开发研究的旧验证组合，本轮尚未据此进行新泛化评测。

| 结构 | 根数 | 示范内容 |
| --- | ---: | --- |
| ATOMIC | 4 | 允许分解时直接实现 |
| 父保留聚合 | 4 | 子 prepare，父处理聚合 |
| 父保留预处理 | 4 | 父 prepare，子 aggregate |
| 双子流水线 | 4 | prepare → aggregate 的数据流 |
| 共同支持上层义务 | 4 | 两个子所有者共同支持 RESULT |
| 内部细化 | 4 | 根委派 compute，compute 继续细化 |

24 棵手写参考树共 58 节点，所有根私有评分及 34 个非根私有检查通过；参考执行不调用模型。导出 52 个允许 ATOMIC/SPLIT 的真实规划状态，其中 24 个根、28 个非根；ATOMIC 28，SPLIT 24。深度上限强制 ATOMIC 的 6 个孙节点不作为停止学习示范。

示范 observation 来自实际 runtime 状态，保留传入契约、依赖和剩余预算。私有评分在执行后独立进行，不进入模型输入。参考源码明确消费子返回值，根和内部组合沿用现有契约、owner 映射及 dataflow，没有新增协议。

监督标签表示可执行责任结构，**不表示冻结 F01 下的最优模式标签**。运行时的完整语义责任状态仍可为 UNKNOWN；有限测试与源码检查不等同于通用责任证明。停止控制中的参考 SPLIT 用于比较过度分解，不预先保证增加深度有收益。

## 构建中修正的具体评分错误

新增 matrix → calendar 组合暴露了原独立 oracle 的错误：输入列和 `[-6,-9]` 时，公开契约要求最大相邻差为 `-3`，但 oracle 把初始最大值设为 0，错误期望 0。参考实现返回 -3，因而被私有评分误判。

已将 calendar oracle 改为从首个相邻差初始化；空或单元素仍返回 0。原始 calendar 的排序输入不受影响。失败构建、原任务与说明保存在 [construction_correction](/home/lc/AuditTrainer/runs/refinement/responsibility_course/construction_correction/reason.json)，在任何课程模型训练和 F01 续跑前重新生成该任务，并完成全部参考执行。添加了对应回归测试；未改写历史已运行实验的评分文件。

## 有限训练与深度 2 设计

课程 SFT：80 次更新，学习率 1e-5，梯度累积 4；共 320 次曝光，其中 256 次课程状态、64 次原结构 SFT 回放。课程按根均衡循环，并轮换根内状态；门控 CE＋平均正文 CE，无额外边界加权、KL 或 RL。规划 LoRA 共享，F01 冻结。 实际完成 80 步与全部 320 次曝光：ATOMIC 162、SPLIT 158，课程非根状态曝光 112 次。训练约 40.3 分钟，F01 指纹不变，保存重载前向差为 0。新适配器仅用于后续评测，未替换任何既有实验基线。

深度 2：calendar→matrix、intervals→matrix、chunks→chunks、words→words 各 4 个配对种子；固定根为 compute 子契约，开放其内部决策，比较规范 ATOMIC、参考 SPLIT、课程模型自然规划，共 48 条。前两根观察内部细化，后两根作为停止控制。

固定计划组不调用学习规划器，只使用 F01，因此在课程训练期间先执行；NATURAL 在课程适配器完成后启动。所有组冻结同一 F01、父组合摘要、max_nodes=7 与原公共预算，不做功能修复。接受固定根和内部计划后重置续跑随机种子，不重置已消耗预算。

自动记录两层 COMPOSE 的实际摘要输入、每层功能状态、CONSUME、依赖调用、协议重试和成本；依赖发生调用不自动解释为语义消费正确。小量参考案例另检查源码中子结果的实际使用。

## 深度 2 的完整结果

全部 48 条完成，无基础设施错误；F01 在三组完全相同且始终冻结，评测没有优化器更新。77 次实际 COMPOSE 输入均包含冻结的父组合摘要，覆盖根和内部节点。

| 任务 | 内部 ATOMIC | 参考 SPLIT | 课程模型自然规划 |
| --- | ---: | ---: | ---: |
| calendar→matrix | 4/4 | 4/4 | 3/4 |
| intervals→matrix | 1/4 | 1/4 | **4/4** |
| chunks→chunks | 4/4 | 4/4 | 2/4 |
| words→words | 4/4 | 0/4 | 3/4 |
| 合计 | **13/16** | **9/16** | **12/16** |

自然规划相对内部 ATOMIC 为 3 胜、4 负、9 平：intervals 获得 3 条额外成功，其他三根共损失 4 条。**内部责任分配已有可用的收益支点，但当前课程模型没有整体超过 ATOMIC。** 本实验没有加入旧共同 SFT 的自然内部规划组，因此也不能据此单独估计课程 SFT 的训练增益。

自然规划首次内部注册 13/16，保留原协议重试后注册 15/16；15 个被接受的内部模式全部为 SPLIT，没有观察到主动 ATOMIC 停止。另一个状态两次数据流拒绝后耗尽重试。宿主在深度上限强制孙节点 ATOMIC，不计为停止能力。

对接受计划按原有契约、责任所有者、父保留职责和数据流进行事后精确归类（规范化别名及顺序）：

| 自然规划实际分配 | 次数 | 根任务通过 |
| --- | ---: | ---: |
| 子 prepare，父保留 aggregate | 10 | 10 |
| 父 prepare，子 aggregate | 2 | 2 |
| prepare → aggregate 双子流水线 | 2 | 0 |
| compute → aggregate 的不兼容组合 | 1 | 0 |
| 未能注册内部计划 | 1 | 0 |

这些结果不表示双子结构本身不可行：参考流水线在 intervals 有一条完整通过。它们说明当前 F01、提示和自然正文分布下，不同责任分配的执行可靠性明显不同；参考结构不等于最优结构。

实际模型输入＋输出 token 分别为 ATOMIC 203,635、参考 SPLIT 454,568、自然规划 539,478；固定条件的计划由脚本给定，不调用学习规划器。自然组因此不能声称效率提升。三组根责任状态均为 UNKNOWN；局部测试、绑定消费和少量成功案例的源码使用证据不等同于完整语义责任证明。

## 失败机制及最小复核

**执行侧：内部结果被当作根输出文本序列化。** 参考 SPLIT 的 7 条失败中，5 条在两个孙节点都通过后，compute 却调用 `json.dumps` 返回字符串；另 2 条包含 aggregate 实现错误。自然规划也有 2 条相同的多余序列化失败。内部局部测试和上层组合要求结果对象，但签名只写 `JSON value`；根说明明确要求 solve 返回 JSON 文本，当前输入没有充分突出两者的边界。

选取 intervals、words 各一条参考组失败，固定全部生成子产物，只把 `return json.dumps(result)` 改成 `return result`。两例公开、私有局部检查均由 FAIL 变为 PASS。新版本另存，原产物不变，0 模型调用、0 更新。这证明这两例的多余序列化足以造成局部失败；不把人工修改计入模型的根任务成功率，也尚未证明仅改提示就能让 F01 稳定修正。

**规划侧：声明有效不等于组合相容。** chunks 的一条自然计划引用未声明的 aggregate 子节点，重试后的数据流仍无效。words 的最后一条先因相同问题被拒绝，随后注册了 `compute → aggregate`：compute 已返回最终结果字典，而 aggregate 要求预处理后的字符串列表。两个子程序各自通过局部验证，内部组合却失败。该例的责任所有者、保留职责和子输出/下游前置条件没有形成一致的细化。

证据：

- [接口反事实复核](/home/lc/AuditTrainer/runs/refinement/depth2_local/interface_counterfactual/result.json)
- [两个子组件通过后的错误序列化](/home/lc/AuditTrainer/runs/refinement/depth2_local/evaluation/REFERENCE_SPLIT/train_course_intervals_matrix_04/repeat_01/runtime/artifacts/N0_p1_compute/v1.py)
- [不兼容的 compute → aggregate 组合](/home/lc/AuditTrainer/runs/refinement/depth2_local/evaluation/NATURAL/train_course_words_words_04/repeat_03/runtime/trace.jsonl)
- [数据流拒绝案例](/home/lc/AuditTrainer/runs/refinement/depth2_local/evaluation/NATURAL/train_course_chunks_chunks_04/repeat_02/runtime/trace.jsonl)

## 下一步顺序

本轮工作包 0—2 已收束，不追加单例调参。继续使用已经具备的结构 SFT、冻结 F01 和递归运行时。

1. **先做小规模内部接口校准。** 在既有 signature / requires / ensures 和父组合输入中，明确当前节点返回原生对象还是编码文本、子输出是否满足下游输入；不新建 schema 或通用类型推理器。用两个已通过子产物的固定内部状态，对原输入与澄清输入各做 4 个配对种子，共 16 条内部组合及祖先续跑。原 48 条结果保持冻结。
2. **据此补内部停止和字段一致性的课程对照。** 当前递归示范的 compute 全部标为 SPLIT，而四个 ATOMIC 课程示范位于根；不能期待这已经覆盖内部主动停止。补允许两种模式的内部 ATOMIC 示范，并处理已出现的悬空数据流和输出/前置条件不匹配，不要求清除所有可能错误。
3. **再进入原定正文—门控闭环的小试验。** 保留 intervals 的责任分配收益支点及 chunks/words 停止控制；正文更新后重新采样，以当前平均 SPLIT 收益（包括拒绝、重试和执行失败）校准门控。完整计划正文、门控和自然根收益分别报告。先完成上面的有限校准，再投入原工作包 3 的 1,296 条对照预算。

新课程适配器是后续试验的候选共同起点，尚不替换既有主基线。上述结果是 4 个已训练根、一个训练种子的受控内部决策诊断；12 个开发验证根尚未新增评测，没有自然整树递归或泛化结论。

## 运行入口与证据

```bash
conda run -n audittrainer python -m audittrainer.refinement_responsibility_course references
conda run -n audittrainer python -m audittrainer.refinement_course_train train
conda run -n audittrainer python -m audittrainer.refinement_depth2_local prepare
conda run -n audittrainer python -m audittrainer.refinement_depth2_local evaluate --arm ATOMIC
conda run -n audittrainer python -m audittrainer.refinement_depth2_local evaluate --arm REFERENCE_SPLIT
conda run -n audittrainer python -m audittrainer.refinement_depth2_local evaluate --arm NATURAL
conda run -n audittrainer python -m audittrainer.refinement_depth2_local report
```

训练入口有意拒绝覆盖已有训练目录；执行入口复用已完成 case，不覆盖未完成产物。

- [课程配置](/home/lc/AuditTrainer/configs/refinement_responsibility_course.yaml)
- [参考执行汇总](/home/lc/AuditTrainer/runs/refinement/responsibility_course/reference_result.json)
- [课程训练结果](/home/lc/AuditTrainer/runs/refinement/responsibility_course/training/training_report.json)
- [固定训练曝光表](/home/lc/AuditTrainer/runs/refinement/responsibility_course/training/request.json)
- [深度 2 配置](/home/lc/AuditTrainer/configs/refinement_depth2_local.yaml)
- [48 条完整结果](/home/lc/AuditTrainer/runs/refinement/depth2_local/result.json)
- [深度 2 固定请求](/home/lc/AuditTrainer/runs/refinement/depth2_local/request.json)

与本轮修改相关的 17 项定向测试通过，覆盖评分反例、诊断目标、课程隔离、真实递归消费、内部决策后随机种子重置及旧固定根行为。
