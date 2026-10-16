# 不同责任的内部compute价值诊断

2026-09-18。沿用handoff_sft执行器、gate_refresh根门控、原MULTI_BODY正文及内部策略，冻结所有参数。

## 固定实验

选择既有课程中全部4个TRAIN内部compute状态：calendar→matrix模4、intervals→matrix模4、chunks→chunks宽4、words→words参数4。选择在本轮结果产生之前确定，覆盖4种已知组合；这些状态曾进入正文SFT，不作为未见任务泛化。

根计划固定为将PREPARE与AGGREGATE一起交给compute；当前compute同时承担两个不同的责任。先实际注册根计划、进入compute，验证完整观察与课程导出状态一致。随后只强制compute第一次选择：ATOMIC或当前正文新生成的SPLIT。后续拒绝、重试和所有执行均沿用当前冻结策略。每状态4份新SPLIT，每份1次完整根续跑，与4个ATOMIC配对，共32条。

种子为计划441000713、执行442000713，加状态序号×100与重复序号。配对初始执行种子相同，不宣称分支后的随机流相同。不能用作者正确计划的效果代替模型生成计划的平均收益。

预先区分四种已审阅计划：parent_aggregate（子做prepare，当前保留aggregate）、parent_prepare（当前保留prepare，子做aggregate）、pipeline（两个子责任及数据流）、fused（整项compute再次委托）。前三者属于不同责任分配；最后一种或完整契约相同的单子计划不算实质细化。结构匹配包含契约、责任所有者、保留职责和数据流，并消除子别名差异。未匹配不自动等于错误，不筛除其执行结果。

主指标为最终根私有成功；辅助记录compute局部私有检查、根真实调用、计划注册和成本。4项状态绑定与结构区分测试通过。单条成功差异（1/4）不形成强方向标签。本轮不训练内部门控。

## 结果

**32条冻结执行已完成。本轮16份新SPLIT全部分离了不同责任，但未观察到内部SPLIT的正成功收益。**

| 内部compute状态 | ATOMIC根成功 | SPLIT根成功 | 新SPLIT责任结构 | 当前解释 |
|---|---:|---:|---|---|
| calendar→matrix，模4 | 0/4 | 0/4 | 子prepare、当前aggregate，4份 | 两种执行均未解决，不能形成强停止标签 |
| intervals→matrix，模4 | 0/4 | 0/4 | prepare→aggregate双子流水线，4份 | 两种执行均未解决，不能形成强停止标签 |
| chunks→chunks，宽4 | 4/4 | 2/4 | 子prepare、当前aggregate，4份 | 当前执行器下支持内部停止 |
| words→words，参数4 | 4/4 | 3/4 | prepare→aggregate双子流水线，4份 | 单条差异，暂不形成强标签 |
| 合计 | **8/16** | **5/16** | 16份均为不同责任分配 | 配对SPLIT 0胜、3负、13平 |

compute局部私有通过数与表中根成功数相同。根实际调用compute的次数，按上述状态顺序，ATOMIC为4、3、4、4，SPLIT为3、2、4、3；调用发生本身不代表功能正确。32次首次内部计划均注册成功，16份SPLIT无同契约重复委托。

## 失败证据

本轮已经越过“只能增加一层同契约委托”的局限，仍存在执行器没有完成所分配责任的问题。以下为保存轨迹中的具体实例，不将代表样本外推为全部失败的唯一原因。

- **calendar SPLIT repeat_02：** prepare正确转换并去重排序分钟数；compute保留aggregate，却调用没有定义或绑定的`_aggregate_child`。见[轨迹](../runs/refinement/compute_values/train_course_calendar_matrix_04/SPLIT/repeat_02/runtime/trace.jsonl)。
- **intervals SPLIT repeat_02：** prepare正确展开区间；aggregate调用未定义的`_base`，后续compute无法产生正确结果。见[轨迹](../runs/refinement/compute_values/train_course_intervals_matrix_04/SPLIT/repeat_02/runtime/trace.jsonl)。
- **chunks SPLIT repeat_01、repeat_03：** prepare正确，compute仅返回分块列表，遗漏契约要求的含`chunks`和`length`的对象；根实际调用后失败。见[repeat_01](../runs/refinement/compute_values/train_course_chunks_chunks_04/SPLIT/repeat_01/runtime/trace.jsonl)、[repeat_03](../runs/refinement/compute_values/train_course_chunks_chunks_04/SPLIT/repeat_03/runtime/trace.jsonl)。
- **words SPLIT repeat_00：** prepare和aggregate完成其职责，但compute提前`json.dumps(result)`；当前契约要求原生结果，最终序列化属于根职责。见[轨迹](../runs/refinement/compute_values/train_course_words_words_04/SPLIT/repeat_00/runtime/trace.jsonl)。

本轮未出现越级消费的`CHILD_NOT_PENDING`或哈希/版本绑定错误。共17次`UNVERIFIED_CONSUMPTION`拒绝，与未通过局部检查的产物有关，已保留在完整执行收益中；不能把首次计划全部注册表述为全程无协议拒绝。

## 成本与验证

| 状态 | 平均计费token A / S | 平均调用 A / S | 平均节点 A / S |
|---|---:|---:|---:|
| calendar→matrix | 20448.25 / 32602.25 | 10 / 14.25 | 2 / 3 |
| intervals→matrix | 20909.75 / 44347.25 | 9.25 / 19.25 | 2 / 4 |
| chunks→chunks | 17343 / 30080.25 | 9 / 14.5 | 2 / 3 |
| words→words | 17050.5 / 37757.5 | 9 / 18.5 | 2 / 4 |

32条完整状态相等检查通过；0基础设施错误、0参数更新，根门控、正文和执行器的内存指纹及模型文件保持。4项定向测试通过，覆盖当前状态/责任结构与既有内部探针兼容性。

运行入口为`python -m audittrainer.refinement_compute_values prepare`、`run --index 0`至`run --index 3`、`report`，使用`audittrainer`环境。实现复用内部动作拦截器，仅将目标节点从默认prepare扩展为可指定compute。

机器结果：[result.json](../runs/refinement/compute_values/result.json)；验收：[validation.json](../runs/refinement/compute_values/validation.json)。

## 决策

**本轮不启动内部门控训练。** 正文已经能在这些已知TRAIN状态生成真正的责任分配，但当前完整执行尚未产生可用于学习“何时值得内部细分”的正收益状态。calendar/intervals的0对0不强造负标签，words的单条差异保留不确定；chunks在当前策略下优先内部停止。此前根分解收益与简单prepare停止偏好仍保留。

下一步优先做有限的固定计划执行因果修正：保留计划和其他代码，只修当前aggregate/compute的未定义调用、遗漏输出字段和提前序列化，检查局部及最终根恢复。确认可恢复后，再决定小规模执行器学习；改变执行器后重新采ATOMIC和新SPLIT价值。人工修正不能计入模型成绩。本轮未进行这些修正或训练。

这些状态曾参与正文SFT，本结果是已知TRAIN内部责任细化的价值诊断，不是未见组合泛化，也不支持“递归普遍无效”的结论。
