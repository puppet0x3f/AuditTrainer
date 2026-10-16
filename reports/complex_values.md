# 新复杂内部任务的冻结价值诊断

2026-09-18。承接[内部价值重估](compute_refresh.md)。原4状态A/S全部成功后，本轮更换任务结构，不继续在简单状态上重复修复或训练门控。

## 预先固定的任务与范围

| 内部compute任务 | PREPARE职责 | 依赖其结果的AGGREGATE职责 |
|---|---|---|
| weighted_routes | 有向非负加权图最短路，处理重复边、零权环和不可达节点 | 按阈值统计可达数量、距离总和、最远节点及字典序平局 |
| stock_events | 按时间及原序重放库存变动，拒绝负库存事件 | 按目标库存计算补货、排序、保留拒绝索引 |
| coverage_queries | 扫描重叠区间，生成最大等覆盖数分段 | 对多个查询计算覆盖长度、覆盖重数积分和峰值 |
| polynomial_chain | 多个稀疏多项式相乘，合并同次项及抵消项 | 指定阶数求导、多点评估及零多项式次数 |

四项在任何模型执行结果产生之前固定，按结构纳入全部任务，不按ATOMIC失败或SPLIT成功筛选。它们比原先简单映射/筛选增加了算法步骤与中间数据依赖，但“对当前模型是否更难”由实测回答，不能预先当作已证实。均为新建开发任务、标记VALIDATION，本轮不进入训练；不是封存最终基准，也不作广泛未见能力声明。

每项6个确定性输入，前2个可见、后4个私有，覆盖空输入、平局、重复、抵消、边界等。期望值由不同算法核对：Dijkstra与迭代松弛、全局事件重放与逐SKU重放、扫描线与整数格点计数、逐项卷积与项组合展开/直接求导。手算样例另行断言。作者完整递归程序4/4通过可见及私有根测试；10项定向测试通过，随后追加的反例测试也通过。作者源码、私有答案不进入模型观察，公开接口和可见样例保留。

## 模型实验

冻结compute_sft执行器、gate_refresh根门控和原MULTI_BODY。固定根fused计划，进入同一内部compute完整观察后，强制首次ATOMIC或当前正文新采样SPLIT。每任务4次A、4份独立新SPLIT各续跑1次，共32条。后续动作自然执行，完整保留拒绝、重试、失败；不按计划匹配结果筛除样本。

沿用max_depth=2等全部运行预算，不因任务更复杂而给SPLIT额外协议预算。根成功为主，局部、根真实调用、责任结构和成本为辅。种子为计划471000713、执行472000713，加任务序号×100和重复序号；共享初始执行种子不等于后续随机流一致。门控不更新，不另跑自然模式评估。

差异仅1/4时保留不确定；更大差异也只是小样本探索证据。成功持平与成本优势分开记录，不制造负成功标签。

## 结果

**32条模型执行完成，当前未得到可靠的内部细分正收益证据。原始测试为ATOMIC 0/16、SPLIT 1/16；唯一通过程序随后被明确的契约反例否定。**

| 新内部任务 | 原始ATOMIC根成功 | 原始SPLIT根成功 | compute局部私有 A / S | 首次SPLIT计划注册 |
|---|---:|---:|---:|---:|
| 加权路径查询 | 0/4 | 1/4* | 0/4、1/4 | 4/4 |
| 库存事件重放 | 0/4 | 0/4 | 0/4、0/4 | 4/4 |
| 覆盖查询 | 0/4 | 0/4 | 0/4、0/4 | 4/4 |
| 多项式乘积求导 | 0/4 | 0/4 | 0/4、0/4 | 3/4 |

15份新SPLIT匹配真实parent_aggregate分配：子节点prepare，当前compute保留aggregate；不存在同契约重复委托。多项式repeat_01的首次计划为INVALID_ACTION_JSON，仍保留在SPLIT均值中，后续重试也未成功。全体首次计划31/32注册，不能称为全部计划有效。根实际调用compute次数按表顺序A为3、4、1、4，S为3、4、1、4；调用发生不等于计算正确。

门控冻结，四状态P(SPLIT)约18.2%、16.5%、18.2%、18.2%。本轮没有自然门控执行组，也没有据结果更新这些概率。

## 唯一通过产物的反例核查

加权路径SPLIT repeat_01通过原先可见及私有根测试，但检查代码发现其最短路实现只按输入顺序松弛边一次。输入契约并未保证拓扑边序，且允许环，因此这不是完整实现。原先6个图样例未暴露这一点，不能把原始1/4包装成可靠的分解正收益。

同一张图`[['s','a',1],['a','b',1]]`与逆序`[['a','b',1],['s','a',1]]`，source均为`s`、limit均为2，正确结果都应为count=3、total=3、farthest='b'。该程序在逆序时漏算b。

对该任务**全部8份已生成程序**统一加入这两个根用例，与原测试一起复核；原始数据、代码和分数保留，程序副本只用于验证，0模型调用、0代码修正。追加检查后A与S均0/4。这个追加集是在观察错误后建立的明确契约反例，不是预注册独立泛化集，也不是一次新采样估值。原始结果仍记为0/4对1/4，反例结果单独存储。

原程序：[路径SPLIT repeat_01](../runs/refinement/complex_values/validation_internal_complex_weighted_routes/SPLIT/repeat_01/runtime/trace.jsonl)。反例：[检查结果](../runs/refinement/complex_values/edge_order_check/result.json)。该条先前通过compute测试，也提醒局部有限用例通过不能代替完整契约正确性。

## 已定位的执行问题

- 加权路径：ATOMIC读取不存在的`items['nodes']`，对子项结构处理错误；SPLIT也有单次松弛、错误遍历以及遗漏当前`_aggregate`实现。见[ATOMIC repeat_00](../runs/refinement/complex_values/validation_internal_complex_weighted_routes/ATOMIC/repeat_00/runtime/trace.jsonl)。
- 库存事件：ATOMIC未按时间重放，遗漏仅在initial/targets出现的SKU；SPLIT按(time,SKU)排序而非(time,原索引)，错误记录拒绝索引，父聚合引用未定义的target。见[SPLIT repeat_00](../runs/refinement/complex_values/validation_internal_complex_stock_events/SPLIT/repeat_00/runtime/trace.jsonl)。
- 覆盖查询：未正确实现重叠重数的分段扫描，聚合将三元分段按二元组解包，另有源码/动作格式错误。见[ATOMIC repeat_00](../runs/refinement/complex_values/validation_internal_complex_coverage_queries/ATOMIC/repeat_00/runtime/trace.jsonl)。
- 多项式：将输入对象误作列表，未实现乘积或求导；SPLIT子函数调用缺失的`_multiply`，compute调用缺失的`_aggregate`。见[SPLIT repeat_00](../runs/refinement/complex_values/validation_internal_complex_polynomial_chain/SPLIT/repeat_00/runtime/trace.jsonl)。

全32条保留的协议拒绝计数为UNVERIFIED_CONSUMPTION 25、INVALID_SOURCE 5、INVALID_ACTION_JSON 6、ARTIFACT_VERSION_MISMATCH 6、UNKNOWN_EVIDENCE 5。不能把所有失败都归因于单一算法问题，也不能把宿主拒绝不正确产物视为协议缺陷。

## 验证与决策

32条均完整结束、0基础设施错误、0参数更新；完整内部观察逐次一致，门控/正文/执行器指纹和文件保持。开发样本training reward始终为null，成功统计取最终私有根测试，没有将开发反馈写入训练数据。10项定向测试＋1项反例回归测试通过，保存的源文件及反例副本哈希一致。

**这轮暴露的是当前系统对新算法、新输入结构的履约泛化边界，尚不能用于训练门控选择分解。** 新任务同时改变算法与数据结构，本实验不是单独隔离“复杂度”的因果消融。两模式都失败的状态不强造ATOMIC标签；唯一原始单次成功本就低于方向判定要求，且已被契约反例否定。

下一步先做一个小型执行器对照，确认新任务失败是否与最近的专门化SFT有关：当前compute_sft与其训练前的handoff_sft，在这4个任务上固定同样的完整parent_aggregate计划，各2个相同种子，共16条自主执行。根门控与正文保持，原始和追加反例均纳入评估。这个对照尚未执行；不能提前断言SFT损伤了泛化。若两者都不足，再用多结构TRAIN课程改善完整算法与保留职责履约，而非继续在失败反馈上增加门控预算。

原先简单任务的执行改善仍成立，保留当前候选与旧执行器；本轮不撤销已验证能力，也不宣称递归无效。四个新任务现为已用开发集。

产物：[原始32条结果](../runs/refinement/complex_values/result.json)、[验证记录](../runs/refinement/complex_values/validation.json)。复现入口`python -m audittrainer.refinement_complex_values prepare/run/report`，run使用`--index 0`至3；`edge_order`为追加的离线反例核查，独立输出且不覆盖原始结果。
