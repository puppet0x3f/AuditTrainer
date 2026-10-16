# 固定产物的准备/聚合职责隔离诊断（已完成）

截至2026-09-22：用户已完成12个离线评分条件，本次核对已有结果。全部条件评分正常，0模型调用、0参数更新；原轨迹、成绩及适配器保持。本次没有启动新实验。

## 实测结果与下一步

| 样本 | 原始 | 只替换prepare | 只替换aggregate | 联合替换 |
| --- | --- | --- | --- | --- |
| 库存状态 stock_status | FAIL | **PASS** | FAIL | **PASS** |
| 单覆盖 coverage_single | FAIL | FAIL | **PASS** | **PASS** |
| 多项式符号 polynomial_signs | FAIL | FAIL | **PASS** | **PASS** |

表中可见与私有根测试结论一致，恢复条件的compute局部私有也通过，根实际调用compute。库存原prepare局部失败，替换后prepare、compute与原根全部恢复；其父聚合和根代码保留即可成功。单覆盖、多项式原prepare已通过，参考替换为源码无变化；只有修正聚合后compute及根恢复。

**三个代表失败均能通过单个计算职责替换恢复功能，当前主要瓶颈是执行器按当前契约实现计算，而不是这些样本必须重新分解或增加层级。** 库存为已知准备算法未被稳定履行；单覆盖/多项式为新聚合语义未被正确实现。不要把库存也统一归为父聚合错误。

库存与多项式保留原根，说明替换计算后原组合链可用。单覆盖四条件均使用同一人工根包装，只有功能恢复证据，尚未证明模型会自主生成根代码、消费与完成交接。全部人工替换均不计模型成绩，也不能把三条代表样本推广为全部失败已解释。

**下一步建议：结束这三条失败的可修复性诊断，转小规模多契约执行履约SFT。** 从root_retention复制候选执行器，冻结transfer_gate、根门控和BODY，保留原执行器对照。训练重点分别是库存事件处理/拒绝索引/不改输入，以及区间和多项式聚合严格按当前契约计算。以同一准备结果对应不同聚合契约的成组TRAIN示例形成对照；同时监督ATOMIC完整compute和分解父节点的保留聚合，避免只会分解后执行。

保留已获得的路径、原覆盖/多项式、转换与根组合能力。训练预算及最终检查点预先固定；不把这三份人工修正轨迹直接作为训练数据，也不将8个开发任务逐条背熟后称泛化。对训练契约另选输入只能检验同契约迁移；跨契约确认必须另留未监督聚合契约。

先以固定计划下的模型自主生成、消费、组合验收局部与根成功（尤其补上单覆盖缺根环节），同时检查旧任务保持；通过后再用同一新执行器重采A/S，旧估值不复用。当前不再加训门控。下一轮训练代码/预算尚未准备，按用户要求应先核对设计与启动配置、提供命令后由用户运行。

- [完整结果](../runs/refinement/cross_repair/result.json)

## 固定设计

| 样本 | 原始强制S轨迹 | 原根是否存在 | 本轮根处理 |
| --- | --- | --- | --- |
| stock_status | repeat_00，FAIL | 是 | 原根代码不变 |
| coverage_single | repeat_00，NO_PROGRAM | 否 | 所有条件添加同一个固定JSON解析/compute调用/序列化包装 |
| polynomial_signs | repeat_01，FAIL | 是 | 原根代码不变 |

多项式使用repeat_01而非repeat_00：repeat_01已有完整的父聚合结构，和其他两个样本一致，可以只替换prepare与父节点_aggregate；repeat_00为流水线且缺少可用父产物，需额外重建，无法保持同样的干预范围。样本选择基于已有失败和可隔离职责，不根据本轮评分挑选。

每样本四条件，共 **12个离线程序条件**：

1. ORIGINAL：原代码（缺根样本使用公共包装）。
2. PREPARE_ONLY：只将prepare函数替换为已冻结的作者参考实现。
3. AGGREGATE_ONLY：只将父compute中的_aggregate函数替换为参考实现。
4. BOTH：同时替换上述两个函数。

替换使用完整参考函数，不称逐行最小补丁。compute调用代码、其他函数、原有根代码、计划、契约、用例均保留。依赖别名与图保持，派生产物版本/哈希按新源码重新绑定，不绕过中间节点。单覆盖和多项式prepare的参考替换可能为无变化；报告同时列出指定替换节点与实际变化节点。

验收同时报告prepare局部私有、compute局部私有、最终根可见/私有结果及根实际调用。两者替换仍失败时记录结果，不自动追加修复。评分基础设施错误单独标识，不解释为功能失败。

**解释边界：** 本轮没有模型调用、没有自主consume重放，不计入模型成功率。两个已有根样本可判断保留原根时的功能恢复；缺根的单覆盖样本只能说明公共包装下的功能恢复，不能称自主交接恢复。三条代表失败不能估计总体故障占比。

## 启动配置

- conda环境：`audittrainer`。
- 数据来源：`runs/refinement/cross_state/result.json`及三条固定S轨迹。
- 参考函数：原任务冻结reference，不使用新模型生成。
- 评分：沿用原framework和原可见/私有用例，不改执行时限。
- 并行：3个CPU worker、12个独立条件目录，无GPU预约、无显存占用、无DDP。
- 进度：终端显示“离线职责诊断 0/12→12/12”，附最近完成样本/条件/根结果；进度表示完成评分，FAIL也是完成的实验结果。
- 原结果/轨迹/代码/契约/参考/测试与模型文件均作哈希核对。仅写新cross_repair目录。
- 为避免覆盖已有实验，`evaluations`目录已存在时拒绝启动；中断后先检查日志，不自动删除或覆盖产物。

## 启动命令

**本轮不需要训练命令；运行以下离线实验命令即可（本轮已完成，无需重跑）：**

```bash
cd /home/lc/AuditTrainer
bash scripts/run_cross_repair.sh
```

脚本先检查配置，再启动3个CPU进程，显示进度条并保存日志，最后自动汇总。若要串行诊断资源问题，可在首次启动时使用：

```bash
conda run --no-capture-output -n audittrainer \
  python -u -m audittrainer.refinement_cross_repair run --workers 1
```

仅检查配置，不评分：

```bash
conda run --no-capture-output -n audittrainer \
  python -m audittrainer.refinement_cross_repair check
```

结果位置：

- 总结果：`runs/refinement/cross_repair/result.json`。
- 日志：`runs/refinement/cross_repair/run.log`。
- 各条件原始评分：`runs/refinement/cross_repair/evaluations/<root>/<condition>/result.json`。
- 可审阅修改：`runs/refinement/cross_repair/<root>/*.diff`与`scenario.json`。

完成后先据PREPARE_ONLY/AGGREGATE_ONLY/BOTH的根恢复定位缺口，再决定独立TRAIN执行器课程。本轮结果如上；尚未启动后续训练。

## 文件

- [并行与进度条入口](../scripts/run_cross_repair.sh)
- [诊断实现](../src/audittrainer/refinement_cross_repair.py)
- [固定配置](../runs/refinement/cross_repair/request.json)
- [静态测试](../tests/test_refinement_cross_repair.py)
- [测试结果](../runs/refinement/cross_repair/tests.log)
