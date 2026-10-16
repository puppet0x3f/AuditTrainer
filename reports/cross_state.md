# 冻结候选的八状态跨契约验证（已完成）

本轮不训练，只检验已固定的transfer_gate门控、root_retention执行器及MULTI_BODY在新内部契约上是否仍能获得和调用分解收益。模型实验已由用户完成；本次仅检查已有结果与代表轨迹，未运行新的模型实验。

## 结果与判断（2026-09-22）

| 新契约 | ATOMIC | SPLIT首次尝试 | 自然门控 | P(SPLIT) | 自然首次S次数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 路径小于阈值 | 1/4 | 4/4 | 4/4 | 83.5% | 4 |
| 路径等于阈值 | 4/4 | 4/4 | 4/4 | 53.1% | 4 |
| 库存有符号余额 | 0/4 | 0/4 | 0/4 | 59.3% | 1 |
| 库存状态分类 | 0/4 | 0/4 | 0/4 | 62.2% | 3 |
| 恰好一次覆盖 | 0/4 | 0/4 | 0/4 | 79.8% | 3 |
| 覆盖缺口 | 0/4 | 0/4 | 0/4 | 94.7% | 3 |
| 多项式符号 | 0/4 | 0/4 | 0/4 | 77.7% | 2 |
| 多项式值与导数之和 | 0/4 | 1/4 | 1/4 | 65.1% | 3 |
| 合计 | **5/32** | **9/32** | **9/32** | — | **23/32** |

自然根成功15.625%→28.125%，相对A增加12.5个百分点，配对4胜0负28平（三条路径小于阈值、一条多项式求和）。自然平均token31,952.8，较A22,613.9增加41.3%；较强制S35,152.7减少9.1%。自然与强制S总成功数相同，不能据此声称捕获所有可能分解收益。

96条均COMPLETE，无编排/评分错误，32组三臂初始状态/根计划/种子一致，四套模型保持，0更新。协议拒绝A30次、S35次、自然35次，均计入结果。S首次31/32注册，无同契约委托。coverage_gaps中的一份UNCONFIRMED实际已注册，为prepare→aggregate流水线，requires标点与参考模板不同导致精确匹配未命中；不能把UNCONFIRMED都视为非法计划。polynomial_sum另一次首次计划确实解析失败，原始标签保留。

**已有跨契约正迁移：新路径契约存在分解收益，当前门控能够调用并获得根成功。** 但证据高度集中：8个自然成功来自两个路径任务，其他6任务合计仅1/24；其中五个任务所有模式零成功。只有路径小于阈值达到预设探索性方向标准；多项式求和差1/4仍为不确定。

门控对失败任务也有较高P(S)，例如覆盖缺口94.7%。因此尚未证明它可靠区分新契约的分解价值；也不能由执行双零断言递归无效或应永久停止。路径等于阈值两臂均4/4，自然四次均S，成本约为A的1.67倍，显示过度分解成本。BODY先验概率不是旧独立门控对照；本轮没有旧门控执行臂，不能独立归因“门控校准造成跨任务增益”。

### 代表失败证据

以下定位针对所列样本，不是对全部失败的完整归因：

- stock_balance/SPLIT/repeat_00：聚合引用未定义stocks/targets并错误遍历SKU列表，最终abstain。
- stock_status/SPLIT/repeat_00：父聚合已按低于/等于/高于分类；prepare却删除输入事件，并将三元事件按二元解包构造rejected，根私有测试报ValueError。并非所有故障都在新聚合。
- coverage_single/SPLIT/repeat_00：将单覆盖长度加、多覆盖长度减，再用除2构造两项输出，未分别累计。
- coverage_gaps/SPLIT/repeat_00：把segments三元组当二元组遍历，且混淆query/segment循环。
- polynomial_signs/SPLIT/repeat_00：把求值点当作系数下标，未求原多项式值，零符号和次数也错误。
- polynomial_sum/SPLIT/repeat_00：只计算一阶导数，忽略请求阶数；repeat_02父聚合计划则通过。

### 下一步建议

保留当前门控/正文/执行器候选，暂不追加门控训练，也不把五个双零任务做停止负标签。下一步先做一个小规模固定产物职责诊断：预先选stock_status、coverage_single、polynomial_signs各一条代表S轨迹，比较原代码、只替换prepare、只替换aggregate、两者替换，共12个离线程序条件；保留其他代码并检查局部与最终根。缺失根产物时明确标为功能重建，不冒充自主恢复。人工替换只作因果诊断，不计模型成功。

确认具体缺口后，在独立TRAIN多契约示例上训练当前契约履行与输入结构处理，并保持原成功任务；不要逐条把这8个开发任务全部拟合完再称泛化。训练或调参使用过的状态归入开发范围，另留新状态确认；执行器变化后重新估A/S，再决定门控校准。下一实验尚未准备或启动，按用户要求需先核对设计和启动配置再给命令。

- [完整结果](../runs/refinement/cross_state/result.json)
- [配对与冻结检查](../runs/refinement/cross_state/validation.json)

## 预先固定的任务与范围

| 序号 | 新状态 | 准备→聚合计算 |
| --- | --- | --- |
| 0 | routes_below | 最短路径→严格小于阈值的节点、数量及距离总和 |
| 1 | routes_exact | 最短路径→恰等于阈值的节点、数量及距离总和 |
| 2 | stock_balance | 顺序处理库存事件→全部SKU的有符号余额 |
| 3 | stock_status | 顺序处理库存事件→低于/等于/高于目标的SKU |
| 4 | coverage_single | 带重数区间扫描→恰好一次及多次覆盖长度 |
| 5 | coverage_gaps | 带重数区间扫描→未覆盖及已覆盖长度 |
| 6 | polynomial_signs | 多项式乘积→原函数值符号及原次数 |
| 7 | polynomial_sum | 多项式乘积→原函数值与指定阶导数值之和 |

八个聚合契约均不同于本轮门控训练的四个契约，且不在当前执行器监督语料中。准备职责和四个算法族是已知的。输入由既有边界模板变换得到，不声称独立输入分布或未见算法泛化。这里覆盖四类语义计算组合，不代表预先强制四种分解拓扑；实际责任分配由冻结BODY新采样。

每状态6组确定输入（2可见、4私有）。覆盖空集合、等值边界、重边/零环、事件同序/拒绝、零目标、区间重叠/空查询、零多项式/空乘积/求导边界。多项式输入统一满足degree与derivative为0..4、系数-3..3。参考实现与独立oracle核对；作者递归程序CPU执行通过可见/私有根测试。这些是评分配置校验，不是模型成绩。

## 96条模型评估

8状态×3臂×4新种子＝96条，0参数更新：

- ATOMIC：首次compute强制直接实现。
- SPLIT：首次compute强制尝试冻结BODY新生成的分解。
- NATURAL：首次compute由transfer_gate自然选择。

每组三臂共享种子与模式抽样操作、初始内部观察和已注册根计划；内部ATOMIC仍保留根分解。后续节点、协议重试保持原策略。首次S失败后转A也完整计入，单独记录首次注册与模式，不以最佳计划替代平均收益。

主指标为最终根私有成功；附带首次SPLIT概率/次数、计划结构、同契约委托、token成本、拒绝及错误。差值≤1/4保留不确定，≥2/4仅作为探索性方向；评分错误不构成方向标签。汇总前检查32组三臂配对和四套模型冻结。

结果分三类决定后续：有新分解收益但门控未调用→扩充TRAIN收益覆盖；门控调用且自然收益改善→独立确认；两臂普遍失败→先处理执行缺口。不得据结果删任务。这里没有旧门控臂，因此不能单独证明“本次校准比旧门控更能泛化”。

## 启动配置与并行代码

- 工作目录 `/home/lc/AuditTrainer`，conda环境 `audittrainer`。
- Qwen3-8B固定revision，本地4bit/bfloat16加载，关闭thinking。
- 门控：`runs/refinement/transfer_gate/training/final_adapter`。
- 执行器：`runs/refinement/root_retention/training/final_adapter`。
- 根门控：`runs/refinement/gate_refresh/training/final_adapter`。
- 正文：`runs/refinement/body_transfer/MULTI_BODY/final_adapter`。
- 执行种子583000713，每状态递增100，每重复递增1。
- 2个独立进程自动预约GPU 0/1，每进程至少16000 MiB空闲；这是评估并行，不是DDP训练。不另设CUDA_VISIBLE_DEVICES。
- worker0负责0/2/4/6，worker1负责1/3/5/7；每状态三臂组内顺序执行，防止重复写目录。worker失败不生成最终汇总。
- 已完成case使用现有缓存；中断在case内部时需先检查其日志和未完成目录，不盲目覆盖重跑。原有GPU预约只在本实验内互斥；避免同时运行其他占用相同GPU的大实验。

**原评估启动命令（本轮已完成，无需重跑）：**

```bash
cd /home/lc/AuditTrainer
bash scripts/run_cross_state_parallel.sh
```

脚本先检查配置，再并行完成8状态，最后自动汇总。可仅做CPU配置检查：

```bash
conda run --no-capture-output -n audittrainer python -m audittrainer.refinement_cross_state check
```

查看日志：

```bash
tail -f runs/refinement/cross_state/run_0.log
```

最终结果：`runs/refinement/cross_state/result.json`、`validation.json`；分状态日志为`run_0.log`至`run_7.log`。模型结果已完成，见结果分析。

## 启动前检查结果

19项CPU测试通过，8个作者参考程序通过原生评分，shell语法检查通过。检查覆盖新语义手算、reference/oracle一致、数据范围、契约与监督分离、三臂干预规则及固定模型配置。模型评估现已由用户完成。

## 文件

- [并行启动脚本](../scripts/run_cross_state_parallel.sh)
- [任务定义](../src/audittrainer/refinement_cross_state_tasks.py)
- [评估入口](../src/audittrainer/refinement_cross_state.py)
- [固定配置](../runs/refinement/cross_state/request.json)
- [CPU测试记录](../runs/refinement/cross_state/tests.log)
