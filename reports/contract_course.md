# 多契约执行履约SFT：配对验收结果

截至2026-09-23：用户已完成216步训练和112条新旧执行器配对续跑；本节分析实际结果。以下原设计与启动命令仅供复现，不表示需要重新启动。

## 结果与判断

训练完成，目标/保持各432次曝光；最终执行器与起点参数不同，磁盘重载前向最大差值为0，原执行器文件保持不变。四个评估分片各28/28完成，112条运行无编排或评分错误；56组的首次实现观察一致。门控、BODY与固定计划未改变。

| 组别 | BEFORE根私有通过 | AFTER根私有通过 | 配对胜/负 |
| --- | ---: | ---: | ---: |
| TRAIN课程 | 8/12 | **12/12** | 4/0 |
| 已监督契约、独立输入开发 | 0/12 | **12/12** | 12/0 |
| 本轮未监督契约开发 | 2/12 | **0/12** | 0/2 |
| 旧任务保持 | 20/20 | **20/20** | 0/0 |
| 合计 | 30/56 | **44/56** | 16/2 |

三个已监督开发契约（库存状态、单覆盖、原函数符号）各从0/4升到4/4；ATOMIC和固定父聚合两种结构均从0/6升到6/6。compute局部私有通过从0/12升到12/12，根实际调用从7/12升到12/12。说明执行器学会在不同输入上履行**已经教过的契约**，且能用于两种责任结构；不能据此宣称未见契约的迁移。

未监督契约分别是库存余额、覆盖缺口、多项式求和；前两者两臂均0，后者固定父聚合从2/2降为0/2。两条损失的prepare均通过私有局部测试，AFTER的compute局部失败，根没有消费/提交程序。保存代码显示AFTER未按`derivative`计算导数：一次把`points`中的x本身当作原值相加，一次把原函数值直接乘2；BEFORE均正确计算原值与指定阶导数值之和。这是已定位的聚合语义退步，不是门控、正文、交接或评分错误。它与同算法族训练中大量原值/导数目标的干扰相容，但单次SFT无法证明具体训练样本的因果来源。

预设字段`proceed_to_fresh_values=true`按“已监督开发提升且旧保持根不退步”计算，因此按**原验收规则**通过；该规则没有将未监督契约退步列为否决项。科学判断应更细：保留本轮执行器作为已监督契约上的候选，不直接替换通用执行器，也不把44/56作为普遍收益。当前固定计划验收不等于新采样SPLIT平均价值或自然门控收益。

下一步优先做一次小规模跨契约纠偏：在独立TRAIN输入上为同一prepare增加输出语义对照，特别覆盖“原值+指定阶导数值”及库存余额、覆盖缺口；旧保持监督继续保留。将这三个已用开发契约转为开发/诊断范围，另预留新的输出契约作为确认。新旧执行器仍用固定计划、同状态同种子配对验收，同时要求已教契约和旧根保持，再决定是否重采A/S和更新门控。不必立刻加大训练步数或更换RL方法。

实际汇总见[机器结果](../runs/refinement/contract_course/result.json)，训练身份与重载见[训练结果](../runs/refinement/contract_course/training/result.json)，配对一致性见[验证记录](../runs/refinement/contract_course/validation.json)。

截至2026-09-22：训练数据、配置、双GPU编排与进度条已准备；仅进行了CPU参考程序评分与静态检查，没有加载模型、训练或运行新旧模型评估。

## 固定设计

从root_retention复制执行器候选，保持transfer_gate、根门控、MULTI_BODY及旧执行器不变。本轮训练执行器按当前契约完成准备、聚合和根组合，不训练门控。

| 准备算法 | TRAIN聚合契约一 | TRAIN聚合契约二 | 不入训的已用开发契约 |
| --- | --- | --- | --- |
| 库存事件处理 | below/equal/above分类 | 缺货补货量 | 全部SKU有符号余额 |
| 区间重数扫描 | 恰好一次/多次覆盖长度 | 重复覆盖/超额覆盖/峰值 | 未覆盖长度 |
| 多项式乘积 | 原函数值符号与次数 | 原值/指定阶导数值/原次数 | 原值与导数值之和 |

每对TRAIN契约使用相同的六组新输入，突出当前契约的输出差异；输入不同于开发用例，未使用失败模型轨迹或人工修正轨迹作为监督。库存覆盖零目标、事件拒绝和顺序；区间覆盖重复区间和空查询；多项式覆盖空乘积、零因子、重复项抵消和不同求导阶数。

每个契约由作者程序生成三种结构的完整write_code示范：内部ATOMIC、parent_aggregate、pipeline，根均保留既定分解。6契约×(2+3+4)＝**54条目标**，包括完整compute、prepare、父聚合、aggregate子函数和根组合。另保留root_retention全部**96条旧监督**（包括此前新契约和别名保持）。18个作者程序已通过CPU可见/私有根评分，不计模型成绩。

固定**216步、学习率1e-5**，每步2目标＋2保持，等权token均值CE。每条目标8次曝光，目标/保持各432次；保持样本各4或5次。训练种子591000713，最终检查点唯一，不按验收挑选中途checkpoint。沿用现有NF4/bfloat16单卡训练、梯度检查点与重载检查；新增progress开关仅改变显示，旧调用默认行为不变。

## 112条自主配对验收

BEFORE使用冻结root_retention；AFTER使用本轮最终执行器。两臂重新运行，同状态、固定计划、同种子，拒绝/重试/失败全部保留。执行器自主实现、消费与组合，不替换为作者代码。

| 组别 | 每个执行器的条数 | 说明 |
| --- | ---: | --- |
| TRAIN课程 | 12 | 6契约×2种子；三个新契约使用ATOMIC，三个对照契约使用父聚合 |
| 已监督契约开发 | 12 | 状态分类/单覆盖/符号，A与固定父聚合各2种子 |
| 未监督契约开发 | 12 | 余额/缺口/求和，A与固定父聚合各2种子 |
| 旧能力保持 | 20 | 10个根×2种子：原四复杂任务、calendar/chunks、两项转换、重复覆盖/联合求值 |
| 合计 | **56** | 两执行器共112条 |

评估种子593000713；每臂按任务索引分两个分片，各28条。检查根私有成功、compute局部通过、可用prepare局部通过、根实际调用、token成本、配对胜负与原参数保持。内部ATOMIC无独立prepare产物，不把缺少prepare产物当作局部失败。

已监督契约使用独立TRAIN开发副本，原文件和测试保持。三个未监督契约此前已经用于开发诊断，仅表示本轮监督中未出现，**不是新的独立盲测**。本轮最多支持已知算法族中的契约迁移；未来正式确认仍须另留新契约。

验收条件：已监督开发根成功高于旧执行器，且每个旧保持根不退步、无评分错误，才建议重采A/S。未监督契约效果单独报告，未改善就不能宣称跨契约学习成功。课程或开发未达到预期时按失败分析，不自动增加步数。这里的分解计划固定，不代表BODY新采样S的平均收益，也不是自然门控成绩；固定计划期间独立内部门控不参与选择，保持其文件不变。

## 双GPU启动与进度条

- 环境：`audittrainer`，目录 `/home/lc/AuditTrainer`。
- 底座：本地Qwen3-8B，固定revision、4bit加载、bfloat16、关闭thinking。
- 初始执行器：`runs/refinement/root_retention/training/final_adapter`。
- 保持的独立内部门控：`runs/refinement/transfer_gate/training/final_adapter`。
- 根门控与BODY沿用gate_refresh和MULTI_BODY。
- **仅GPU 0和1**；训练至少22000 MiB空闲，评估至少16000 MiB。自动预约互斥，不设置CUDA_VISIBLE_DEVICES。
- 不是DDP训练：第一阶段单卡训练与另一卡旧模型基线评估并行；二者完成后，第二阶段双卡并行跑AFTER两个分片；最后自动汇总。
- 训练显示“执行器SFT 0/216”及当前loss；每个评估分片显示“BEFORE/AFTER 分片N 0/28”。各自日志保留进度，完整训练history在训练result.json。

**推荐完整启动命令（尚未执行）：**

```bash
cd /home/lc/AuditTrainer
bash scripts/run_contract_course.sh
```

此命令包含训练及全部112条验收。若只想先启动训练：

```bash
cd /home/lc/AuditTrainer
set -o pipefail
conda run --no-capture-output -n audittrainer \
  python -u -m audittrainer.refinement_contract_course train \
  2>&1 | tee runs/refinement/contract_course/train.log
```

两种方式选一种，不重复启动。完整脚本拒绝覆盖已有training目录；单独训练结束后，需要以下评估命令，而非重跑完整脚本：

```bash
cd /home/lc/AuditTrainer
for method in BEFORE AFTER; do
  pids=()
  for shard in 0 1; do
    conda run --no-capture-output -n audittrainer \
      python -u -m audittrainer.refinement_contract_course evaluate \
      --method "$method" --shard "$shard" \
      > "runs/refinement/contract_course/${method}_${shard}.log" 2>&1 &
    pids+=("$!")
  done
  status=0
  for pid in "${pids[@]}"; do wait "$pid" || status=1; done
  if [ "$status" -ne 0 ]; then exit 1; fi
done
conda run --no-capture-output -n audittrainer \
  python -m audittrainer.refinement_contract_course report
```

检查但不加载模型：

```bash
conda run --no-capture-output -n audittrainer \
  python -m audittrainer.refinement_contract_course check
```

结果为 `runs/refinement/contract_course/result.json`，训练结果为`training/result.json`；评估日志为`BEFORE_0.log`、`BEFORE_1.log`、`AFTER_0.log`、`AFTER_1.log`。原样本和模型不会被覆盖。进程失败时脚本停止后续阶段，需检查对应日志。

## 启动前检查

3项CPU单元测试通过：训练/开发输入分离、同准备不同聚合的成对数据、54目标等量曝光、96条保持完整、未监督契约无监督泄漏、18个参考程序评分、两GPU分片预算和进度接口。shell语法检查通过；配置检查为READY。首次准备发现的io_type和同族跨split登记问题均已在模型启动前修正。

- [实现](../src/audittrainer/refinement_contract_course.py)
- [双GPU完整启动脚本](../scripts/run_contract_course.sh)
- [固定配置](../runs/refinement/contract_course/request.json)
- [测试记录](../runs/refinement/contract_course/tests.log)
