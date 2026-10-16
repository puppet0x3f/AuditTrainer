# 新执行器收益驱动的独立内部门控校准（待用户运行）

截至2026-09-21：32步训练与32条自然执行已完成；本次只读取已有产物并运行离线汇总，没有启动新的模型实验。

## 已完成训练结果

| 状态 | 训练前P(SPLIT) | 训练后P(SPLIT) |
| --- | ---: | ---: |
| 路径 | 22.2700% | 99.9983% |
| 库存 | 29.4215% | 29.4215% |
| 重复覆盖 | 43.7824% | 40.7333% |
| 多项式 | 29.4215% | 99.9996% |

正收益状态调用概率显著上升；库存保持，重复覆盖降低3.05个百分点。四份正文生成逐token一致，根门控概率与根门控/BODY/执行器参数保持，重载差0。证明本轮已学会调用这两个已知正收益状态，同时隔离正文更新；尚不能证明自然根成功提高或跨任务泛化。

训练完成时预定下一步为32条BEFORE/AFTER新种子自然对照（现已完成，结果见下节），未追加训练或调整检查点。无需因重复覆盖约3个百分点漂移立即重训；应看自然成功、分解位置与成本。训练后的接近100%来自强正价值差和小KL系数，不是独立可靠性置信度。

[训练结果](../runs/refinement/transfer_gate/training/result.json)

## 已完成自然执行与分析

| 状态 | BEFORE根成功 | AFTER根成功 | 首次SPLIT次数 |
| --- | ---: | ---: | ---: |
| 路径 | 1/4 | **4/4** | 1→4 |
| 库存 | 1/4 | **0/4** | 2→2 |
| 重复覆盖 | 4/4 | **4/4** | 2→2 |
| 多项式 | 2/4 | **3/4** | 3→4 |
| 合计 | 8/16 | **11/16** | 8→12 |

根成功50%→68.75%，增加18.75个百分点；配对4胜1负11平。平均轨迹token 27,992.875→30,333.3125，增加8.36%。四条新增成功均对应路径/多项式从ATOMIC改选SPLIT；两保持状态首次选择逐种子未变。32条无评分或编排错误，16组初始内部观察/根计划/种子一致；四份BODY逐token保持，根门控/BODY/执行器参数冻结，重载差0。

**本轮已经把已知正收益状态的调用学习转化为自然根成功收益，值得保留transfer_gate作为候选。** 样本为四个已用开发状态，不声称跨任务泛化或全递归规划完成。

### 库存唯一配对退步

repeat_02两次均选择SPLIT、相同首次计划与准备代码。首次父聚合生成前，观察中的剩余token分别179536/179537，仅此字段出现差异；随后生成不同聚合代码。BEFORE首次源码缩进错误，协议重试后生成正确代码并成功；AFTER按target是否非零筛选库存，最终失败并abstain。该条必须计为实际配对退步，但不支持“门控降低了库存分解概率导致失败”或“冻结执行器参数被训练损坏”。现有轨迹不足以断言一个token差异的具体来源及其单独因果效应，不将同种子等同于后续逐token执行相同。

### 多项式剩余失败

repeat_03在BEFORE/AFTER均首次选择SPLIT，均JSON解析拒绝后由原继续策略改为ATOMIC，随后生成错误的联合求值代码并失败。不是已注册分解代码失败，也不是门控未调用SPLIT。提高首次概率不能自动消除计划格式和重试回退问题。

### 下一步建议

结束当前四状态上的门控拟合，冻结transfer_gate、root_retention与BODY，不继续加训练步数。下一项高信息量实验是跨状态诊断：预选约8个未用于本轮训练和开发评估的新内部状态、至少4种责任结构，每状态强制A/新采样S/当前自然门控各4种子，共96条。仍固定根计划，纳入全部拒绝、重试、失败和成本；先判断新状态是否存在分解收益，再判断门控是否调用，不按结果挑任务。明确区分同契约新输入与未训练契约，本轮验证不了未见算法泛化。

库存执行与多项式格式重试保留为局部问题，不扩成新的大规模执行器课程、不覆盖本轮失败。用户授权准备下一实验时，再冻结任务清单、核对启动配置并给命令，由用户运行；此处未启动或创建下一轮实验。

- [自然执行汇总](../runs/refinement/transfer_gate/result.json)
- [配对与冻结检查](../runs/refinement/transfer_gate/validation.json)

## 设计核对

使用transfer_refresh的32条当前策略执行结果：路径和多项式A0/4、S4/4；库存双0；重复覆盖双4。仅校准首次compute门控，从**当前internal_gate**复制适配器，不回退BODY。冻结root_retention执行器、根门控、MULTI_BODY；后续内部决策及协议重试仍沿用估值时策略。

目标为 `sigmoid(logit(p_old) + (Q_S-Q_A)/0.1)`；以当前内部策略概率为KL参考，零成本权重，与当前成功收益一致。不是将执行成功率直接设为门控概率。平局、不确定或评分异常状态保持原概率，不构造强负标签。

| 状态 | 当前P(S) | 监督目标P(S) | 用途 |
| --- | ---: | ---: | --- |
| 路径阈值外查询 | 22.2700% | 99.9842% | 正收益调用 |
| 库存正剩余 | 29.4215% | 29.4215% | 保持，不把双失败作负标签 |
| 重复覆盖 | 43.7824% | 43.7824% | 保持，不强推分解 |
| 多项式联合求值 | 29.4215% | 99.9891% | 正收益调用 |

- 固定32步，学习率1e-5，每步四状态等权，128次曝光，训练种子571000713。只使用最终检查点，不按验收成绩选点。
- 训练前后各做四份同种子BODY生成检查，验证正文逐token保持；检查根门控概率、执行器/BODY参数保持和检查点重载。
- BEFORE/AFTER自然执行各4状态×4新种子，共32条。两臂都使用root_retention执行器，仅首次内部门控不同；自然种子573000713，与价值采集分离。固定根计划，所有拒绝、重试和失败保留。
- 主要看自然根成功、正收益状态的分解调用、配对胜负；同时看两保持状态的概率漂移和token成本。结果包含评分错误时单独列出，不当作训练负回报。
- 本轮为已用开发状态上的有限校准，不是跨任务泛化或根/内部全自然联动。没有把旧执行器下的其他状态收益合并为新标签；本轮保持检查也只覆盖列出的两个状态。

重复覆盖S臂含一次首次拒绝后转ATOMIC成功，目标使用完整策略收益；不会将其伪装为四份分解计划均成功。路径/多项式的收益差来自各4种子，仅作本轮校准依据，必须再用自然执行检验实际收益。

## 启动配置

- 环境：conda `audittrainer`；工作目录 `/home/lc/AuditTrainer`。
- 底座：本地Qwen3-8B，固定revision，4bit加载，bfloat16，关闭thinking。
- 初始内部门控：`runs/refinement/internal_gate/training/final_adapter`。
- 执行器：`runs/refinement/root_retention/training/final_adapter`。
- 根门控：`runs/refinement/gate_refresh/training/final_adapter`。
- 正文：`runs/refinement/body_transfer/MULTI_BODY/final_adapter`。
- GPU：自动预约0/1/2，训练至少22000 MiB空闲，评估至少16000 MiB；不手动设置CUDA_VISIBLE_DEVICES。
- 启动前自动核对源估值、状态和原适配器哈希。已有training目录时拒绝覆盖；本入口不是断点续训。

训练启动命令（**本次已完成，无需重跑**）：

```bash
cd /home/lc/AuditTrainer
set -o pipefail
conda run --no-capture-output -n audittrainer python -u -m audittrainer.refinement_transfer_gate train 2>&1 | tee runs/refinement/transfer_gate/train.log
```

训练完成标志为 `runs/refinement/transfer_gate/training/result.json`。训练后自行运行自然对照与汇总（下列顺序版命令未执行）：

```bash
cd /home/lc/AuditTrainer
for method in BEFORE AFTER; do
  for shard in 0 1 2; do
    conda run --no-capture-output -n audittrainer python -u -m audittrainer.refinement_transfer_gate natural --method "$method" --shard "$shard" > "runs/refinement/transfer_gate/natural_${method}_${shard}.log" 2>&1 || exit 1
  done
done
conda run --no-capture-output -n audittrainer python -m audittrainer.refinement_transfer_gate report
```

仅配置检查，不加载模型：

```bash
conda run --no-capture-output -n audittrainer python -m audittrainer.refinement_transfer_gate check
```

## 文件

- [实现](../src/audittrainer/refinement_transfer_gate.py)
- [固定配置](../runs/refinement/transfer_gate/request.json)
- [GPU配置](../runs/refinement/transfer_gate/gpu_resources.json)
- [单元测试](../tests/test_refinement_transfer_gate.py)
- [测试记录](../runs/refinement/transfer_gate/tests.log)
