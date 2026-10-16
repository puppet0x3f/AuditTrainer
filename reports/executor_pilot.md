# 责任约束执行器 pilot：设计与结果

## 已完成结果（2026-10-06）

汇总入口曾把每个方法的 `evaluation_cases=32` 错当成 BEFORE/AFTER 合计数，
因此用 `// 2` 期待 16 条并在所有分片完成后触发断言。已改为按方法检查 32 条；
六个分片无需重跑，离线报告已成功生成。

| 组别 | BEFORE 根成功 | AFTER 根成功 | AFTER 局部 private |
| --- | ---: | ---: | ---: |
| fit | 0/8 | 6/8 | 6/8 |
| new | 2/8 | 6/8 | 6/8 |
| retention | 16/16 | 13/16 | 13/16 |

配对结果为 10 胜、19 平、3 负；保持集逐根不退步条件未通过，
`proceed_to_value_refresh=false`。保持集退步集中在
`validation_cross_state_stock_status`（2/2 失败）和
`validation_internal_complex_stock_events`（1/2 失败），其中两条出现
`INVALID_ACTION_JSON`/`ARTIFACT_VERSION_MISMATCH`，另一条为
`UNVERIFIED_CONSUMPTION`。因此这不是漏跑或评分器错误，而是新执行器提升新契约的同时损害了保持能力。

门控校准在开发状态上已经显示可学习，但旧跨契约集合随后被执行器课程使用，不能再作为独立泛化集。本轮冻结根门控和 `MULTI_BODY`，从当前 `contract_correction` 执行器复制一份，只用经过 reference、visible/private 测试和父消费核验的责任执行轨迹补充输入结构与输出语义。

## 数据与目标

四个新输出契约分别覆盖库存有符号差、区间覆盖直方图、多项式原值/一阶导数成对返回、路径距离直方图。每类打包六个独立 TRAIN 输入；每个任务各保存 atomic 与 fused-root/pipeline 两种成功 teacher 轨迹，最终收集 20 条 `write_code` 目标。每条训练目标固定曝光 8 次。

保持集从当前 `contract_correction` 训练语料按哈希固定抽取 40 条，只作为 retention 目标，避免训练新职责时遗忘已有能力。训练为 80 步、学习率 `1e-5`，每步 2 条新职责目标 + 2 条保持目标；单一最终检查点，不按中途结果挑选。

验证集使用同四个新输出契约的另一组输入，输入与训练集分离；另有 8 个当前能力保持根。BEFORE 使用当前执行器，AFTER 使用 pilot 适配器，根计划、门控、正文、测试和种子配对固定。报告局部 private 通过、prepare/aggregate 父消费、最终根成功、拒绝/错误和 token 成本。只有新输入的局部及根成功提升且保持集逐根不退步，才继续重采当前执行器下的 ATOMIC/SPLIT 价值。

## 执行前核对

已完成 CPU 准备和 `READY` 检查：20 条新目标、40 条保持样本、80 步、32 条配对验收；reference/private suites 由 `pack` 与独立 oracle 交叉核验。训练数据和 adapter 哈希写入固定请求，随后训练与评估按此配置完成。

训练仍是单卡 QLoRA，GPU 并行用于评估：脚本同时启动训练和三个 BEFORE 分片，由预约器在 GPU 0/1/2 中选择空闲卡；训练结束后再运行三个 AFTER 分片。每个训练/评估进程有 `tqdm` 进度条和独立日志。显存要求训练至少 22,000 MiB，评估至少 16,000 MiB。

## 复现启动命令

```bash
cd /home/lc/AuditTrainer
bash scripts/run_executor_pilot.sh
```

脚本会先做不加载模型的配置检查；已有 `training` 目录时拒绝覆盖。训练或任一 BEFORE 分片失败时不会启动 AFTER，任一 AFTER 分片失败时不会汇总。结果写入 `runs/refinement/executor_pilot/result.json` 和 `validation.json`。

- [实现](../src/audittrainer/refinement_executor_pilot.py)
- [启动脚本](../scripts/run_executor_pilot.sh)
- [固定请求](../runs/refinement/executor_pilot/request.json)
