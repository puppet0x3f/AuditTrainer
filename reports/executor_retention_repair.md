# 执行器保持能力修复：设计与启动配置

## 目的

上一轮 executor pilot 把新输入根成功从 2/8 提升到 6/8，但 retention 从 16/16 降到 13/16。退步集中在库存状态与复杂库存事件，并出现 `INVALID_ACTION_JSON`、`ARTIFACT_VERSION_MISMATCH` 和 `UNVERIFIED_CONSUMPTION`。本轮只修复执行器保持能力，不训练规划器、根门控或 `MULTI_BODY`。

## 固定设计

- 起点是已完成 pilot 的 AFTER adapter：`runs/refinement/executor_pilot/training/final_adapter`。
- 复用 pilot 的 20 条新契约成功目标作为 anchor，避免修复时丢失新能力。
- retention 训练集固定为 40 条：库存状态与库存事件各 9 条经过 review 的责任轨迹，加 22 条 pilot 保持样本。库存轨迹来自独立 TRAIN 根，不使用本轮 validation private 标签。
- 只更新 executor LoRA，冻结 gate、BODY、planner decisions、测试集合和评估种子。
- 24 步、学习率 `2e-6`；每步 2 条新契约 anchor + 2 条 retention，合计各 48 次曝光。单一最终 checkpoint，不按评估结果挑选。
- 评估复用 pilot 的 32 条配对案例（fit/new/retention 各 8/8/16），pilot AFTER 结果作为已完成的固定 baseline；修复 adapter 的 3 个分片在 GPU 0/1/2 上并行运行，分片和训练均带 `tqdm`。

只有 retention 逐根不退步、至少恢复一个 baseline 失败，且 fit/new 不退步时，才允许下一步重采 ATOMIC/SPLIT 价值。否则继续停留在执行器修复，不构造条件规划器数据。

## 启动前核对

已完成 CPU 准备与 `READY` 检查：20 条 anchor、18 条定向 retention、22 条广泛 retention、24 步、32 条配对验收；`py_compile`、`bash -n` 及相关 5 项测试通过。检查过程未加载模型。

## 用户启动命令

```bash
cd /home/lc/AuditTrainer
bash scripts/run_executor_retention_repair.sh
```

脚本会先校验固定请求和 pilot baseline，训练完成后启动三个并行评估分片，最后写入 `runs/refinement/executor_retention_repair/result.json`。已有 `training` 目录时脚本拒绝覆盖。

- [实现](../src/audittrainer/refinement_executor_retention_repair.py)
- [启动脚本](../scripts/run_executor_retention_repair.sh)
- [固定请求](../runs/refinement/executor_retention_repair/request.json)
