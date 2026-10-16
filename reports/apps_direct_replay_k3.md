# DIRECT_REPLAY 固定预算执行器评测

这一步先完成已有执行器数据策略的验证，再决定是否继续构造新的责任约束训练数据。训练已经完成，不重复训练：固定数据为 3,315 个 APPS/TACO 直接目标和 177 个 APPS replay 曝光，共 3,492 个样本、219 个优化步；适配器为 `runs/refinement/apps_direct_retention_sft/training/final_adapter`。

评测固定 100 个未见 TACO 任务，每题 3 个随机候选，分别保存原始 `solve(text)` 和标准输入 bridge 结果。指标是公开样例通过且 9 个私有测试全部通过；同时报告程序存在、公开/私有通过和候选异常。BASE、DIVERSE、TACO、DIRECT_CODE 的冻结结果由 `apps_taco_direct_k3` 引用，不重新生成。

本轮使用 GPU 0、1、2 分别处理 `index modulo 3` 的任务分片；每张卡 100 条新轨迹，模块内已有 tqdm 进度条。bridge 是 CPU 分片，不占用 GPU。结果和分片日志写入 `runs/refinement/apps_taco_direct_replay_k3/`。

首次运行没有形成结果文件：三个 GPU 分片都在遇到候选程序 `TIMEOUT` 后退出。候选超时是模型结果的一部分，不是分片未完成；评测入口原先把它误判为未完成候选。现在入口保留 `TIMEOUT` 和 `OUTPUT_LIMIT` 作为合法失败，并继续拒绝真正的 `SANDBOX_ERROR`、身份不匹配或损坏产物。已有 14 条候选结果会在重跑时复用，不需要删除。

```bash
bash scripts/run_apps_taco_direct_replay_k3.sh
```

启动前脚本会运行 `check`，不会覆盖已完成分片。若中途退出，可以重新运行同一命令；已完成的分片会被跳过，未完成的分片继续执行。当前不启动下一轮训练；先根据这次 replay 相对 DIRECT_CODE/DIVERSE/TACO 的严格私有通过率和 retention 结果决定后续数据构造。

## 结果

重跑完成，100 个 TACO 未见任务、每题 3 个候选均已汇总。replay 在原始接口和 bridge 接口的 selected strict k1/k3 都是 **6/100、12/100**；DIVERSE 冻结 adapter 是原始接口 **8/100、12/100**，bridge **11/100、13/100**。replay 的候选 oracle k3 是 **14/100**，与 DIVERSE bridge 相同，但没有增加新的可选成功根。

按同根 k3 配对，replay bridge 相对 DIVERSE bridge 为 **3 胜、4 负、93 平**；相对 DIVERSE ATOMIC 为 **3 胜、7 负、90 平**；相对 bridge BASE 为 **4 胜、16 负、80 平**。replay bridge 在 EASY 为 9/50、MEDIUM 为 3/50；bridge 对 replay 本身没有改变任何根的 selected strict 结果，说明这次主要限制是代码语义/算法正确性，不是 stdio 适配。

因此，177 条 APPS replay 曝光在 TACO 未见任务上没有证明泛化收益；继续增加同类 replay 或直接扩大混合数据量的优先级很低。随后完成的 APPS retention 评测在严格指标上 replay 与 LARGE 均为 0/48，属于零分地板，既不能证明遗忘也不能证明保持；详见 [APPS retention 报告](apps_direct_replay_retention.md)。
