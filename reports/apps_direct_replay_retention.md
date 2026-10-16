# APPS replay retention 评测

TACO 未见任务评测中，固定预算 replay adapter 的严格通过率没有超过 DIVERSE adapter。下一步只验证它是否保持原有 APPS 能力，不启动训练。

评测固定的 48 个 APPS validation roots，每题 3 个候选，使用与冻结 LARGE direct arm 相同的 prompt、seed、解码、公开样例选择器和私有评分器。replay adapter 是 `runs/refinement/apps_direct_retention_sft/training/final_adapter`。GPU 0、1、2 各处理 16 个 roots；模块内提供 tqdm，日志写入 `runs/refinement/apps_direct_replay_retention/logs/`。

主要指标是 selected k1/k3 的 strict private pass、private pass、candidate-level oracle pass，以及 replay 相对冻结 LARGE 的逐题 win/loss/tie。私有结果只在汇总阶段读取，不参与候选选择。

```bash
bash scripts/run_apps_direct_replay_retention.sh
```

如果中断，可以重复运行；已完成分片会跳过。只有在 replay 不降低 APPS retention，且确实改善或至少保持 TACO/其他新数据集表现时，才进入责任约束执行器 pilot；否则停止继续增加 replay 样本，优先修正数据和接口分布。

## 结果

评测已完成。selected strict k1/k3 均为 **0/48**，与冻结 LARGE direct 完全相同，因此 paired tie 为 **48/48**。这个 tie 不能解释为“保持能力已被证明”：冻结 LARGE 自身在同一严格指标上也是 0/48，评测落在零分地板。

逐候选复核提供的信号是：replay 在 144 个候选中生成 135 个程序，公开样例通过 9 个；冻结 LARGE 分别为 131、7。replay 有 1 个 root 的候选通过严格校验私有测试（4 个候选日志标为 PASS，其中 3 个的实际输出经 `apps_scale._valid_private` 复核为错误），LARGE 也有 1 个 root；两者都没有候选同时通过私有测试和公开样例。replay 的 k3 输出 token 为 56,292，LARGE 为 67,983，不能把较短输出直接当作正确性提升。

因此本轮没有证明 replay 造成 APPS 遗忘，也没有证明它保持了 APPS 泛化。当前 APPS direct retention 集合的严格指标不适合作为下一轮检查点选择依据；下一次应使用预先固定、带独立输入探针且基线有非零通过的集合，并单独报告 selected private、candidate-oracle private、public、strict 四项。

结合已完成的 TACO replay：replay bridge selected strict k1/k3 为 **6/100、12/100**，低于 DIVERSE bridge 的 **11/100、13/100**，candidate-oracle k3 为 **14/100**，没有新根成功。结论是固定预算 replay 尚未带来跨数据集收益；不继续扩大同类 replay 或直接训练规划器。
