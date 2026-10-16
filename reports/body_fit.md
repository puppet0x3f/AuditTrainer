# 纯正文可学习性诊断

2026-09-17，COMPLETE。此实验是推进流程的工作包 1；使用工作包 0 修补后的评分确认目标仍然成立。

**结论：calendar 的明确好计划可以通过正文 SFT 学成稳定的条件自由生成。** 起点与最终模型各采样 64 条新种子的条件 SPLIT，完整目标责任结构由 27/64 提高到 64/64。达到预先记录的 52/64 诊断参考值，结束单例调参，进入多结构课程。

| 指标 | 原共同 SFT | PURE_BODY |
| --- | ---: | ---: |
| 完整目标结构 | 27/64 | 64/64 |
| 首次宿主注册 | 50/64 | 64/64 |
| 固定目标平均正文 CE | 0.197223 | 0.000062454 |
| P(SPLIT) | 22.270% | 20.181% |

起点中的其余 37 条既包含合法的其他结构，也包含 14 条注册拒绝，不能全部称为责任错误。拒绝为遗漏义务所有者 8、JSON 无效 3、未知所有者 2、数据流无效 1；最终模型没有目标结构字段不匹配。

从 `runs/refinement/PR1C_support/adaptation/final_adapter` 独立分支，只训练 174 token 目标的平均正文 CE；学习率 1e-5、80 次 AdamW 更新、梯度裁剪 1，无门控监督、KL、保持损失或 RL。规划 LoRA 发生更新，F01 指纹不变，保存重载前向差为 0。

本实验说明正文学习路径可行，不能把提升单独归因于移除 KL 或某个超参数。没有新增完整自然执行，也不据此声称自然收益或泛化改善。门控概率反而略降，说明正文拟合与能力调用仍需分别测量。PURE_BODY 仅作诊断，不作为课程训练起点。

后续已分别完成两条独立分支：[门控与正文课程及深度 2 诊断](/home/lc/AuditTrainer/reports/responsibility_course.md)，以及直接沿用本实验纯正文目标的[多任务推广测试](/home/lc/AuditTrainer/reports/body_transfer.md)。后者的训练状态完整目标为 7/96 → 82/96，未训练组合完整分配为 29/96 → 91/96，其中内部节点为 13/48 → 47/48。两条分支都从原共同 SFT 开始，各自回答不同问题；本报告的单例 27/64 → 64/64 结论保持不变。

证据：

- [配置](/home/lc/AuditTrainer/configs/refinement_body_fit.yaml)
- [完整结果](/home/lc/AuditTrainer/runs/refinement/body_fit/result.json)
- [训练记录](/home/lc/AuditTrainer/runs/refinement/body_fit/PURE_BODY/training_report.json)
- [B0 自由生成](/home/lc/AuditTrainer/runs/refinement/body_fit/evaluation/B0/result.json)
- [PURE_BODY 自由生成](/home/lc/AuditTrainer/runs/refinement/body_fit/evaluation/PURE_BODY/result.json)
