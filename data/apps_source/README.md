# F9 APPS 来源

数据集：`codeparrot/apps`，revision `21e74ddf8de1a21436da12e3e653065c5213e9d1`。
经 hf-mirror.com 读取同 revision 的 `train.jsonl` 前 100 条，原始顺序保存在 `train_first100.jsonl`。
`source_first100.json` 记录下载时的初始选择计划；检查后发现这些记录均只有一个 stdin 用例，无法采用
“一例可见、其余私有”的划分。因此实际导入使用首三个 stdin 记录，visible_cases=0，原始测试只用于离线评分。
最终实际选择与拒收原因以 `../apps_f9/import_report.json` 和各 root/test manifest 为准。

读取未执行其中的参考 solutions；公开任务文件不包含 solutions。最初的 12 条读取及其来源记录也保留。
只读取 train 范围，未读取官方 test 范围；本次所有检查过的任务用于开发，不能视为新的 SEALED 数据。

来源：[Hugging Face 固定 revision](https://huggingface.co/datasets/codeparrot/apps/tree/21e74ddf8de1a21436da12e3e653065c5213e9d1)、
[APPS 官方项目](https://github.com/hendrycks/apps)。

## 多任务训练抽取（2026-09-23）

另外取得同 revision 的完整 `train.jsonl`（5,000 条，SHA-256
`45e82ef22ed8e7c0c04d881a21b923e9dd233157896b0b8d5b3493e887499cae`）。
本地文件 `train_full.jsonl` 已被 `.gitignore` 排除；可从上面的固定 revision 下载。
`audittrainer.apps_multitask prepare` 在原始 train 中按固定哈希顺序，排除 F9 已使用题目，
仅抽取受支持的 stdin 题，并在沙箱中验证参考解及其 `solve(text)` 适配。
结果位于 `data/refinement/apps_multitask/`：192 道 TRAIN、48 道独立留出 VALIDATION；
单例题没有在线可见测试，多例题只公开第一例，其余仅用于离线评分。
VALIDATION 仍来自 APPS **train**，不是官方 test 或新的密封基准。
