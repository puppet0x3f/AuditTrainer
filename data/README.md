# 数据与模型下载

仓库只保留轻量的测试夹具和元数据；训练集、评测集、运行轨迹、模型权重和生成的训练文件不会提交到 GitHub。

需要复现实验时，请按固定版本下载：

- APPS：[`codeparrot/apps` revision `21e74ddf8de1a21436da12e3e653065c5213e9d1`](https://huggingface.co/datasets/codeparrot/apps/tree/21e74ddf8de1a21436da12e3e653065c5213e9d1)
- TACO：[`BAAI/TACO`](https://huggingface.co/datasets/BAAI/TACO)
- Qwen3-8B：从模型卡提供的官方地址下载，并放到仓库根目录的 `Qwen3-8B/`。

下载后的路径应与 `configs/` 和 `scripts/` 中的默认路径一致。相关目录已加入 `.gitignore`，不会被误提交。
