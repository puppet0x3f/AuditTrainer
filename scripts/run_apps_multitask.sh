#!/usr/bin/env bash
# Executor APPS SFT on GPU 0 while GPU 1 runs the frozen BEFORE control;
# AFTER free-execution controls are split across GPU 0 and GPU 1.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY=/home/lc/.conda/envs/audittrainer/bin/python
"$PY" -m audittrainer.apps_multitask check
if [ -e runs/refinement/apps_multitask/training ]; then
  echo '已有 APPS 训练目录；请检查旧运行，脚本不会覆盖。' >&2
  exit 1
fi
mkdir -p runs/refinement/apps_multitask/logs
train() {
  CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 "$PY" -u -m audittrainer.apps_multitask train \
    2>&1 | tee runs/refinement/apps_multitask/logs/train.log
}
evaluate() {
  local condition="$1" shard="$2" shards="$3" gpu="$4"
  CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=4 "$PY" -u -m audittrainer.apps_multitask evaluate \
    --condition "$condition" --shard "$shard" --shards "$shards" \
    2>&1 | tee "runs/refinement/apps_multitask/logs/${condition}_${shard}.log"
}
train & train_pid=$!
evaluate BEFORE 0 1 1 & before_pid=$!
status=0
wait "$train_pid" || status=1
wait "$before_pid" || status=1
if (( status )); then
  echo '训练或 BEFORE 评估失败；AFTER 尚未启动。' >&2
  exit 1
fi
evaluate AFTER 0 2 0 & after0_pid=$!
evaluate AFTER 1 2 1 & after1_pid=$!
status=0
wait "$after0_pid" || status=1
wait "$after1_pid" || status=1
if (( status )); then
  echo 'AFTER 评估未完成；尚未汇总。' >&2
  exit 1
fi
"$PY" -m audittrainer.apps_multitask report --shards 2
