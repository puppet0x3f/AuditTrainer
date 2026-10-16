#!/usr/bin/env bash
# GPU 0/1/2: gate training plus queued BEFORE shards, then AFTER shards.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
conda run --no-capture-output -n audittrainer \
  python -m audittrainer.refinement_gate_contrast check
if [ -e runs/refinement/gate_contrast/training ]; then
  echo 'training目录已存在；本脚本不覆盖已有训练。' >&2
  exit 1
fi
train() {
  conda run --no-capture-output -n audittrainer \
    python -u -m audittrainer.refinement_gate_contrast train \
    2>&1 | tee runs/refinement/gate_contrast/train.log
}
natural() {
  local method="$1" shard="$2"
  conda run --no-capture-output -n audittrainer \
    python -u -m audittrainer.refinement_gate_contrast natural \
    --method "$method" --shard "$shard" \
    2>&1 | tee "runs/refinement/gate_contrast/${method}_${shard}.log"
}
train &
train_pid=$!
before_pids=()
for shard in 0 1 2; do
  natural BEFORE "$shard" &
  before_pids+=("$!")
done
status=0
wait "$train_pid" || status=1
for pid in "${before_pids[@]}"; do wait "$pid" || status=1; done
if ((status)); then
  echo '训练或BEFORE评估失败；未启动AFTER。' >&2
  exit 1
fi
after_pids=()
for shard in 0 1 2; do
  natural AFTER "$shard" &
  after_pids+=("$!")
done
status=0
for pid in "${after_pids[@]}"; do wait "$pid" || status=1; done
if ((status)); then
  echo 'AFTER评估失败；未汇总结果。' >&2
  exit 1
fi
conda run --no-capture-output -n audittrainer \
  python -m audittrainer.refinement_gate_contrast report
