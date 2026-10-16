#!/usr/bin/env bash
# GPU 0/1/2: train plus queued baseline shards; then three AFTER shards.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
conda run --no-capture-output -n audittrainer \
  python -m audittrainer.refinement_contract_correction check
if [ -e runs/refinement/contract_correction/training ]; then
  echo 'training目录已存在；本脚本不覆盖已有训练。' >&2
  exit 1
fi
train() {
  conda run --no-capture-output -n audittrainer \
    python -u -m audittrainer.refinement_contract_correction train \
    2>&1 | tee runs/refinement/contract_correction/train.log
}
evaluate() {
  local method="$1" shard="$2"
  conda run --no-capture-output -n audittrainer \
    python -u -m audittrainer.refinement_contract_correction evaluate \
    --method "$method" --shard "$shard" \
    2>&1 | tee "runs/refinement/contract_correction/${method}_${shard}.log"
}
train &
train_pid=$!
before_pids=()
for shard in 0 1 2; do
  evaluate BEFORE "$shard" &
  before_pids+=("$!")
done
status=0
wait "$train_pid" || status=1
for pid in "${before_pids[@]}"; do wait "$pid" || status=1; done
if ((status)); then
  echo '训练或基线分片失败；未启动AFTER。' >&2
  exit 1
fi
after_pids=()
for shard in 0 1 2; do
  evaluate AFTER "$shard" &
  after_pids+=("$!")
done
status=0
for pid in "${after_pids[@]}"; do wait "$pid" || status=1; done
if ((status)); then
  echo 'AFTER分片失败；未汇总结果。' >&2
  exit 1
fi
conda run --no-capture-output -n audittrainer \
  python -m audittrainer.refinement_contract_correction report
