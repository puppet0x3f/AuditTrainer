#!/usr/bin/env bash
# Repair the completed executor pilot from its AFTER adapter; evaluate three shards in parallel.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

conda run --no-capture-output -n audittrainer \
  python -m audittrainer.refinement_executor_retention_repair check
if [ -e runs/refinement/executor_retention_repair/training ]; then
  echo 'training目录已存在；本脚本不覆盖已有训练。' >&2
  exit 1
fi

conda run --no-capture-output -n audittrainer \
  python -u -m audittrainer.refinement_executor_retention_repair train \
  2>&1 | tee runs/refinement/executor_retention_repair/train.log

evaluate() {
  local shard="$1"
  conda run --no-capture-output -n audittrainer \
    python -u -m audittrainer.refinement_executor_retention_repair evaluate \
    --shard "$shard" \
    2>&1 | tee "runs/refinement/executor_retention_repair/REPAIR_${shard}.log"
}

pids=()
for shard in 0 1 2; do
  evaluate "$shard" &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
if ((status)); then
  echo 'REPAIR分片失败；未执行汇总。' >&2
  exit 1
fi

conda run --no-capture-output -n audittrainer \
  python -m audittrainer.refinement_executor_retention_repair report
