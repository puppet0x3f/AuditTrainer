#!/usr/bin/env bash
# Two frozen-model evaluation workers, using GPU 0 and GPU 1 only.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
conda run --no-capture-output -n audittrainer \
  python -m audittrainer.refinement_contract_value_refresh check
worker() {
  local worker_index="$1" task_index
  for ((task_index=worker_index; task_index<3; task_index+=2)); do
    conda run --no-capture-output -n audittrainer \
      python -u -m audittrainer.refinement_contract_value_refresh run \
      --index "$task_index" \
      2>&1 | tee "runs/refinement/contract_value_refresh/run_${task_index}.log"
  done
}
worker 0 &
pid0=$!
worker 1 &
pid1=$!
status=0
wait "$pid0" || status=1
wait "$pid1" || status=1
if ((status)); then
  echo '模型评估失败；请检查run_*.log，尚未汇总。' >&2
  exit 1
fi
conda run --no-capture-output -n audittrainer \
  python -m audittrainer.refinement_contract_value_refresh report
