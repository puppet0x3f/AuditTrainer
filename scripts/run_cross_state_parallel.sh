#!/usr/bin/env bash
# Two independent model-evaluation workers; no training.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
conda run --no-capture-output -n audittrainer python -m audittrainer.refinement_cross_state check
worker() {
  local worker_index="$1" task_index
  for ((task_index=worker_index; task_index<8; task_index+=2)); do
    conda run --no-capture-output -n audittrainer \
      python -u -m audittrainer.refinement_cross_state run --index "$task_index" \
      > "runs/refinement/cross_state/run_${task_index}.log" 2>&1 || return 1
  done
}
pids=()
for worker_index in 0 1; do
  worker "$worker_index" &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
if ((status)); then
  echo '评估进程失败，请检查 runs/refinement/cross_state/run_*.log；未执行汇总。' >&2
  exit 1
fi
conda run --no-capture-output -n audittrainer python -m audittrainer.refinement_cross_state report
