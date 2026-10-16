#!/usr/bin/env bash
# Two frozen A/S anchor states, one worker per GPU 0/1; no training.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
conda run --no-capture-output -n audittrainer \
  python -m audittrainer.refinement_anchor_value_refresh check
evaluate() {
  local index="$1"
  conda run --no-capture-output -n audittrainer \
    python -u -m audittrainer.refinement_anchor_value_refresh run \
    --index "$index" \
    2>&1 | tee "runs/refinement/anchor_value_refresh/run_${index}.log"
}
evaluate 0 &
pid0=$!
evaluate 1 &
pid1=$!
status=0
wait "$pid0" || status=1
wait "$pid1" || status=1
if ((status)); then
  echo '评估失败；请检查run_0.log和run_1.log，尚未汇总。' >&2
  exit 1
fi
conda run --no-capture-output -n audittrainer \
  python -m audittrainer.refinement_anchor_value_refresh report
