#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
PYTHON="/home/lc/.conda/envs/audittrainer/bin/python"
MODULE="audittrainer.apps_direct_replay_retention"
OUT="runs/refinement/apps_direct_replay_retention"
mkdir -p "$OUT/logs"

if [[ -e "$OUT/request.json" ]]; then
  "$PYTHON" -m "$MODULE" check
else
  "$PYTHON" -m "$MODULE" prepare
fi

pids=()
for shard in 0 1 2; do
  CUDA_VISIBLE_DEVICES="$shard" OMP_NUM_THREADS=4 "$PYTHON" -u -m "$MODULE" \
    evaluate --shard "$shard" 2>&1 | tee "$OUT/logs/evaluate_${shard}.log" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
if [[ "$status" -ne 0 ]]; then
  exit "$status"
fi

"$PYTHON" -u -m "$MODULE" report 2>&1 | tee "$OUT/logs/report.log"
