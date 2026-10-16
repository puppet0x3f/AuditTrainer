#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
PYTHON="/home/lc/.conda/envs/audittrainer/bin/python"
MODULE="audittrainer.apps_taco_direct_code_k3"

"$PYTHON" -m "$MODULE" check

pids=()
for shard in 0 1; do
  "$PYTHON" -m "$MODULE" evaluate --shard "$shard" &
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

pids=()
for shard in 0 1; do
  "$PYTHON" -m "$MODULE" bridge --shard "$shard" &
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

"$PYTHON" -m "$MODULE" report
