#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
PYTHON="/home/lc/.conda/envs/audittrainer/bin/python"
MODULE="audittrainer.apps_diverse_sft"

"$PYTHON" -m "$MODULE" check
CUDA_VISIBLE_DEVICES=0 "$PYTHON" -m "$MODULE" train
"$PYTHON" -m "$MODULE" check --trained

CUDA_VISIBLE_DEVICES=0 "$PYTHON" -m "$MODULE" evaluate --shard 0 &
pid0=$!
CUDA_VISIBLE_DEVICES=1 "$PYTHON" -m "$MODULE" evaluate --shard 1 &
pid1=$!
CUDA_VISIBLE_DEVICES=2 "$PYTHON" -m "$MODULE" evaluate --shard 2 &
pid2=$!

status=0
wait "$pid0" || status=1
wait "$pid1" || status=1
wait "$pid2" || status=1
if (( status != 0 )); then
    exit "$status"
fi
"$PYTHON" -m "$MODULE" report
