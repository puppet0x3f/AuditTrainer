#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
PYTHON="/home/lc/.conda/envs/audittrainer/bin/python"

"$PYTHON" -m audittrainer.apps_taco_transfer check
"$PYTHON" -m audittrainer.apps_taco_interface replay

CUDA_VISIBLE_DEVICES=0 "$PYTHON" -m audittrainer.apps_taco_transfer evaluate --shard 0 &
pid0=$!
CUDA_VISIBLE_DEVICES=1 "$PYTHON" -m audittrainer.apps_taco_transfer evaluate --shard 1 &
pid1=$!
CUDA_VISIBLE_DEVICES=2 "$PYTHON" -m audittrainer.apps_taco_transfer evaluate --shard 2 &
pid2=$!

status=0
wait "$pid0" || status=1
wait "$pid1" || status=1
wait "$pid2" || status=1
if (( status != 0 )); then
    exit "$status"
fi
"$PYTHON" -m audittrainer.apps_taco_transfer report
