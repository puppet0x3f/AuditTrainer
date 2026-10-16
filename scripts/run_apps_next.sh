#!/usr/bin/env bash
# APPS diagnostic controls: saved-code probes, then a frozen-model prompt control.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY=/home/lc/.conda/envs/audittrainer/bin/python

"$PY" -m audittrainer.apps_next check
mkdir -p runs/refinement/apps_next/logs
"$PY" -u -m audittrainer.apps_next score-probes \
  2>&1 | tee runs/refinement/apps_next/logs/probes.log

evaluate() {
  local shard="$1" gpu="$2"
  CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=4 "$PY" -u -m audittrainer.apps_next evaluate \
    --shard "$shard" --shards 3 \
    2>&1 | tee "runs/refinement/apps_next/logs/interface_${shard}.log"
}
evaluate 0 0 & p0=$!
evaluate 1 1 & p1=$!
evaluate 2 2 & p2=$!
status=0
wait "$p0" || status=1
wait "$p1" || status=1
wait "$p2" || status=1
if (( status )); then
  echo '接口提示对照未全部完成；检查三个 shard 日志后重跑对应 evaluate 命令。' >&2
  exit 1
fi
"$PY" -m audittrainer.apps_next report | tee runs/refinement/apps_next/logs/report.log
