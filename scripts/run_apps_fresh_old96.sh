#!/usr/bin/env bash
# OLD96 single-sample control on the frozen 48 fresh APPS competition roots.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY=/home/lc/.conda/envs/audittrainer/bin/python
OUT=runs/refinement/apps_fresh_old96

"$PY" -m audittrainer.apps_fresh_old96 check
if [ -e "$OUT/result.json" ]; then
  echo 'APPS旧执行器对照结果已存在；脚本不会覆盖。' >&2
  exit 1
fi
mkdir -p "$OUT/logs"

run_shard() {
  local shard="$1"
  CUDA_VISIBLE_DEVICES="$shard" OMP_NUM_THREADS=4 "$PY" -u \
    -m audittrainer.apps_fresh_old96 evaluate --shard "$shard" \
    2>&1 | tee "$OUT/logs/shard_${shard}.log"
}
pids=()
for shard in 0 1 2; do
  if [ ! -e "$OUT/shard_${shard}.json" ]; then
    run_shard "$shard" & pids+=($!)
  fi
done
status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
if (( status )); then
  echo 'APPS旧执行器对照分片未全部完成；检查对应日志。' >&2
  exit 1
fi
"$PY" -m audittrainer.apps_fresh_old96 report 2>&1 | tee "$OUT/logs/report.log"
