#!/usr/bin/env bash
# Three CPU workers reproduce teacher selection and score frozen APPS probes.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY=/home/lc/.conda/envs/audittrainer/bin/python
OUT=runs/refinement/apps_teacher_audit

"$PY" -m audittrainer.apps_teacher_audit check
if [ -e "$OUT/result.json" ] || [ -e "$OUT/shard_0.json" ] || [ -e "$OUT/shard_1.json" ] || [ -e "$OUT/shard_2.json" ]; then
  echo '审计结果或分片已存在；脚本不会覆盖。' >&2
  exit 1
fi
mkdir -p "$OUT"

run_shard() {
  local shard="$1"
  OMP_NUM_THREADS=1 "$PY" -u -m audittrainer.apps_teacher_audit score-shard --shard "$shard" \
    2>&1 | tee "$OUT/shard_${shard}.log"
}
run_shard 0 & p0=$!
run_shard 1 & p1=$!
run_shard 2 & p2=$!
status=0
wait "$p0" || status=1
wait "$p1" || status=1
wait "$p2" || status=1
if (( status )); then
  echo '参考解审计有分片未完成；检查分片日志。' >&2
  exit 1
fi
"$PY" -m audittrainer.apps_teacher_audit report 2>&1 | tee "$OUT/report.log"
