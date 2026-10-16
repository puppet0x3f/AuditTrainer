#!/usr/bin/env bash
# Re-score saved OLD96 and LARGE programs on public examples and three diagnostic roots.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY=/home/lc/.conda/envs/audittrainer/bin/python
OUT=runs/refinement/apps_fresh_audit

"$PY" -m audittrainer.apps_fresh_audit check
if [ -e "$OUT/result.json" ]; then
  echo 'APPS新题复评结果已存在；脚本不会覆盖。' >&2
  exit 1
fi
mkdir -p "$OUT/logs"
if [ ! -e "$OUT/signals.json" ]; then
  OMP_NUM_THREADS=1 "$PY" -u -m audittrainer.apps_fresh_audit score \
    2>&1 | tee "$OUT/logs/score.log"
fi
"$PY" -m audittrainer.apps_fresh_audit report 2>&1 | tee "$OUT/logs/report.log"
