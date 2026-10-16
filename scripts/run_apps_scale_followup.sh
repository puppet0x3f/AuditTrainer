#!/usr/bin/env bash
# CPU-only scoring of saved APPS programs on new independent inputs.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY=/home/lc/.conda/envs/audittrainer/bin/python
OUT=runs/refinement/apps_scale_followup

if [ -e "$OUT/result.json" ]; then
  echo '补充测试结果已存在；脚本不会覆盖。' >&2
  exit 1
fi
mkdir -p "$OUT"
"$PY" -u -m audittrainer.apps_scale_followup score 2>&1 | tee "$OUT/run.log"
