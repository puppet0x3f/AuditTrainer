#!/usr/bin/env bash
# Equal-update APPS data-scale comparison: two training GPUs, then three eval GPUs.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY=/home/lc/.conda/envs/audittrainer/bin/python
OUT=runs/refinement/apps_scale

"$PY" -m audittrainer.apps_scale check
if [ -e "$OUT/SMALL/training" ] || [ -e "$OUT/LARGE/training" ]; then
  echo '训练目录已存在；请先检查已有运行，脚本不会覆盖。' >&2
  exit 1
fi
mkdir -p "$OUT/logs"

train() {
  local arm="$1" gpu="$2"
  CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=4 "$PY" -u -m audittrainer.apps_scale train \
    --arm "$arm" 2>&1 | tee "$OUT/logs/train_${arm}.log"
}
train SMALL 0 & small_pid=$!
train LARGE 1 & large_pid=$!
status=0
wait "$small_pid" || status=1
wait "$large_pid" || status=1
if (( status )); then
  echo '训练未全部完成；评测尚未启动。' >&2
  exit 1
fi

evaluate_shard() {
  local arm="$1" shard="$2" shards="$3" gpu="$4"
  CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=4 "$PY" -u -m audittrainer.apps_scale evaluate \
    --arm "$arm" --shard "$shard" --shards "$shards" \
    2>&1 | tee "$OUT/logs/eval_${arm}_${shard}.log"
}
for arm in BASE OLD96 SMALL LARGE; do
  evaluate_shard "$arm" 0 3 0 & p0=$!
  evaluate_shard "$arm" 1 3 1 & p1=$!
  evaluate_shard "$arm" 2 3 2 & p2=$!
  status=0
  wait "$p0" || status=1
  wait "$p1" || status=1
  wait "$p2" || status=1
  if (( status )); then
    echo "$arm 评测未全部完成；检查对应分片日志。" >&2
    exit 1
  fi
done

"$PY" -u -m audittrainer.apps_scale score-probes \
  2>&1 | tee "$OUT/logs/probes.log"
"$PY" -m audittrainer.apps_scale report | tee "$OUT/logs/report.log"
