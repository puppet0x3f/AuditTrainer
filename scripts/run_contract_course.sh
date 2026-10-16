#!/usr/bin/env bash
# GPU 0/1: train + baseline concurrently, then two AFTER evaluation shards.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
conda run --no-capture-output -n audittrainer python -m audittrainer.refinement_contract_course check
train() {
  conda run --no-capture-output -n audittrainer python -u -m audittrainer.refinement_contract_course train \
    2>&1 | tee runs/refinement/contract_course/train.log
}
evaluate() {
  local method="$1" shard="$2"
  conda run --no-capture-output -n audittrainer python -u -m audittrainer.refinement_contract_course evaluate \
    --method "$method" --shard "$shard" \
    2>&1 | tee "runs/refinement/contract_course/${method}_${shard}.log"
}
baseline() {
  evaluate BEFORE 0 || return 1
  evaluate BEFORE 1 || return 1
}
if [ -e runs/refinement/contract_course/training ]; then
  echo 'training目录已存在；本脚本不覆盖已有训练，请使用单独评估命令。' >&2
  exit 1
fi
train &
train_pid=$!
baseline &
baseline_pid=$!
status=0
wait "$train_pid" || status=1
wait "$baseline_pid" || status=1
if ((status)); then echo '训练或基线评估失败，请检查日志；未启动AFTER。' >&2; exit 1; fi
evaluate AFTER 0 &
pid0=$!
evaluate AFTER 1 &
pid1=$!
status=0
wait "$pid0" || status=1
wait "$pid1" || status=1
if ((status)); then echo 'AFTER评估失败，请检查日志；未生成汇总。' >&2; exit 1; fi
conda run --no-capture-output -n audittrainer python -m audittrainer.refinement_contract_course report
