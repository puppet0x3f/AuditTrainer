#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
PYTHON="/home/lc/.conda/envs/audittrainer/bin/python"

command="${1:-train}"
case "$command" in
  check)
    "$PYTHON" -m audittrainer.apps_direct_retention_sft check
    ;;
  train)
    "$PYTHON" -m audittrainer.apps_direct_retention_sft check
    "$PYTHON" -u -m audittrainer.apps_direct_retention_sft train
    ;;
  *)
    echo "usage: $0 [check|train]" >&2
    exit 2
    ;;
esac
