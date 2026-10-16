#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
PYTHON="/home/lc/.conda/envs/audittrainer/bin/python"

"$PYTHON" -m audittrainer.apps_direct_sft check
"$PYTHON" -m audittrainer.apps_direct_sft train
