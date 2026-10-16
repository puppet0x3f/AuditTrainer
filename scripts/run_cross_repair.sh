#!/usr/bin/env bash
# CPU-only fixed-artifact experiment; shows a 12-condition progress bar.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
conda run --no-capture-output -n audittrainer python -m audittrainer.refinement_cross_repair check
conda run --no-capture-output -n audittrainer \
  python -u -m audittrainer.refinement_cross_repair run --workers 3 \
  2>&1 | tee runs/refinement/cross_repair/run.log
