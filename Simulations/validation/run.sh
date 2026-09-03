#!/usr/bin/env bash
# One command: draw the sequences, compute the TraMineR reference, run the check.
# Needs R with TraMineR + jsonlite, and a python with numpy and numba.
set -euo pipefail
cd "$(dirname "$0")"
PY="${PY:-../../.venv/bin/python}"
PYTHONPATH="$(cd .. && pwd)" "$PY" make_sequences.py "$@"
Rscript tramineR_reference.R
PYTHONPATH="$(cd .. && pwd)" "$PY" compare_om.py
