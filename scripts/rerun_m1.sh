#!/usr/bin/env bash
# Run from any directory; use the already activated server Python environment.
set -euo pipefail
SCRIPT_DIR="${BASH_SOURCE[0]%/*}"
if [[ "$SCRIPT_DIR" == "${BASH_SOURCE[0]}" ]]; then SCRIPT_DIR=.; fi
ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
exec "${PYTHON:-python}" "$ROOT/scripts/rerun_m1.py" "$@"
