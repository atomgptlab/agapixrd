#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DEFAULT_JARVIS_PYTHON="/home/ccamp104/scratchkchoudh2/ccamp104/miniconda3/envs/jarvis/bin/python"
PYTHON="${PYTHON:-}"

if [ -z "$PYTHON" ]; then
  if [ -x "$DEFAULT_JARVIS_PYTHON" ]; then
    PYTHON="$DEFAULT_JARVIS_PYTHON"
  else
    PYTHON="python"
  fi
fi

cd "$PROJECT_ROOT"

DIRS_LIST="${1:-scripts/benchmark/runs_to_compile.txt}"
if [ "$#" -gt 0 ]; then
  shift
fi

"$PYTHON" scripts/benchmark/compile_paper_results.py "$DIRS_LIST" \
  --runs-root runs \
  --output-dir paper_results \
  --prefix paper_results \
  "$@"
