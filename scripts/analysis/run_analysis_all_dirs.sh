#!/bin/bash
# Run analyse_filtered_rruff.py for every run directory that contains a results CSV.

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SCRIPT="$PROJECT_ROOT/scripts/analysis/analyse_filtered_rruff.py"
PYTHON=/home/ccamp104/scratchkchoudh2/ccamp104/miniconda3/envs/jarvis/bin/python
RUNS_DIR="$PROJECT_ROOT/runs"

for run_dir in "$RUNS_DIR"/*/; do
    # Skip non-directories and the figures summary folder
    [ -d "$run_dir" ] || continue
    dirname="$(basename "$run_dir")"
    [ "$dirname" = "refinement_summary_figures" ] && continue

    # Find the first CSV in this directory (not in subdirs)
    csv="$(find "$run_dir" -maxdepth 1 -name "*.csv" | head -1)"
    if [ -z "$csv" ]; then
        echo "SKIP $dirname (no CSV found)"
        continue
    fi

    outdir="$run_dir/analysis"
    echo ">>> $dirname"
    "$PYTHON" "$SCRIPT" --input "$csv" --output-dir "$outdir"
    echo ""
done
