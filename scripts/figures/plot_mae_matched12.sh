#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

python "$PROJECT_ROOT/scripts/figures/plot_matched_vanilla_refinement_mae.py" \
    "$PROJECT_ROOT/dir_lists/all12_dirs.txt"
