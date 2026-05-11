#!/usr/bin/env bash
# Print a summary of how far each pipeline has progressed.
# Run from any directory; resolves paths relative to this script's repo root.

REPO="$(cd "$(dirname "$0")/.." && pwd)"
RUNS_DIR="$REPO/runs"

# Raw average iteration time in seconds (float) from the last 20 log entries.
# Reads the "[Xs]" timing on each PM/DG/Best result line. Prints nothing if unavailable.
_avg_iter_seconds() {
    local log_file="$1"
    [[ -z "$log_file" ]] && return
    grep -oP 'Best=.*\[\K[0-9.]+(?=s\])' "$log_file" | tail -20 | \
        awk 'BEGIN{sum=0;n=0} {sum+=$1;n++} END{if(n>0) printf "%.2f", sum/n}'
}

# Human-readable duration from seconds: Xs / X.Xm / X.Xh / X.Xd
format_duration() {
    awk "BEGIN {
        s = $1
        if      (s < 60)    printf \"%.0fs\",   s
        else if (s < 3600)  printf \"%.1fm\",   s/60
        else if (s < 86400) printf \"%.1fh\",   s/3600
        else                printf \"%.1fd\",   s/86400
    }"
}

print_section() {
    local label="$1" total="$2" cache_subdir="$3" log_prefix="$4"
    shift 4
    local pipelines=("$@")

    printf "\n=== %s (target: %d structures) ===\n" "$label" "$total"
    printf "%-42s  %8s  %8s  %9s  %6s  %9s  %8s  %s\n" \
        "Pipeline" "Cache" "CSV rows" "Remaining" "Done%" "Avg/iter" "ETA" "Log file"
    printf "%-42s  %8s  %8s  %9s  %6s  %9s  %8s  %s\n" \
        "------------------------------------------" "--------" "--------" "---------" "------" "---------" "--------" "--------"

    for NAME in "${pipelines[@]}"; do
        DIR="$RUNS_DIR/$NAME"
        CACHE=$(find "$DIR/$cache_subdir" -name '*.json' 2>/dev/null | wc -l)
        CSV_FILE="$DIR/${NAME}_results.csv"
        if [[ -f "$CSV_FILE" ]]; then
            CSV_ROWS=$(( $(wc -l < "$CSV_FILE") - 1 ))
        else
            CSV_ROWS=0
        fi
        REMAINING=$(( total - CACHE ))
        PCT=$(awk "BEGIN { printf \"%.1f\", $CACHE / $total * 100 }")

        JOB_NAME="${log_prefix}${NAME#*_sequential_}"
        LATEST=$(ls -t "$REPO/logs/${JOB_NAME}_"*.out 2>/dev/null | head -1)
        LOG_BASE="(no log found)"
        [[ -n "$LATEST" ]] && LOG_BASE="$(basename "$LATEST")"

        RAW=$(_avg_iter_seconds "$LATEST")
        if [[ -n "$RAW" ]]; then
            AVG="$(awk "BEGIN { printf \"%.1fs\", $RAW }")"
            ETA="$(format_duration "$(awk "BEGIN { printf \"%.0f\", $RAW * $REMAINING }")")"
        else
            AVG="N/A"
            ETA="N/A"
        fi

        printf "%-42s  %8d  %8d  %9d  %5s%%  %9s  %8s  %s\n" \
            "$NAME" "$CACHE" "$CSV_ROWS" "$REMAINING" "$PCT" "$AVG" "$ETA" "$LOG_BASE"
    done
    echo ""
}

# ── Alexandria PBE-hull (current full dataset) ────────────────────────────────
ALEX_TOTAL=87631  # 116712 total, 87631 pass abc<=10 Å
ALEX_PIPELINES=(
    alex_sequential_no_refinement
    alex_sequential_no_refinement_alignnff
    alex_sequential_bmgn
    alex_sequential_bmgn_alignnff
    alex_sequential_gsas2
    alex_sequential_gsas2_alignnff
)
print_section "Alexandria PBE-hull" "$ALEX_TOTAL" ".alex_cache" "alex_seq_" "${ALEX_PIPELINES[@]}"

# ── RRUFF powder XRD ──────────────────────────────────────────────────────────
RRUFF_TOTAL=276  # 803 total, 276 pass abc<=10 Å
RRUFF_PIPELINES=(
    rruff_no_refinement
    rruff_no_refinement_alignnff
    rruff_bmgn
    rruff_bmgn_alignnff
    rruff_gsas2
    rruff_gsas2_alignnff
)

printf "\n=== RRUFF powder XRD (target: %d structures) ===\n" "$RRUFF_TOTAL"
printf "%-42s  %8s  %8s  %9s  %6s  %9s  %8s  %s\n" \
    "Pipeline" "Cache" "CSV rows" "Remaining" "Done%" "Avg/iter" "ETA" "Log file"
printf "%-42s  %8s  %8s  %9s  %6s  %9s  %8s  %s\n" \
    "------------------------------------------" "--------" "--------" "---------" "------" "---------" "--------" "--------"

for NAME in "${RRUFF_PIPELINES[@]}"; do
    DIR="$RUNS_DIR/$NAME"
    CACHE=$(find "$DIR/.rruff_cache" -name '*.json' 2>/dev/null | wc -l)
    CSV_FILE="$DIR/${NAME}_results.csv"
    if [[ -f "$CSV_FILE" ]]; then
        CSV_ROWS=$(( $(wc -l < "$CSV_FILE") - 1 ))
    else
        CSV_ROWS=0
    fi
    REMAINING=$(( RRUFF_TOTAL - CACHE ))
    PCT=$(awk "BEGIN { printf \"%.1f\", $CACHE / $RRUFF_TOTAL * 100 }")

    JOB_NAME="${NAME/rruff_no_refinement/rruff_none}"
    LATEST=$(ls -t "$REPO/logs/${JOB_NAME}_"*.out 2>/dev/null | head -1)
    LOG_BASE="(no log found)"
    [[ -n "$LATEST" ]] && LOG_BASE="$(basename "$LATEST")"

    RAW=$(_avg_iter_seconds "$LATEST")
    if [[ -n "$RAW" ]]; then
        AVG="$(awk "BEGIN { printf \"%.1fs\", $RAW }")"
        ETA="$(format_duration "$(awk "BEGIN { printf \"%.0f\", $RAW * $REMAINING }")")"
    else
        AVG="N/A"
        ETA="N/A"
    fi

    printf "%-42s  %8d  %8d  %9d  %5s%%  %9s  %8s  %s\n" \
        "$NAME" "$CACHE" "$CSV_ROWS" "$REMAINING" "$PCT" "$AVG" "$ETA" "$LOG_BASE"
done
echo ""
