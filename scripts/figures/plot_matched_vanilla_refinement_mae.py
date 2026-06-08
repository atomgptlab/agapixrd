#!/usr/bin/env python3

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib as mpl
mpl.use("Agg")
mpl.rcParams["font.family"] = "serif"

import matplotlib.pyplot as plt


# ───────────────────────── config / labels ─────────────────────────
PARAMS = ["a", "b", "c", "alpha", "beta", "gamma"]

length_params = ["a", "b", "c"]
angle_params = ["alpha", "beta", "gamma"]

ax_label_map = {
    "a": r"$a$",
    "b": r"$b$",
    "c": r"$c$",
    "alpha": r"$\alpha$",
    "beta": r"$\beta$",
    "gamma": r"$\gamma$",
}

param_colors = {
    "a": "#E76F51",
    "b": "#F4A261",
    "c": "#E9C46A",
    "alpha": "#457B9D",
    "beta": "#2A9D8F",
    "gamma": "#577590",
}

DATASET_ORDER = ["rruff", "alex"]

METHOD_ORDER = {
    "no_refinement": "No refinement",
    "bmgn": "BMGN",
    "gsas2": "GSAS-II",
}

EXPECTED_RUNS = [
    "rruff_no_refinement",
    "rruff_bmgn",
    "rruff_gsas2",
    "alex_no_refinement",
    "alex_bmgn",
    "alex_gsas2",
]


# ───────────────────────── debug helpers ─────────────────────────
def debug_print_schema(df: pd.DataFrame, label: str, max_rows: int = 5):
    """
    Print schema information without changing program behavior.

    This is intentionally verbose because the point is to inspect the exact
    CSV structure before changing the matching logic.
    """
    print("\n" + "=" * 90, file=sys.stderr)
    print(f"SCHEMA DEBUG: {label}", file=sys.stderr)
    print("=" * 90, file=sys.stderr)

    print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns", file=sys.stderr)

    print("\nColumns:", file=sys.stderr)
    for i, col in enumerate(df.columns):
        print(f"  [{i:02d}] {col}    dtype={df[col].dtype}", file=sys.stderr)

    id_like = [
        c for c in df.columns
        if any(tok in c.lower() for tok in ["id", "jid", "name", "formula"])
    ]

    match_like = [
        c for c in df.columns
        if any(
            tok in c.lower()
            for tok in [
                "match",
                "success",
                "similarity",
                "method",
                "rmsd",
                "score",
                "valid",
                "best",
            ]
        )
    ]

    lattice_like = [
        c for c in df.columns
        if any(
            tok in c.lower()
            for tok in [
                "pred_a",
                "pred_b",
                "pred_c",
                "pred_alpha",
                "pred_beta",
                "pred_gamma",
                "rruff_a",
                "rruff_b",
                "rruff_c",
                "rruff_alpha",
                "rruff_beta",
                "rruff_gamma",
                "alex_a",
                "alex_b",
                "alex_c",
                "alex_alpha",
                "alex_beta",
                "alex_gamma",
            ]
        )
    ]

    interesting = []
    for group in (id_like, match_like, lattice_like):
        for c in group:
            if c not in interesting:
                interesting.append(c)

    if interesting:
        print("\nInteresting columns preview:", file=sys.stderr)
        preview_cols = interesting[:40]
        print(df[preview_cols].head(max_rows).to_string(index=False), file=sys.stderr)

    if match_like:
        print("\nMatch/success/similarity-like column summaries:", file=sys.stderr)
        for c in match_like:
            print("-" * 60, file=sys.stderr)
            print(f"{c}    dtype={df[c].dtype}", file=sys.stderr)

            try:
                numeric = pd.to_numeric(df[c], errors="coerce")
                if numeric.notna().sum() > 0:
                    print(numeric.describe().to_string(), file=sys.stderr)
                else:
                    nunique = df[c].nunique(dropna=False)
                    print(f"nunique including NaN: {nunique}", file=sys.stderr)
                    print(df[c].value_counts(dropna=False).head(20).to_string(), file=sys.stderr)
            except Exception as e:
                print(f"Could not summarize {c}: {e}", file=sys.stderr)

    print("\nFirst rows, all columns:", file=sys.stderr)
    with pd.option_context("display.max_columns", None, "display.width", 240):
        print(df.head(max_rows).to_string(index=False), file=sys.stderr)

    print("=" * 90 + "\n", file=sys.stderr)


def debug_print_autodetect_attempt(df: pd.DataFrame, label: str):
    """
    Print what the current autodetection logic is about to look for.
    This does not change logic; it only makes failure more interpretable.
    """
    print("\n" + "-" * 90, file=sys.stderr)
    print(f"AUTODETECT DEBUG: {label}", file=sys.stderr)
    print("-" * 90, file=sys.stderr)

    id_candidates = [
        "id",
        "material_id",
        "structure_id",
        "sample_id",
        "entry_id",
        "jid",
        "rruff_id",
        "alex_id",
        "mp_id",
        "database_id",
        "source_id",
    ]

    match_candidates = [
        "matched",
        "match",
        "is_match",
        "structure_match",
        "structure_matched",
        "pymatgen_match",
        "pmg_match",
        "structurematcher_match",
        "structure_matcher_match",
        "sm_match",
        "valid_match",
    ]

    print("\nID candidates currently checked, in order:", file=sys.stderr)
    for c in id_candidates:
        status = "FOUND" if c in df.columns else "missing"
        print(f"  {c}: {status}", file=sys.stderr)

    print("\nMatch candidates currently checked, in order:", file=sys.stderr)
    for c in match_candidates:
        status = "FOUND" if c in df.columns else "missing"
        print(f"  {c}: {status}", file=sys.stderr)

    fuzzy_match_cols = [
        c for c in df.columns
        if "match" in c.lower() and not c.lower().endswith("rate")
    ]

    print("\nFuzzy match-like columns containing 'match':", file=sys.stderr)
    if fuzzy_match_cols:
        for c in fuzzy_match_cols:
            print(f"  {c}", file=sys.stderr)
    else:
        print("  none", file=sys.stderr)

    success_similarity_cols = [
        c for c in df.columns
        if any(tok in c.lower() for tok in ["success", "similarity", "best", "method"])
    ]

    print("\nSuccess/similarity/best/method columns present:", file=sys.stderr)
    if success_similarity_cols:
        for c in success_similarity_cols:
            print(f"  {c}", file=sys.stderr)
    else:
        print("  none", file=sys.stderr)

    print("-" * 90 + "\n", file=sys.stderr)


# ───────────────────────── path / CSV helpers ─────────────────────────
def find_runs_root(start: Path) -> Path:
    """Allow running either from project root or from inside runs/."""
    if (start / "runs").is_dir():
        return start / "runs"
    if start.name == "runs" and start.is_dir():
        return start
    raise FileNotFoundError(
        "Could not find a runs/ directory. Run this from the project root "
        "or from inside runs/."
    )


def load_dir_list(path: Path) -> list[str]:
    """Parse a text file of refinement directory names, one per line."""
    dirs = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        dirs.append(line)

    if not dirs:
        print(f"ERROR: No entries found in {path}", file=sys.stderr)
        sys.exit(1)

    return dirs


def newest_results_csv(run_dir: Path) -> Path | None:
    """
    Prefer the newest timestamped rruff_results_*/results.csv.
    Fall back to *_results.csv if needed.
    Return None if nothing is found.
    """
    timestamped = sorted(
        run_dir.glob("rruff_results_*/results.csv"),
        key=lambda p: p.stat().st_mtime,
    )
    if timestamped:
        return timestamped[-1]

    fallback = sorted(
        run_dir.glob("*_results.csv"),
        key=lambda p: p.stat().st_mtime,
    )
    if fallback:
        return fallback[-1]

    return None


def parse_run_key(run_key: str) -> tuple[str, str] | None:
    """
    Return (dataset, method) for keys like:
        rruff_no_refinement
        rruff_bmgn
        rruff_gsas2
        alex_no_refinement
        alex_bmgn
        alex_gsas2

    ALIGNN-FF directories are intentionally excluded.
    """
    if run_key.endswith("_alignnff"):
        return None

    for dataset in DATASET_ORDER:
        prefix = f"{dataset}_"
        if run_key.startswith(prefix):
            method = run_key[len(prefix):]
            if method in METHOD_ORDER:
                return dataset, method

    return None


# ───────────────────────── schema detection ─────────────────────────
def detect_id_col(df: pd.DataFrame, user_col: str | None = None) -> str:
    if user_col is not None:
        if user_col not in df.columns:
            debug_print_autodetect_attempt(df, "detect_id_col failed with explicit user_col")
            raise ValueError(f"Requested id column '{user_col}' not found.")
        return user_col

    candidates = [
        "id",
        "material_id",
        "structure_id",
        "sample_id",
        "entry_id",
        "jid",
        "rruff_id",
        "alex_id",
        "mp_id",
        "database_id",
        "source_id",
    ]

    print("\nDEBUG: Trying to detect ID column.", file=sys.stderr)
    print(f"DEBUG: Available columns: {list(df.columns)}", file=sys.stderr)
    print(f"DEBUG: Candidate ID columns: {candidates}", file=sys.stderr)

    for col in candidates:
        if col in df.columns:
            print(f"DEBUG: Detected ID column: {col}", file=sys.stderr)
            return col

    lower_map = {c.lower(): c for c in df.columns}
    for col in candidates:
        if col.lower() in lower_map:
            detected = lower_map[col.lower()]
            print(f"DEBUG: Detected ID column by lowercase match: {detected}", file=sys.stderr)
            return detected

    debug_print_autodetect_attempt(df, "detect_id_col failed")

    raise ValueError(
        "Could not automatically detect an ID column. "
        "Pass --id-col explicitly. Available columns:\n"
        + ", ".join(df.columns)
    )


def detect_match_col(df: pd.DataFrame, user_col: str | None = None) -> str:
    if user_col is not None:
        if user_col not in df.columns:
            debug_print_autodetect_attempt(df, "detect_match_col failed with explicit user_col")
            raise ValueError(f"Requested match column '{user_col}' not found.")
        return user_col

    candidates = [
        "matched",
        "match",
        "is_match",
        "structure_match",
        "structure_matched",
        "pymatgen_match",
        "pmg_match",
        "structurematcher_match",
        "structure_matcher_match",
        "sm_match",
        "valid_match",
    ]

    print("\nDEBUG: Trying to detect StructureMatcher match column.", file=sys.stderr)
    print(f"DEBUG: Available columns: {list(df.columns)}", file=sys.stderr)
    print(f"DEBUG: Candidate match columns: {candidates}", file=sys.stderr)

    success_similarity_cols = [
        c for c in df.columns
        if any(tok in c.lower() for tok in ["success", "similarity", "best", "method"])
    ]
    print(
        f"DEBUG: Columns that look relevant but are not currently used by match detection: "
        f"{success_similarity_cols}",
        file=sys.stderr,
    )

    for col in candidates:
        if col in df.columns:
            print(f"DEBUG: Detected match column: {col}", file=sys.stderr)
            return col

    lower_map = {c.lower(): c for c in df.columns}
    for col in candidates:
        if col.lower() in lower_map:
            detected = lower_map[col.lower()]
            print(f"DEBUG: Detected match column by lowercase match: {detected}", file=sys.stderr)
            return detected

    fuzzy = [
        c for c in df.columns
        if "match" in c.lower() and not c.lower().endswith("rate")
    ]

    print(f"DEBUG: Fuzzy match columns found: {fuzzy}", file=sys.stderr)

    if len(fuzzy) == 1:
        print(f"DEBUG: Detected fuzzy match column: {fuzzy[0]}", file=sys.stderr)
        return fuzzy[0]

    debug_print_autodetect_attempt(df, "detect_match_col failed")
    debug_print_schema(df, "CSV where detect_match_col failed")

    raise ValueError(
        "Could not automatically detect a StructureMatcher match column. "
        "Pass --match-col explicitly. Available columns:\n"
        + ", ".join(df.columns)
    )


def match_mask(series: pd.Series) -> pd.Series:
    """
    Convert common match encodings to boolean.

    Accepts:
        True/False
        1/0
        yes/no
        match/no_match
        matched/unmatched
        success/failure
    """
    print("\nDEBUG: Building match mask.", file=sys.stderr)
    print(f"DEBUG: Match column dtype: {series.dtype}", file=sys.stderr)
    print("DEBUG: Match column head:", file=sys.stderr)
    print(series.head(10).to_string(index=False), file=sys.stderr)

    if pd.api.types.is_bool_dtype(series):
        mask = series.fillna(False)
        print(f"DEBUG: Boolean match mask true count: {mask.sum()} / {len(mask)}", file=sys.stderr)
        return mask

    if pd.api.types.is_numeric_dtype(series):
        mask = series.fillna(0).astype(float) > 0
        print(f"DEBUG: Numeric match mask true count: {mask.sum()} / {len(mask)}", file=sys.stderr)
        return mask

    normalized = series.astype(str).str.strip().str.lower()
    mask = normalized.isin(
        {
            "true",
            "t",
            "1",
            "yes",
            "y",
            "match",
            "matched",
            "success",
            "successful",
            "valid",
        }
    )

    print("DEBUG: Normalized match value counts:", file=sys.stderr)
    print(normalized.value_counts(dropna=False).head(20).to_string(), file=sys.stderr)
    print(f"DEBUG: String match mask true count: {mask.sum()} / {len(mask)}", file=sys.stderr)

    return mask


# ───────────────────────── MAE helpers ─────────────────────────
def find_exp_col(df: pd.DataFrame, param: str) -> str | None:
    """Find the ground-truth column for a lattice parameter by trying known prefixes."""
    for prefix in ("rruff_", "alex_", "gt_", "true_", "target_", "exp_"):
        col = f"{prefix}{param}"
        if col in df.columns:
            return col
    return None


def find_pred_col(df: pd.DataFrame, param: str) -> str | None:
    """Find the prediction/refined column for a lattice parameter."""
    candidates = [
        f"pred_{param}",
        f"prediction_{param}",
        f"refined_{param}",
        f"calc_{param}",
    ]

    for col in candidates:
        if col in df.columns:
            return col

    return None


def compute_mae(df: pd.DataFrame, param: str) -> tuple[float, int]:
    """Return (mae, n_valid) for one lattice parameter."""
    exp_col = find_exp_col(df, param)
    pred_col = find_pred_col(df, param)

    print(
        f"DEBUG: compute_mae param={param}: exp_col={exp_col}, pred_col={pred_col}, n_rows={len(df)}",
        file=sys.stderr,
    )

    if exp_col is None or pred_col is None:
        return np.nan, 0

    sub = df[[exp_col, pred_col]].dropna()

    if sub.empty:
        return np.nan, 0

    mae = (sub[pred_col] - sub[exp_col]).abs().mean()
    return float(mae), int(len(sub))


# ───────────────────────── loading / filtering ─────────────────────────
def load_run_csvs(
    runs_root: Path,
    run_keys: list[str],
    debug_schema: bool = False,
) -> dict[tuple[str, str], dict]:
    """
    Load only non-ALIGNN-FF runs corresponding to:
        dataset ∈ {rruff, alex}
        method ∈ {no_refinement, bmgn, gsas2}
    """
    out = {}

    for run_key in run_keys:
        parsed = parse_run_key(run_key)
        if parsed is None:
            continue

        dataset, method = parsed
        run_dir = runs_root / run_key

        if not run_dir.is_dir():
            print(f"WARNING: Missing directory {run_dir} — skipped", file=sys.stderr)
            continue

        csv_path = newest_results_csv(run_dir)
        if csv_path is None:
            print(f"WARNING: No results CSV found in {run_dir} — skipped", file=sys.stderr)
            continue

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"WARNING: Failed to read {csv_path} ({e}) — skipped", file=sys.stderr)
            continue

        out[(dataset, method)] = {
            "run_key": run_key,
            "csv_path": csv_path,
            "df": df,
        }

        print(f"DEBUG: Loaded {run_key} -> {csv_path}", file=sys.stderr)
        print(f"DEBUG: {run_key} shape = {df.shape}", file=sys.stderr)
        print(f"DEBUG: {run_key} columns = {list(df.columns)}", file=sys.stderr)

        if debug_schema:
            debug_print_schema(df, run_key)

    missing = []
    for dataset in DATASET_ORDER:
        for method in METHOD_ORDER:
            if (dataset, method) not in out:
                missing.append(f"{dataset}_{method}")

    if missing:
        print(
            "WARNING: Missing expected non-ALIGNN-FF runs:\n  "
            + "\n  ".join(missing),
            file=sys.stderr,
        )

    return out


def get_dataset_matched_ids(
    run_data: dict,
    dataset: str,
    id_col: str | None,
    match_col: str | None,
    require_all_methods: bool = True,
) -> tuple[set, str, str]:
    """
    Get the vanilla DiffractionGPT matched IDs for one dataset.

    If require_all_methods is True, further restrict to IDs present in all
    available methods for that dataset. This gives a truly paired comparison.
    """
    vanilla_key = (dataset, "no_refinement")

    if vanilla_key not in run_data:
        raise RuntimeError(f"Missing vanilla run for dataset '{dataset}'.")

    vanilla_df = run_data[vanilla_key]["df"]

    print("\n" + "#" * 90, file=sys.stderr)
    print(f"DEBUG: Finding matched vanilla IDs for dataset={dataset}", file=sys.stderr)
    print(f"DEBUG: Vanilla run key = {run_data[vanilla_key]['run_key']}", file=sys.stderr)
    print(f"DEBUG: Vanilla CSV path = {run_data[vanilla_key]['csv_path']}", file=sys.stderr)
    print(f"DEBUG: Vanilla shape = {vanilla_df.shape}", file=sys.stderr)
    print(f"DEBUG: Vanilla columns = {list(vanilla_df.columns)}", file=sys.stderr)
    print("#" * 90, file=sys.stderr)

    debug_print_autodetect_attempt(vanilla_df, f"{dataset} vanilla no-refinement")

    detected_id_col = detect_id_col(vanilla_df, id_col)
    detected_match_col = detect_match_col(vanilla_df, match_col)

    print(f"DEBUG: {dataset}: detected_id_col = {detected_id_col}", file=sys.stderr)
    print(f"DEBUG: {dataset}: detected_match_col = {detected_match_col}", file=sys.stderr)

    mask = match_mask(vanilla_df[detected_match_col])
    matched = vanilla_df.loc[mask, detected_id_col]
    matched_ids = set(matched.dropna().astype(str))

    print(
        f"DEBUG: {dataset}: vanilla matched IDs before paired intersection = "
        f"{len(matched_ids)}",
        file=sys.stderr,
    )

    print(f"DEBUG: {dataset}: first 20 matched IDs:", file=sys.stderr)
    for mid in list(sorted(matched_ids))[:20]:
        print(f"  {mid}", file=sys.stderr)

    if require_all_methods:
        print(
            f"DEBUG: {dataset}: restricting matched IDs to IDs present in all available methods.",
            file=sys.stderr,
        )

        for method in METHOD_ORDER:
            key = (dataset, method)

            if key not in run_data:
                print(
                    f"DEBUG: {dataset}/{method}: missing run_data entry; skipped intersection.",
                    file=sys.stderr,
                )
                continue

            df = run_data[key]["df"]
            this_id_col = detect_id_col(df, id_col)
            this_ids = set(df[this_id_col].dropna().astype(str))

            before = len(matched_ids)
            matched_ids &= this_ids
            after = len(matched_ids)

            print(
                f"DEBUG: {dataset}/{method}: id_col={this_id_col}, "
                f"rows={len(df)}, unique_ids={len(this_ids)}, "
                f"intersection {before} -> {after}",
                file=sys.stderr,
            )

    print(
        f"DEBUG: {dataset}: final matched paired ID count = {len(matched_ids)}",
        file=sys.stderr,
    )

    return matched_ids, detected_id_col, detected_match_col


def build_matched_summary(
    run_data: dict,
    id_col: str | None,
    match_col: str | None,
    require_all_methods: bool = True,
) -> tuple[pd.DataFrame, dict[str, set]]:
    rows = []
    matched_ids_by_dataset = {}

    for dataset in DATASET_ORDER:
        if (dataset, "no_refinement") not in run_data:
            print(
                f"WARNING: No no-refinement run found for dataset={dataset}; skipped.",
                file=sys.stderr,
            )
            continue

        matched_ids, detected_id_col, detected_match_col = get_dataset_matched_ids(
            run_data=run_data,
            dataset=dataset,
            id_col=id_col,
            match_col=match_col,
            require_all_methods=require_all_methods,
        )

        matched_ids_by_dataset[dataset] = matched_ids

        print(
            f"DEBUG: {dataset}: using {len(matched_ids)} vanilla-matched IDs "
            f"from id_col='{detected_id_col}', match_col='{detected_match_col}'",
            file=sys.stderr,
        )

        if not matched_ids:
            print(
                f"WARNING: No matched vanilla IDs found for {dataset}.",
                file=sys.stderr,
            )
            continue

        for method in METHOD_ORDER:
            key = (dataset, method)

            if key not in run_data:
                print(
                    f"WARNING: Missing run_data for {dataset}/{method}; skipped.",
                    file=sys.stderr,
                )
                continue

            df = run_data[key]["df"].copy()
            this_id_col = detect_id_col(df, id_col)

            df["_id_as_str"] = df[this_id_col].astype(str)
            sub = df[df["_id_as_str"].isin(matched_ids)].copy()

            print("\n" + "~" * 90, file=sys.stderr)
            print(f"DEBUG: Building MAE row for {dataset}/{method}", file=sys.stderr)
            print(f"DEBUG: run_key = {run_data[key]['run_key']}", file=sys.stderr)
            print(f"DEBUG: csv_path = {run_data[key]['csv_path']}", file=sys.stderr)
            print(f"DEBUG: id_col = {this_id_col}", file=sys.stderr)
            print(f"DEBUG: original rows = {len(df)}", file=sys.stderr)
            print(f"DEBUG: rows after matched-ID filter = {len(sub)}", file=sys.stderr)
            print(f"DEBUG: unique IDs after filter = {sub['_id_as_str'].nunique()}", file=sys.stderr)

            if len(sub) > 0:
                preview_cols = [
                    c for c in [
                        this_id_col,
                        "rruff_id",
                        "entry_id",
                        "dg_success",
                        "dg_similarity",
                        "pm_success",
                        "pm_similarity",
                        "best_method",
                        "best_similarity",
                        "pred_a",
                        "pred_b",
                        "pred_c",
                        "pred_alpha",
                        "pred_beta",
                        "pred_gamma",
                        "rruff_a",
                        "rruff_b",
                        "rruff_c",
                        "rruff_alpha",
                        "rruff_beta",
                        "rruff_gamma",
                        "alex_a",
                        "alex_b",
                        "alex_c",
                        "alex_alpha",
                        "alex_beta",
                        "alex_gamma",
                    ]
                    if c in sub.columns
                ]
                print("DEBUG: Filtered subset preview:", file=sys.stderr)
                print(sub[preview_cols].head(5).to_string(index=False), file=sys.stderr)

            print("~" * 90 + "\n", file=sys.stderr)

            rec = {
                "dataset": dataset,
                "method": method,
                "method_label": METHOD_ORDER[method],
                "cluster_label": f"{dataset.upper()} / {METHOD_ORDER[method]}",
                "run_key": run_data[key]["run_key"],
                "csv_path": str(run_data[key]["csv_path"]),
                "id_col": this_id_col,
                "n_vanilla_matched_ids": len(matched_ids),
                "n_rows_after_filter": len(sub),
            }

            for p in PARAMS:
                mae, n_valid = compute_mae(sub, p)
                rec[f"MAE.{p}"] = mae
                rec[f"N.{p}"] = n_valid

            rows.append(rec)

    if not rows:
        print("ERROR: No rows available for matched-subset summary.", file=sys.stderr)
        sys.exit(1)

    summary = pd.DataFrame(rows)

    dataset_cat = pd.CategoricalDtype(DATASET_ORDER, ordered=True)
    method_cat = pd.CategoricalDtype(list(METHOD_ORDER.keys()), ordered=True)

    summary["dataset"] = summary["dataset"].astype(dataset_cat)
    summary["method"] = summary["method"].astype(method_cat)
    summary = summary.sort_values(["dataset", "method"]).reset_index(drop=True)

    return summary, matched_ids_by_dataset


# ───────────────────────── plotting ─────────────────────────
def summary_to_mae_df(summary: pd.DataFrame) -> pd.DataFrame:
    mae_cols = [f"MAE.{p}" for p in PARAMS]
    return (
        summary.set_index("cluster_label")[mae_cols]
        .rename(columns=lambda c: ax_label_map[c.split(".")[-1]])
    )


def draw_matched_subset_plot(
    summary: pd.DataFrame,
    output_path: Path,
    title: str,
    figsize=(14, 8),
):
    mae_df = summary_to_mae_df(summary)

    labels = list(mae_df.index)
    x = np.arange(len(labels))
    width = 0.12

    display_len = [ax_label_map[p] for p in length_params]
    display_ang = [ax_label_map[p] for p in angle_params]

    offsets = {
        display_len[0]: -2.5 * width,
        display_len[1]: -1.5 * width,
        display_len[2]: -0.5 * width,
        display_ang[0]:  0.5 * width,
        display_ang[1]:  1.5 * width,
        display_ang[2]:  2.5 * width,
    }

    fig, ax = plt.subplots(figsize=figsize)
    ax2 = ax.twinx()
    ax2.patch.set_alpha(0.0)

    for p, col in zip(length_params, display_len):
        ax.bar(
            x + offsets[col],
            mae_df[col].values,
            width=width,
            edgecolor="k",
            color=param_colors[p],
            label=f"{col} (Å)",
            zorder=3,
        )

    for p, col in zip(angle_params, display_ang):
        ax2.bar(
            x + offsets[col],
            mae_df[col].values,
            width=width,
            edgecolor="k",
            color=param_colors[p],
            label=f"{col} (°)",
            zorder=3,
        )

    len_max = (
        np.nanmax(mae_df[display_len].values)
        if not mae_df[display_len].isna().all().all()
        else 1.0
    )
    ang_max = (
        np.nanmax(mae_df[display_ang].values)
        if not mae_df[display_ang].isna().all().all()
        else 1.0
    )

    ax.set_ylim(0, 1.35 * len_max if len_max > 0 else 1.0)
    ax2.set_ylim(0, 1.35 * ang_max if ang_max > 0 else 1.0)

    ax.set_ylabel("Mean Absolute Error — lengths (Å)", fontsize=16)
    ax2.set_ylabel("Mean Absolute Error — angles (°)", fontsize=16)
    ax.set_title(title, fontsize=18)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=13)

    ax.tick_params(axis="y", labelsize=13)
    ax2.tick_params(axis="y", labelsize=13)

    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    if len(labels) == 6:
        ax.axvline(2.5, color="k", linewidth=1.0, alpha=0.35)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()

    ax.legend(
        h1 + h2,
        l1 + l2,
        title="Lattice parameter",
        title_fontsize=13,
        fontsize=12,
        ncol=2,
        loc="upper right",
        frameon=True,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"DEBUG: Saved plot -> {output_path}", file=sys.stderr)


# ───────────────────────── main workflow ─────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compute lattice-parameter MAE for only the vanilla "
            "DiffractionGPT predictions that matched the ground truth by "
            "StructureMatcher, then compare no-refinement, BMGN, and GSAS-II "
            "on that same ID subset. ALIGNN-FF runs are excluded."
        )
    )

    parser.add_argument(
        "dir_list",
        type=Path,
        help="Text file containing refinement directory names.",
    )

    parser.add_argument(
        "--id-col",
        default=None,
        help="Optional explicit ID column shared across result CSVs.",
    )

    parser.add_argument(
        "--match-col",
        default=None,
        help=(
            "Optional explicit StructureMatcher boolean column in the vanilla "
            "no-refinement CSVs."
        ),
    )

    parser.add_argument(
        "--allow-missing-method-ids",
        action="store_true",
        help=(
            "By default, the matched ID subset is further restricted to IDs "
            "present in all three methods for a dataset. This flag disables "
            "that stricter paired-ID intersection."
        ),
    )

    parser.add_argument(
        "--output-prefix",
        default="matched_vanilla_refinement_mae_noalignn",
        help="Prefix for output PNG/CSV files written into runs/.",
    )

    parser.add_argument(
        "--debug-schema",
        action="store_true",
        help=(
            "Print detailed schema diagnostics for every loaded CSV before "
            "trying to compute the matched subset."
        ),
    )

    args = parser.parse_args()

    root = Path.cwd()
    runs_root = find_runs_root(root)
    print(f"DEBUG: Using runs directory: {runs_root}", file=sys.stderr)

    all_run_keys = load_dir_list(args.dir_list)

    selected_run_keys = [
        key for key in all_run_keys
        if key in EXPECTED_RUNS
    ]

    if not selected_run_keys:
        print(
            "ERROR: No expected non-ALIGNN-FF directories found in dir list.",
            file=sys.stderr,
        )
        sys.exit(1)

    print("DEBUG: Selected runs:", file=sys.stderr)
    for key in selected_run_keys:
        print(f"  - {key}", file=sys.stderr)

    run_data = load_run_csvs(
        runs_root=runs_root,
        run_keys=selected_run_keys,
        debug_schema=args.debug_schema,
    )

    summary, matched_ids_by_dataset = build_matched_summary(
        run_data=run_data,
        id_col=args.id_col,
        match_col=args.match_col,
        require_all_methods=not args.allow_missing_method_ids,
    )

    summary_path = runs_root / f"{args.output_prefix}.csv"
    plot_path = runs_root / f"{args.output_prefix}.png"

    summary.to_csv(summary_path, index=False)
    print(f"DEBUG: Saved summary -> {summary_path}", file=sys.stderr)

    for dataset, ids in matched_ids_by_dataset.items():
        id_path = runs_root / f"{args.output_prefix}_{dataset}_ids.txt"
        id_path.write_text("\n".join(sorted(ids)) + "\n")
        print(f"DEBUG: Saved matched IDs for {dataset} -> {id_path}", file=sys.stderr)

    draw_matched_subset_plot(
        summary=summary,
        output_path=plot_path,
        title="Refinement MAE on Vanilla DiffractionGPT StructureMatcher-Matched Subset",
        figsize=(15, 8),
    )

    display_cols = (
        ["dataset", "method_label", "run_key", "n_vanilla_matched_ids", "n_rows_after_filter"]
        + [f"MAE.{p}" for p in PARAMS]
        + [f"N.{p}" for p in PARAMS]
        + ["csv_path"]
    )

    print("\nMatched-subset MAE summary:")
    print(summary[display_cols].to_string(index=False))

    print(f"\nSaved plot:    {plot_path}")
    print(f"Saved summary: {summary_path}")

    for dataset in DATASET_ORDER:
        if dataset in matched_ids_by_dataset:
            print(
                f"Saved {dataset.upper()} matched IDs: "
                f"{runs_root / f'{args.output_prefix}_{dataset}_ids.txt'}"
            )

    print("DEBUG: All done.", file=sys.stderr)


if __name__ == "__main__":
    main()
