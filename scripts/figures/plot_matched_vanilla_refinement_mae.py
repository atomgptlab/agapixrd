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


def find_runs_root(start: Path) -> Path:
    if (start / "runs").is_dir():
        return start / "runs"
    if start.name == "runs" and start.is_dir():
        return start
    raise FileNotFoundError(
        "Could not find a runs/ directory. Run this from the project root "
        "or from inside runs/."
    )


def load_dir_list(path: Path) -> list[str]:
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

    ALIGNN-FF directories are intentionally excluded elsewhere.
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


def detect_id_col(df: pd.DataFrame, user_col: str | None = None) -> str:
    if user_col is not None:
        if user_col not in df.columns:
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

    for col in candidates:
        if col in df.columns:
            return col

    lower_map = {c.lower(): c for c in df.columns}
    for col in candidates:
        if col.lower() in lower_map:
            return lower_map[col.lower()]

    raise ValueError(
        "Could not automatically detect an ID column. "
        "Pass --id-col explicitly. Available columns:\n"
        + ", ".join(df.columns)
    )


def detect_match_col(df: pd.DataFrame, user_col: str | None = None) -> str:
    if user_col is not None:
        if user_col not in df.columns:
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

    for col in candidates:
        if col in df.columns:
            return col

    lower_map = {c.lower(): c for c in df.columns}
    for col in candidates:
        if col.lower() in lower_map:
            return lower_map[col.lower()]

    fuzzy = [
        c for c in df.columns
        if "match" in c.lower() and not c.lower().endswith("rate")
    ]
    if len(fuzzy) == 1:
        return fuzzy[0]

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
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(float) > 0

    normalized = series.astype(str).str.strip().str.lower()
    return normalized.isin(
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


def find_exp_col(df: pd.DataFrame, param: str) -> str | None:
    for prefix in ("rruff_", "alex_", "gt_", "true_", "target_", "exp_"):
        col = f"{prefix}{param}"
        if col in df.columns:
            return col
    return None


def find_pred_col(df: pd.DataFrame, param: str) -> str | None:
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
    exp_col = find_exp_col(df, param)
    pred_col = find_pred_col(df, param)

    if exp_col is None or pred_col is None:
        return np.nan, 0

    sub = df[[exp_col, pred_col]].dropna()
    if sub.empty:
        return np.nan, 0

    mae = (sub[pred_col] - sub[exp_col]).abs().mean()
    return float(mae), int(len(sub))


def load_run_csvs(
    runs_root: Path,
    run_keys: list[str],
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

    detected_id_col = detect_id_col(vanilla_df, id_col)
    detected_match_col = detect_match_col(vanilla_df, match_col)

    matched = vanilla_df.loc[match_mask(vanilla_df[detected_match_col]), detected_id_col]
    matched_ids = set(matched.dropna().astype(str))

    if require_all_methods:
        for method in METHOD_ORDER:
            key = (dataset, method)
            if key not in run_data:
                continue

            df = run_data[key]["df"]
            this_id_col = detect_id_col(df, id_col)
            this_ids = set(df[this_id_col].dropna().astype(str))
            matched_ids &= this_ids

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
                continue

            df = run_data[key]["df"].copy()
            this_id_col = detect_id_col(df, id_col)

            df["_id_as_str"] = df[this_id_col].astype(str)
            sub = df[df["_id_as_str"].isin(matched_ids)].copy()

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

    # Visual separator between RRUFF and Alex clusters.
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

    args = parser.parse_args()

    root = Path.cwd()
    runs_root = find_runs_root(root)
    print(f"DEBUG: Using runs directory: {runs_root}", file=sys.stderr)

    all_run_keys = load_dir_list(args.dir_list)

    # Keep only the six non-ALIGNN-FF directories relevant to this analysis.
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

    # Save the actual ID subsets too. This is useful for auditing whether
    # the seed/prediction correspondence assumption holds.
    for dataset, ids in matched_ids_by_dataset.items():
        id_path = runs_root / f"{args.output_prefix}_{dataset}_ids.txt"
        id_path.write_text("\n".join(sorted(ids)) + "\n")

    draw_matched_subset_plot(
        summary=summary,
        output_path=plot_path,
        title=(
            "Refinement MAE on Vanilla DiffractionGPT StructureMatcher-Matched Subset"
        ),
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
