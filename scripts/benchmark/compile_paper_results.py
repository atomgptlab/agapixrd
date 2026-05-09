#!/usr/bin/env python3
"""Compile manuscript-ready AGAPI-XRD result summaries.

Usage:
    python scripts/benchmark/compile_paper_results.py dir_lists/all12_dirs.txt

The directory list is a plain text file with one run directory per line. Blank
lines and comments beginning with "#" are ignored. Entries may be run names
under --runs-root, paths relative to the current working directory, or absolute
paths.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import pandas as pd

from compile_agapi_replication_metrics import (
    CRYSTAL_SYSTEM_ORDER,
    PARAMS_ALL,
    PARAMS_METHOD,
    build_cross_run_summary,
    jsonable,
    load_run_dataframe,
    read_run_list,
    resolve_run_dir,
    summarize_run,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compile all AGAPI-XRD paper result summaries from a run directory list.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "dirs_list",
        help="Text file listing run directories to include, for example dir_lists/all12_dirs.txt.",
    )
    parser.add_argument(
        "--runs-root",
        default="runs",
        help="Root directory used to resolve bare run names from dirs_list.",
    )
    parser.add_argument(
        "--output-dir",
        default="paper_results",
        help="Directory where JSON and CSV summaries will be written.",
    )
    parser.add_argument(
        "--prefix",
        default="paper_results",
        help="Filename prefix for generated summary files.",
    )
    parser.add_argument(
        "--jsd-bins",
        type=int,
        default=30,
        help="Number of histogram bins for Jensen-Shannon divergence calculations.",
    )
    parser.add_argument(
        "--laplace-alpha",
        type=float,
        default=1e-8,
        help="Laplace smoothing constant for histogram-based JSD calculations.",
    )
    parser.add_argument(
        "--relative-error-mode",
        choices=["entrywise_mean", "pooled_lengths"],
        default="entrywise_mean",
        help="How to compute relative-error threshold fractions for a,b,c.",
    )
    parser.add_argument(
        "--raw-rruff-count",
        type=int,
        default=1362,
        help="Study-context count for raw RRUFF entries before filtering.",
    )
    parser.add_argument(
        "--unique-valid-chemistry-count",
        type=int,
        default=803,
        help="Study-context count for RRUFF entries with unique valid chemistry.",
    )
    return parser.parse_args()


def pct(value: Any) -> Any:
    if value is None:
        return None
    try:
        x = float(value)
    except Exception:
        return None
    if math.isnan(x):
        return None
    return 100.0 * x


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


def build_payload(
    dirs_list: Path,
    runs_root: Path,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    metric_args = SimpleNamespace(
        jsd_bins=args.jsd_bins,
        laplace_alpha=args.laplace_alpha,
        relative_error_mode=args.relative_error_mode,
        raw_rruff_count=args.raw_rruff_count,
        unique_valid_chemistry_count=args.unique_valid_chemistry_count,
    )

    run_specs = read_run_list(dirs_list)
    run_dataframes: Dict[str, pd.DataFrame] = {}
    run_summaries: Dict[str, Dict[str, Any]] = {}

    for spec in run_specs:
        run_dir = resolve_run_dir(spec, runs_root)
        run_name = run_dir.name
        if run_name in run_dataframes:
            raise ValueError(f"Duplicate run name after resolution: {run_name}")
        df, provenance = load_run_dataframe(run_dir)
        run_dataframes[run_name] = df
        run_summaries[run_name] = summarize_run(
            run_name=run_name,
            run_dir=run_dir,
            df=df,
            provenance=provenance,
            args=metric_args,
        )

    return {
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "dirs_list": str(dirs_list),
            "runs_root": str(runs_root),
            "requested_runs": run_specs,
            "resolved_run_names": list(run_dataframes.keys()),
            "jsd_bins": args.jsd_bins,
            "laplace_alpha": args.laplace_alpha,
            "relative_error_mode": args.relative_error_mode,
            "raw_rruff_count": args.raw_rruff_count,
            "unique_valid_chemistry_count": args.unique_valid_chemistry_count,
        },
        "runs": run_summaries,
        "cross_run_summary": build_cross_run_summary(
            run_dataframes=run_dataframes,
            run_summaries=run_summaries,
            args=metric_args,
        ),
    }


def flatten_run_overview(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for run_name in payload["cross_run_summary"]["run_order"]:
        summary = payload["runs"][run_name]
        source = summary["source"]
        bench = summary["benchmark_summary"]
        cov = summary["coverage_metrics"]
        rel = summary["distribution_threshold_metrics"]
        ref = summary["refinement_quality"]
        runtime = summary["runtime_and_failures"]
        lattice = summary["overall_lattice_accuracy"]

        row: Dict[str, Any] = {
            "run_name": run_name,
            "ground_truth_dataset": source.get("ground_truth_dataset"),
            "primary_file": source.get("primary_file"),
            "source_format": source.get("format"),
            "total_entries": bench.get("total_entries_processed"),
            "complete_lattice_entries": bench.get("entries_with_complete_lattice"),
            "has_crystal_system_labels": bench.get("has_crystal_system_labels"),
            "pm_total_count": cov["pattern_matching_total"]["count"],
            "pm_total_pct": cov["pattern_matching_total"]["pct"],
            "pm_formula_count": cov["pattern_matching_formula"]["count"],
            "pm_formula_pct": cov["pattern_matching_formula"]["pct"],
            "pm_elements_count": cov["pattern_matching_elements"]["count"],
            "pm_elements_pct": cov["pattern_matching_elements"]["pct"],
            "diffractgpt_count": cov["diffractgpt"]["count"],
            "diffractgpt_pct": cov["diffractgpt"]["pct"],
            "combined_any_count": cov["combined_any_method"]["count"],
            "combined_any_pct": cov["combined_any_method"]["pct"],
            "unmatched_count": cov["unmatched"]["count"],
            "unmatched_pct": cov["unmatched"]["pct"],
            "rel_length_within_10pct": pct(rel.get("fraction_predictions_within_10pct")),
            "rel_length_within_30pct": pct(rel.get("fraction_predictions_within_30pct")),
            "n_refined_success": ref.get("n_refined_success"),
            "mean_rwp": ref.get("mean_rwp"),
            "median_rwp": ref.get("median_rwp"),
            "num_errors": runtime.get("num_errors"),
            "median_time_s": runtime.get("median_time_s"),
        }
        for param in PARAMS_ALL:
            block = lattice[param]
            row[f"{param}_n"] = block.get("n")
            row[f"{param}_mae"] = block.get("mae")
            row[f"{param}_rmse"] = block.get("rmse")
            row[f"{param}_r2"] = block.get("r2")
            row[f"{param}_skill"] = block.get("skill")
            row[f"{param}_jsd"] = block.get("jsd")
        rows.append(row)
    return rows


def flatten_lattice_metrics(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for run_name in payload["cross_run_summary"]["run_order"]:
        summary = payload["runs"][run_name]
        dataset = summary["source"].get("ground_truth_dataset")
        for param in PARAMS_ALL:
            block = summary["overall_lattice_accuracy"][param]
            rows.append({
                "run_name": run_name,
                "ground_truth_dataset": dataset,
                "parameter": param,
                "n": block.get("n"),
                "mae": block.get("mae"),
                "rmse": block.get("rmse"),
                "r2": block.get("r2"),
                "median_abs_error": block.get("median_abs_error"),
                "mad_exp": block.get("mad_exp"),
                "skill": block.get("skill"),
                "jsd": block.get("jsd"),
            })
    return rows


def flatten_method_breakdown(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for run_name in payload["cross_run_summary"]["run_order"]:
        summary = payload["runs"][run_name]
        dataset = summary["source"].get("ground_truth_dataset")
        for method, method_block in summary["method_breakdown"].items():
            for param in PARAMS_METHOD:
                metric = method_block[param]
                rows.append({
                    "run_name": run_name,
                    "ground_truth_dataset": dataset,
                    "method": method,
                    "method_n": method_block.get("n"),
                    "parameter": param,
                    "n": metric.get("n"),
                    "mae": metric.get("mae"),
                    "skill": metric.get("skill"),
                })
    return rows


def flatten_crystal_system_breakdown(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for run_name in payload["cross_run_summary"]["run_order"]:
        summary = payload["runs"][run_name]
        dataset = summary["source"].get("ground_truth_dataset")
        for crystal_system in CRYSTAL_SYSTEM_ORDER:
            block = summary["crystal_system_breakdown"][crystal_system]
            rows.append({
                "run_name": run_name,
                "ground_truth_dataset": dataset,
                "crystal_system": crystal_system,
                "n": block.get("n"),
                "a_mae": block["a"].get("mae"),
                "a_skill": block["a"].get("skill"),
                "volume_mae": block["volume"].get("mae"),
                "volume_skill": block["volume"].get("skill"),
            })
    return rows


def flatten_pairwise_jsd(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    pairwise = payload["cross_run_summary"]["pairwise_run_comparisons"]
    for key, block in pairwise.items():
        left, right = block["pair"]
        for param, metric in block["pairwise_predicted_distribution_jsd"].items():
            rows.append({
                "comparison": key,
                "left_run": left,
                "right_run": right,
                "parameter": param,
                "n_left": metric.get("n_run_a"),
                "n_right": metric.get("n_run_b"),
                "jsd": metric.get("jsd"),
            })
    return rows


def flatten_dataset_groups(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for dataset, block in payload["cross_run_summary"]["dataset_groups"].items():
        rows.append({
            "ground_truth_dataset": dataset,
            "n_runs": block.get("n_runs"),
            "common_entry_count": block.get("common_entry_count"),
            "run_order": ";".join(block.get("run_order", [])),
            "total_entries_by_run": json.dumps(block.get("total_entries_by_run", {}), sort_keys=True),
        })
    return rows


def write_outputs(payload: Dict[str, Any], output_dir: Path, prefix: str) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": output_dir / f"{prefix}_summary.json",
        "run_overview": output_dir / f"{prefix}_run_overview.csv",
        "lattice_metrics": output_dir / f"{prefix}_lattice_metrics.csv",
        "method_breakdown": output_dir / f"{prefix}_method_breakdown.csv",
        "crystal_system_breakdown": output_dir / f"{prefix}_crystal_system_breakdown.csv",
        "pairwise_jsd": output_dir / f"{prefix}_pairwise_jsd.csv",
        "dataset_groups": output_dir / f"{prefix}_dataset_groups.csv",
    }

    paths["json"].write_text(json.dumps(jsonable(payload), indent=2))
    write_csv(flatten_run_overview(payload), paths["run_overview"])
    write_csv(flatten_lattice_metrics(payload), paths["lattice_metrics"])
    write_csv(flatten_method_breakdown(payload), paths["method_breakdown"])
    write_csv(flatten_crystal_system_breakdown(payload), paths["crystal_system_breakdown"])
    write_csv(flatten_pairwise_jsd(payload), paths["pairwise_jsd"])
    write_csv(flatten_dataset_groups(payload), paths["dataset_groups"])
    return paths


def main() -> None:
    args = parse_args()
    dirs_list = Path(args.dirs_list).resolve()
    runs_root = Path(args.runs_root).resolve()
    output_dir = Path(args.output_dir).resolve()

    payload = build_payload(dirs_list=dirs_list, runs_root=runs_root, args=args)
    paths = write_outputs(payload=payload, output_dir=output_dir, prefix=args.prefix)

    run_order = payload["cross_run_summary"]["run_order"]
    print(f"Compiled {len(run_order)} runs from {dirs_list}")
    for name in run_order:
        summary = payload["runs"][name]
        dataset = summary["source"].get("ground_truth_dataset")
        entries = summary["benchmark_summary"].get("total_entries_processed")
        complete = summary["benchmark_summary"].get("entries_with_complete_lattice")
        coverage = summary["coverage_metrics"]["combined_any_method"]["pct"]
        print(
            f"  {name}: dataset={dataset}, entries={entries}, "
            f"complete_lattice={complete}, combined_coverage={coverage:.1f}%"
        )

    print("Wrote:")
    for path in paths.values():
        print(f"  {path}")


if __name__ == "__main__":
    main()
