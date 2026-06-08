#!/usr/bin/env python3
"""Publication-quality learning-curve plots from aggregated transductive finetuning results.

Mirrors the logic of ``plot_results.py`` for the transductive (node-level) setting.
All plotting functions are imported directly from ``plot_results.py`` so that any
visual improvements there are automatically inherited.  Only the paths and the
``main()`` entry point differ:

  * Input CSV:  ``outputs_transductive/aggregated_results.csv``
  * Figures:    ``outputs_transductive/figures/``

Run the full data-processing pipeline first::

    python scripts/plotting/run_all_transductive.py --entity <wandb-entity>

Then generate figures::

    python scripts/plotting/plot_results_transductive.py
    python scripts/plotting/plot_results_transductive.py --input path/to/aggregated_results.csv
    python scripts/plotting/plot_results_transductive.py --output-dir my_figures/
    python scripts/plotting/plot_results_transductive.py --fmt pdf

Note: metadata / graph-property correlation plots from ``plot_results.py`` are omitted
here because those analyses rely on graph-level structural properties that are not
meaningful for node-level transductive datasets.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

# ── Re-use all plot functions from the inductive pipeline ─────────────────────
# These functions operate on DataFrames and accept explicit output_dir arguments,
# so they work unchanged for transductive data as long as the CSV schema is the same
# (which it is — process_project_transductive.py produces the identical column layout).
from scripts.plotting.plot_results import (
    plot_dataset,
    plot_all_datasets,
    plot_all_datasets_percentage,
    plot_dataset_percentage,
    plot_improvement_barplots,
    plot_average_improvement,
    _generate_frac_bars,
    COL_DATASET,
    COL_FRACTION,
)
from scripts.plotting.shared_baseline import apply_shared_random_init_baseline

import pandas as pd

# ── Transductive output paths ─────────────────────────────────────────────────
_OUTPUTS_BASE_T    = _SCRIPT_DIR / "outputs_transductive"
DEFAULT_INPUT_T    = _OUTPUTS_BASE_T / "aggregated_results.csv"
DEFAULT_OUTPUT_DIR_T = _OUTPUTS_BASE_T / "figures"


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot aggregated transductive finetuning results.",
    )
    p.add_argument(
        "--input", type=Path, default=None,
        help="Path to aggregated_results.csv "
             "(default: outputs_transductive/aggregated_results.csv).",
    )
    p.add_argument(
        "--output-dir", type=Path, default=None,
        help="Directory to write figures into "
             "(default: outputs_transductive/figures/).",
    )
    p.add_argument(
        "--fmt", default="png", choices=["png", "pdf", "svg"],
        help="Output format (default: png).",
    )
    p.add_argument(
        "--datasets", nargs="+", default=None,
        help="Restrict to specific dataset names (default: all in the CSV).",
    )
    p.add_argument(
        "--no-curves", action="store_true",
        help="Skip per-dataset learning-curve figures.",
    )
    p.add_argument(
        "--no-master", action="store_true",
        help="Skip the combined all-datasets master learning-curve figure.",
    )
    p.add_argument(
        "--no-pct-master", action="store_true",
        help="Skip the combined all-datasets percentage learning-curve figure.",
    )
    p.add_argument(
        "--no-percentage", action="store_true",
        help="Skip percentage-normalised per-dataset learning-curve figures.",
    )
    p.add_argument(
        "--no-barplots", action="store_true",
        help="Skip per-(fraction, scope) grouped improvement bar charts.",
    )
    p.add_argument(
        "--no-avg-improvement", action="store_true",
        help="Skip the average-improvement summary figure.",
    )
    p.add_argument(
        "--no-bars-frac", action="store_true",
        help="Skip per-fraction bar-chart figures.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    input_path = args.input      or DEFAULT_INPUT_T
    output_dir = args.output_dir or DEFAULT_OUTPUT_DIR_T

    print(f"Setting  : transductive (node-level)")
    print(f"Input    : {input_path}")
    print(f"Out dir  : {output_dir}")

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}\n"
            f"Run 'python scripts/plotting/run_all_transductive.py --entity <entity>' first."
        )

    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} rows from {input_path}")
    df = apply_shared_random_init_baseline(df)

    if COL_DATASET not in df.columns:
        raise ValueError(
            f"Expected column {COL_DATASET!r} not found. "
            f"Available columns: {list(df.columns)}"
        )

    datasets = args.datasets or sorted(df[COL_DATASET].dropna().unique())
    print(f"Datasets : {datasets}")

    # ── Per-dataset learning-curve figures ────────────────────────────────────
    if not args.no_curves:
        print(f"\n── Learning-curve plots ({len(datasets)} dataset(s)) ──")
        for dataset in datasets:
            print(f"  [{dataset}]")
            try:
                plot_dataset(df, dataset, output_dir, fmt=args.fmt)
            except Exception as exc:
                print(f"    ERROR: {exc}")

    # ── Master all-datasets learning-curve grid ───────────────────────────────
    if not args.no_master:
        print(f"\n── Master all-datasets learning-curve figure(s) ──")
        for scope in (None, "probe", "full"):
            label = "combined" if scope is None else scope
            try:
                plot_all_datasets(
                    df, output_dir, fmt=args.fmt,
                    datasets=datasets, scope=scope,
                )
            except Exception as exc:
                print(f"  ERROR ({label}): {exc}")

    # ── Master all-datasets percentage figure ─────────────────────────────────
    if not args.no_pct_master:
        print(f"\n── Master all-datasets percentage learning-curve figure(s) ──")
        for scope in (None, "probe", "full"):
            label = "combined" if scope is None else scope
            try:
                plot_all_datasets_percentage(
                    df, output_dir, fmt=args.fmt,
                    datasets=datasets, scope=scope,
                )
            except Exception as exc:
                print(f"  ERROR ({label}): {exc}")

    # ── Per-dataset percentage-normalised figures ─────────────────────────────
    if not args.no_percentage:
        pct_dir = output_dir / "percentage_results"
        print(f"\n── Percentage-normalised learning curves ({len(datasets)} dataset(s)) ──")
        for dataset in datasets:
            print(f"  [{dataset}]")
            try:
                plot_dataset_percentage(df, dataset, pct_dir, fmt=args.fmt)
            except Exception as exc:
                print(f"    ERROR: {exc}")

    # ── Per-fraction bar charts ───────────────────────────────────────────────
    if not args.no_bars_frac:
        n_fracs = len(df[COL_FRACTION].dropna().unique())
        print(f"\n── Per-fraction bar charts ({n_fracs} fraction(s)) ──")
        try:
            _generate_frac_bars(df, output_dir, fmt=args.fmt, datasets=datasets)
        except Exception as exc:
            print(f"  ERROR: {exc}")

    # ── Improvement bar charts ────────────────────────────────────────────────
    if not args.no_barplots:
        bars_dir = output_dir / "improvement_bars"
        fractions = sorted(df[COL_FRACTION].dropna().unique())
        n_figs    = len(fractions) * 2
        print(f"\n── Improvement bar charts ({n_figs} figure(s)) ──")
        try:
            plot_improvement_barplots(df, bars_dir, fmt=args.fmt, datasets=datasets)
        except Exception as exc:
            print(f"  ERROR: {exc}")

    # ── Average-improvement summary ───────────────────────────────────────────
    if not args.no_avg_improvement:
        avg_dir = output_dir / "improvement_bars"
        print("\n── Average-improvement summary figure ──")
        try:
            plot_average_improvement(df, avg_dir, fmt=args.fmt, datasets=datasets)
        except Exception as exc:
            print(f"  ERROR: {exc}")

    print(f"\nDone. Figures in: {output_dir}")


if __name__ == "__main__":
    main()
