#!/usr/bin/env python3
"""Merge per-project CSVs into one table (union of columns, NaN where missing).

** seed_subsample pipeline **
Input CSVs were produced by ``process_project.py`` which already filtered to
``ft_seed_subsample == True`` runs only.

Reads every ``*.csv`` in ``outputs/processed_projects/`` except ``*_flagged.csv``
and writes ``outputs/aggregated_results.csv``.

Usage
-----
    python scripts/plotting/combine_results.py
    python scripts/plotting/combine_results.py --input-dir path/to/processed_projects
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_SCRIPT_DIR  = Path(__file__).resolve().parent
_OUTPUTS_BASE = _SCRIPT_DIR / "outputs"
DEFAULT_INPUT_DIR   = _OUTPUTS_BASE / "processed_projects"   # legacy / fallback
DEFAULT_OUTPUT_PATH = _OUTPUTS_BASE / "aggregated_results.csv"  # legacy / fallback

sys.path.insert(0, str(_SCRIPT_DIR.parents[1]))

from scripts.plotting.shared_baseline import apply_shared_random_init_baseline
from scripts.plotting.wb_table import order_columns


def combine_processed_csvs(
    input_dir: Path,
    output_path: Path,
) -> pd.DataFrame:
    """Concatenate per-project files; same name → same column, else new column with NaN."""
    paths = sorted(
        p for p in input_dir.glob("*.csv")
        if p.is_file() and not p.name.endswith("_flagged.csv")
    )
    if not paths:
        print(
            f"  [combine] WARNING: no processed CSV files found in {input_dir}.\n"
            f"  [combine] This likely means all projects were filtered out because\n"
            f"  [combine] no runs had ft_seed_subsample=True.  Nothing to combine."
        )
        return pd.DataFrame()

    frames = []
    for p in paths:
        try:
            df = pd.read_csv(p)
        except pd.errors.EmptyDataError:
            continue
        if not df.empty:
            frames.append(df)
    if not frames:
        print(
            f"  [combine] WARNING: all processed CSV files in {input_dir} are empty.\n"
            f"  [combine] Nothing to combine."
        )
        return pd.DataFrame()
    combined = pd.concat(frames, axis=0, ignore_index=True, sort=False)
    combined = apply_shared_random_init_baseline(combined)
    combined = order_columns(combined)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    print(f"  merged {len(paths)} file(s) → {len(combined)} rows, {len(combined.columns)} columns")
    print(f"  wrote {output_path}")
    return combined


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge per-project processed CSVs.")
    p.add_argument(
        "--model", default=None,
        help="Model backbone ('gin' or 'gpse_backbone'). When set, defaults to "
             "outputs/{model}/processed_projects/ → outputs/{model}/aggregated_results.csv.",
    )
    p.add_argument("--input-dir", type=Path, default=None,
                   help="Override input directory (overrides --model default).")
    p.add_argument("--output", type=Path, default=None,
                   help="Override output CSV path (overrides --model default).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.model is not None:
        input_dir   = args.input_dir or _OUTPUTS_BASE / args.model / "processed_projects"
        output_path = args.output    or _OUTPUTS_BASE / args.model / "aggregated_results.csv"
    else:
        input_dir   = args.input_dir   or DEFAULT_INPUT_DIR
        output_path = args.output      or DEFAULT_OUTPUT_PATH
    combine_processed_csvs(input_dir, output_path)


if __name__ == "__main__":
    main()
