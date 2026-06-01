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

_SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = _SCRIPT_DIR / "outputs" / "processed_projects"
DEFAULT_OUTPUT_PATH = _SCRIPT_DIR / "outputs" / "aggregated_results.csv"

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
    p.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    combine_processed_csvs(args.input_dir, args.output)


if __name__ == "__main__":
    main()
