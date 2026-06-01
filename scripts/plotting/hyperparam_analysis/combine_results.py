#!/usr/bin/env python3
"""Merge per-project raw hyperparam CSVs into one table.

Usage
-----
    python scripts/plotting/hyperparam_analysis/combine_results.py
    python scripts/plotting/hyperparam_analysis/combine_results.py --input-dir path/to/csvs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.plotting.hyperparam_analysis.columns import order_hyperparam_columns

DEFAULT_INPUT_DIR = _SCRIPT_DIR / "outputs" / "processed_projects"
DEFAULT_OUTPUT_PATH = _SCRIPT_DIR / "outputs" / "all_runs.csv"


def combine_raw_csvs(
    input_dir: Path,
    output_path: Path,
) -> pd.DataFrame:
    paths = sorted(p for p in input_dir.glob("*.csv") if p.is_file())
    if not paths:
        raise FileNotFoundError(f"No CSV files in {input_dir}")

    frames: list[pd.DataFrame] = []
    for p in paths:
        try:
            df = pd.read_csv(p)
        except pd.errors.EmptyDataError:
            continue
        if not df.empty:
            frames.append(df)
    if not frames:
        raise FileNotFoundError(f"All CSV files in {input_dir} are empty.")

    combined = pd.concat(frames, axis=0, ignore_index=True, sort=False)
    combined = order_hyperparam_columns(combined)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    print(f"  merged {len(paths)} file(s) → {len(combined)} rows, {len(combined.columns)} columns")
    print(f"  wrote {output_path}")
    return combined


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge per-project raw hyperparam CSVs.")
    p.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    combine_raw_csvs(args.input_dir, args.output)


if __name__ == "__main__":
    main()
