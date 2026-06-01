#!/usr/bin/env python3
"""Fetch raw runs for all projects in ``config.yaml``, combine, and plot.

** seed_subsample pipeline **
Only runs where ``ft_seed_subsample == True`` are fetched and kept; others
are dropped verbosely.  Use ``scripts/plotting_legacy/hyperparam_analysis/``
for the legacy fixed-subset pipeline.

Usage
-----
    python scripts/plotting/hyperparam_analysis/run_all.py
    python scripts/plotting/hyperparam_analysis/run_all.py --config path/to/config.yaml
    python scripts/plotting/hyperparam_analysis/run_all.py --skip-fetch --skip-plot
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.plotting.hyperparam_analysis.combine_results import combine_raw_csvs
from scripts.plotting.hyperparam_analysis.fetch_project import fetch_project_raw
from scripts.plotting.hyperparam_analysis.plot_hyperparams import plot_all

DEFAULT_CONFIG = _SCRIPT_DIR / "config.yaml"


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hyperparam analysis pipeline (fetch → combine → plot).")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--skip-fetch", action="store_true", help="Only combine + plot existing CSVs.")
    p.add_argument("--skip-combine", action="store_true", help="Skip merge step.")
    p.add_argument("--skip-plot", action="store_true", help="Skip figure generation.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    entity = cfg.get("entity", "")
    state = cfg.get("state", "finished") or None
    projects: list[str] = list(cfg.get("projects") or [])
    out_cfg = cfg.get("output") or {}

    processed_dir = _SCRIPT_DIR / out_cfg.get("processed_dir", "outputs/processed_projects")
    combined_csv = _SCRIPT_DIR / out_cfg.get("combined_csv", "outputs/all_runs.csv")
    figures_dir = _SCRIPT_DIR / out_cfg.get("figures_dir", "outputs/figures")
    fmt = cfg.get("figure_format", "png")

    if not args.skip_fetch:
        if not entity:
            raise ValueError("config.yaml must set 'entity' when fetching from W&B")
        if not projects:
            print("No projects in config.yaml — nothing to fetch.")
        print(f"\n{'═'*60}")
        print(f"  Fetch raw runs  ({len(projects)} project(s))")
        print(f"  Entity: {entity}")
        print(f"{'═'*60}\n")
        for i, project in enumerate(projects, start=1):
            print(f"[{i}/{len(projects)}] {project}")
            fetch_project_raw(entity, project, state=state or "finished", output_dir=processed_dir)

    if not args.skip_combine:
        print("\nCombining per-project CSVs …")
        combine_raw_csvs(processed_dir, combined_csv)

    if not args.skip_plot:
        if not combined_csv.is_file():
            raise FileNotFoundError(f"Combined CSV not found: {combined_csv}")
        import pandas as pd
        df = pd.read_csv(combined_csv)
        print(f"\nPlotting hyperparameter grids ({len(df)} runs) …")
        plot_all(df, figures_dir, fmt=fmt)

    print(f"\n{'═'*60}\n")


if __name__ == "__main__":
    main()
