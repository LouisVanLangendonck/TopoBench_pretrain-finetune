#!/usr/bin/env python3
"""Fetch all W&B runs for one finetune project (no seed aggregation or selection).

** seed_subsample pipeline **
Only runs where ``ft_seed_subsample == True`` are kept; others are dropped
verbosely.  Use ``scripts/plotting_legacy/hyperparam_analysis/fetch_project.py``
for the legacy fixed-subset pipeline.

Writes ``outputs/processed_projects/<project>.csv`` with:
  - identity columns (dataset, model, pretraining task, ft_mode, ft_fraction)
  - ``hyperparam_*`` columns derived from ``pretrained_config_model.*`` (+ ``ft_pooling``)
  - all metric columns (``test/*``, ``best_*``, …)

Usage
-----
    python scripts/plotting/hyperparam_analysis/fetch_project.py \\
        --entity <wandb-entity> \\
        --project finetune_gin_pretrain_sweep_dgi_PROTEINS
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.plotting.hyperparam_analysis.columns import (
    sanitize_filename,
    transform_raw_table,
)
from scripts.plotting.process_project import (
    build_raw_table,
    extract_finetune_dataset_name,
    fetch_runs,
    filter_seed_subsample,
    load_dataset_monitor_info,
)

DEFAULT_OUTPUT_DIR = _SCRIPT_DIR / "outputs" / "processed_projects"


def _attach_monitor_info(
    df: pd.DataFrame,
    finetune_data_name: str | None,
    dataset_monitor_info: dict[str, tuple[str, str]],
) -> pd.DataFrame:
    if df.empty:
        return df
    task, metric, mode = None, None, "max"
    if finetune_data_name and finetune_data_name in dataset_monitor_info:
        task, metric = dataset_monitor_info[finetune_data_name]
        from scripts.plotting.process_project import _derive_mode
        mode = _derive_mode(task, metric)
    out = df.copy()
    out["monitor_task"] = task
    out["monitor_metric"] = metric
    out["monitor_mode"] = mode
    out["monitor_test_column"] = f"test/{metric}" if metric else None
    return out


def fetch_project_raw(
    entity: str,
    project: str,
    *,
    state: str = "finished",
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """Download runs, transform columns, write CSV. Returns the table."""
    dmi = load_dataset_monitor_info()
    ft_data_name = extract_finetune_dataset_name(project)
    runs = fetch_runs(entity, project, state=state)
    raw = build_raw_table(runs, project)
    if raw.empty:
        print("  no runs — skipping write")
        return raw

    raw = filter_seed_subsample(raw, project)
    if raw.empty:
        print("  no seed_subsample=True runs — skipping write")
        return raw

    table = transform_raw_table(raw)
    table = _attach_monitor_info(table, ft_data_name, dmi)

    out_dir = output_dir or DEFAULT_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{sanitize_filename(project)}.csv"
    table.to_csv(out_path, index=False)
    print(f"  wrote {out_path}  ({len(table)} runs, {len(table.columns)} columns)")
    return table


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch raw W&B runs for hyperparam analysis.")
    p.add_argument("--entity", required=True, help="W&B entity.")
    p.add_argument("--project", required=True, help="W&B finetune project name.")
    p.add_argument("--state", default="finished", help="W&B state filter (empty = all).")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    state = args.state if args.state else None
    fetch_project_raw(
        args.entity,
        args.project,
        state=state or "finished",
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
