#!/usr/bin/env python3
"""Export aggregated fine-tuning W&B results to CSV.

Pipeline
--------
1. Fetch all runs from finetune W&B projects.
2. Group by every finetuning + pretraining hyperparameter except ``ft_train_seed``.
3. Keep groups with exactly ``len(train_seeds)`` distinct seeds; flag the rest.
4. Average test metrics (mean / std) across seeds.
5. Keep only hyperparameters that actually vary in the aggregated table.
6. For each (dataset, pretraining method, GIN architecture, ft_mode, ft_fraction),
   select the hyperparameter setting with the best mean test monitor metric.

Usage
-----
    python scripts/plotting/process_to_csv.py \\
        [--entity ...] \\
        [--projects finetune_gin_pretrain_sweep_dgi_MUTAG ...] \\
        [--config scripts/finetuning/sweep_config.yaml] \\
        [--output-dir results/plotting] \\
        [--dry-run]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.plotting.aggregate import process_finetune_projects
from scripts.plotting.defaults import load_finetune_sweep_defaults


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Aggregate W&B fine-tuning runs into analysis CSVs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--entity", default=None,
        help="W&B entity (default: from finetuning sweep_config.yaml).",
    )
    p.add_argument(
        "--projects", nargs="+", default=None,
        help="Finetune W&B project names (default: finetune_<each pretrain project>).",
    )
    p.add_argument(
        "--pretrain-projects", nargs="+", default=None, dest="pretrain_projects",
        help="Pretrain project names; finetune names are derived as finetune_<name>.",
    )
    p.add_argument(
        "--config", default=None,
        help="Path to finetuning sweep_config.yaml for defaults.",
    )
    p.add_argument(
        "--train-seeds", nargs="+", type=int, default=None, dest="train_seeds",
        help="Expected ft_train_seed values per hyperparameter group.",
    )
    p.add_argument(
        "--output-dir", default=str(_PROJECT_ROOT / "results" / "plotting"),
        help="Directory for output CSV files.",
    )
    p.add_argument(
        "--state", default="finished",
        help="W&B run state filter (default: finished). Use empty string for all.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print project list and exit without calling W&B.",
    )
    return p.parse_args()


def resolve_projects(args: argparse.Namespace, defaults: dict) -> list[str]:
    if args.projects:
        return list(args.projects)
    pretrain = args.pretrain_projects or defaults["pretrain_projects"]
    return [f"finetune_{p}" for p in pretrain]


def main() -> None:
    args = parse_args()
    defaults = load_finetune_sweep_defaults(args.config)

    entity = args.entity or defaults["entity"]
    projects = resolve_projects(args, defaults)
    seeds = args.train_seeds if args.train_seeds is not None else defaults["train_seeds"]
    state = args.state if args.state else None

    print(f"\n{'═'*65}")
    print("  W&B → CSV aggregation")
    print(f"  entity  : {entity}")
    print(f"  projects: {len(projects)}")
    print(f"  seeds   : {seeds} (expected per hyperparameter group)")
    print(f"{'═'*65}\n")

    if args.dry_run:
        for p in projects:
            print(f"  {entity}/{p}")
        print("\n  [DRY RUN] No W&B fetch.")
        return

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    final_df, agg_df, flagged_df = process_finetune_projects(
        entity, projects, expected_seeds=seeds, state=state or "finished",
    )

    final_path = out_dir / "finetune_best_hyperparams.csv"
    agg_path = out_dir / "finetune_seed_aggregated.csv"
    flag_path = out_dir / "finetune_flagged_groups.csv"
    raw_path = out_dir / "finetune_summary.txt"

    final_df.to_csv(final_path, index=False)
    agg_df.to_csv(agg_path, index=False)
    flagged_df.to_csv(flag_path, index=False)

    with open(raw_path, "w") as f:
        f.write(f"entity: {entity}\n")
        f.write(f"projects: {len(projects)}\n")
        f.write(f"expected_seeds: {seeds}\n")
        f.write(f"aggregated_groups: {len(agg_df)}\n")
        f.write(f"flagged_groups: {len(flagged_df)}\n")
        f.write(f"final_rows: {len(final_df)}\n")

    print(f"\n  Aggregated groups (all seeds OK): {len(agg_df)}")
    print(f"  Flagged groups (seed mismatch)  : {len(flagged_df)}")
    print(f"  Final selected rows             : {len(final_df)}")
    print(f"\n  Wrote:")
    print(f"    {final_path}")
    print(f"    {agg_path}")
    print(f"    {flag_path}")
    print(f"    {raw_path}")
    print(f"\n{'═'*65}\n")


if __name__ == "__main__":
    main()
