#!/usr/bin/env python3
"""Process every project in WANDB_PROJECTS below, then merge CSVs.

** seed_subsample pipeline **
Only runs where ``ft_seed_subsample == True`` are kept; any others are
filtered out verbosely per-project.  Use ``scripts/plotting_legacy/run_all.py``
for the legacy fixed-subset pipeline.

Edit the list in this file (comment lines in/out) to choose which projects to run.

Usage
-----
    python scripts/plotting/run_all.py --entity <wandb-entity>
    python scripts/plotting/run_all.py --entity <wandb-entity> --skip-combine
    python scripts/plotting/run_all.py --entity <wandb-entity> --select-on-test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.plotting.combine_results import combine_processed_csvs
from scripts.plotting.process_project import process_project

DEFAULT_OUTPUT_DIR = _SCRIPT_DIR / "outputs" / "processed_projects"
DEFAULT_COMBINED = _SCRIPT_DIR / "outputs" / "aggregated_results.csv"

# ── W&B finetune projects (comment out any you want to skip) ─────────────────
WANDB_PROJECTS: list[str] = [
    # "finetune_gin_pretrain_sweep_graphmaev2_BBB_Martins_seedsub",
    # "finetune_gin_pretrain_sweep_graphmaev2_CYP3A4_Veith_seedsub",
    # "finetune_gin_pretrain_sweep_graphmaev2_IMDB-BINARY_seedsub",
    # "finetune_gin_pretrain_sweep_graphmaev2_PROTEINS_seedsub",
    # "finetune_gin_pretrain_sweep_graphmaev2_Clearance_Hepatocyte_AZ_seedsub",
    # "finetune_gin_pretrain_sweep_graphmaev2_ogbg-molhiv_seedsub",
    # "finetune_gin_pretrain_sweep_dgi_BBB_Martins_seedsub",
    # "finetune_gin_pretrain_sweep_dgi_CYP3A4_Veith_seedsub",
    # "finetune_gin_pretrain_sweep_dgi_IMDB-BINARY_seedsub",
    # "finetune_gin_pretrain_sweep_dgi_PROTEINS_seedsub",
    # "finetune_gin_pretrain_sweep_dgi_Clearance_Hepatocyte_AZ_seedsub",
    # "finetune_gin_pretrain_sweep_dgi_ogbg-molhiv_seedsub",
    # "finetune_gin_pretrain_sweep_vgae_BBB_Martins_seedsub",
    # "finetune_gin_pretrain_sweep_vgae_CYP3A4_Veith_seedsub",
    # "finetune_gin_pretrain_sweep_vgae_IMDB-BINARY_seedsub",
    # "finetune_gin_pretrain_sweep_vgae_PROTEINS_seedsub",
    # "finetune_gin_pretrain_sweep_vgae_Clearance_Hepatocyte_AZ_seedsub",
    # "finetune_gin_pretrain_sweep_vgae_ogbg-molhiv_seedsub",
    "finetune_gin_pretrain_sweep_graphcl_BBB_Martins_seedsub",
    "finetune_gin_pretrain_sweep_graphcl_CYP3A4_Veith_seedsub",
    "finetune_gin_pretrain_sweep_graphcl_IMDB-BINARY_seedsub",
    "finetune_gin_pretrain_sweep_graphcl_PROTEINS_seedsub",
    "finetune_gin_pretrain_sweep_graphcl_Clearance_Hepatocyte_AZ_seedsub",
    "finetune_gin_pretrain_sweep_graphcl_ogbg-molhiv_seedsub",
    "finetune_gin_pretrain_sweep_bgrl_BBB_Martins_seedsub",
    "finetune_gin_pretrain_sweep_bgrl_CYP3A4_Veith_seedsub",
    "finetune_gin_pretrain_sweep_bgrl_IMDB-BINARY_seedsub",
    "finetune_gin_pretrain_sweep_bgrl_PROTEINS_seedsub",
    "finetune_gin_pretrain_sweep_bgrl_Clearance_Hepatocyte_AZ_seedsub",
    "finetune_gin_pretrain_sweep_bgrl_ogbg-molhiv_seedsub",    
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Process all projects then combine CSVs.")
    p.add_argument("--entity", default="louis-van-langendonck-universitat-polit-cnica-de-catalunya", help="W&B entity.")
    p.add_argument("--train-seeds", nargs="+", type=int, default=[0, 1, 2, 3])
    p.add_argument("--state", default="finished", help="W&B state filter (empty = all).")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--combined-output", type=Path, default=DEFAULT_COMBINED)
    p.add_argument("--skip-combine", action="store_true", help="Only write per-project CSVs.")
    p.add_argument(
        "--select-on-test",
        action="store_true",
        help="Pick best hyperparameters by mean test metric (default: validation at best epoch).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    projects = list(WANDB_PROJECTS)
    state = args.state if args.state else None

    if not projects:
        print("No projects in WANDB_PROJECTS. Edit scripts/plotting/run_all.py.")
        return

    print(f"\n{'═'*60}")
    print(f"  Projects: {len(projects)}  (from run_all.py WANDB_PROJECTS)")
    print(f"  Entity  : {args.entity}")
    print(f"  Seeds   : {args.train_seeds}")
    selection = "test" if args.select_on_test else "validation (best_epoch/val/*)"
    print(f"  Select  : {selection}")
    print(f"{'═'*60}\n")

    for i, project in enumerate(projects, start=1):
        print(f"[{i}/{len(projects)}] {project}")
        process_project(
            args.entity,
            project,
            expected_seeds=args.train_seeds,
            state=state or "finished",
            output_dir=args.output_dir,
            select_on_test=args.select_on_test,
        )

    if not args.skip_combine:
        print("\nCombining per-project CSVs …")
        combine_processed_csvs(args.output_dir, args.combined_output)

    print(f"\n{'═'*60}\n")


if __name__ == "__main__":
    main()
