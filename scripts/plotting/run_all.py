#!/usr/bin/env python3
"""Process every project in WANDB_PROJECTS below, then merge CSVs.

Edit the list in this file (comment lines in/out) to choose which projects to run.

Usage
-----
    python scripts/plotting/run_all.py --entity <wandb-entity>
    python scripts/plotting/run_all.py --entity <wandb-entity> --skip-combine
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
    # "finetune_gin_pretrain_sweep_graphmaev2_IMDB-BINARY",
    # "finetune_gin_pretrain_sweep_graphmaev2_MUTAG",
    # "finetune_gin_pretrain_sweep_graphmaev2_PROTEINS",
    # "finetune_gin_pretrain_sweep_graphmaev2_Caco2_Wang",
    # "finetune_gin_pretrain_sweep_graphmaev2_Clearance_Hepatocyte_AZ",
    # "finetune_gin_pretrain_sweep_graphmaev2_ogbg-molbace",
    # "finetune_gin_pretrain_sweep_dgi_IMDB-BINARY",
    # "finetune_gin_pretrain_sweep_dgi_MUTAG",
    # "finetune_gin_pretrain_sweep_dgi_PROTEINS",
    # "finetune_gin_pretrain_sweep_dgi_Caco2_Wang",
    # "finetune_gin_pretrain_sweep_dgi_Clearance_Hepatocyte_AZ",
    # "finetune_gin_pretrain_sweep_dgi_ogbg-molbace",
    # "finetune_gin_pretrain_sweep_graphcl_IMDB-BINARY",
    # "finetune_gin_pretrain_sweep_graphcl_MUTAG",
    # "finetune_gin_pretrain_sweep_graphcl_PROTEINS",
    # "finetune_gin_pretrain_sweep_graphcl_Caco2_Wang",
    # "finetune_gin_pretrain_sweep_graphcl_Clearance_Hepatocyte_AZ",
    # "finetune_gin_pretrain_sweep_graphcl_ogbg-molbace",
    # "finetune_gin_pretrain_sweep_bgrl_IMDB-BINARY",
    # "finetune_gin_pretrain_sweep_bgrl_MUTAG",
    # "finetune_gin_pretrain_sweep_bgrl_PROTEINS",
    # "finetune_gin_pretrain_sweep_bgrl_Caco2_Wang",
    # "finetune_gin_pretrain_sweep_bgrl_Clearance_Hepatocyte_AZ",
    # "finetune_gin_pretrain_sweep_bgrl_ogbg-molbace",
    # "finetune_gin_pretrain_sweep_vgae_IMDB-BINARY",
    # "finetune_gin_pretrain_sweep_vgae_MUTAG",
    # "finetune_gin_pretrain_sweep_vgae_PROTEINS",
    # "finetune_gin_pretrain_sweep_vgae_Caco2_Wang",
    # "finetune_gin_pretrain_sweep_vgae_Clearance_Hepatocyte_AZ",
    # "finetune_gin_pretrain_sweep_vgae_ogbg-molbace",
    "finetune_gin_pretrain_sweep_bgrl_BBB_Martins",
    "finetune_gin_pretrain_sweep_bgrl_CYP3A4_Veith",
    "finetune_gin_pretrain_sweep_bgrl_ogbg-molhiv",
    "finetune_gin_pretrain_sweep_graphmaev2_BBB_Martins",
    "finetune_gin_pretrain_sweep_graphmaev2_CYP3A4_Veith",
    "finetune_gin_pretrain_sweep_graphmaev2_ogbg-molhiv",
    "finetune_gin_pretrain_sweep_dgi_BBB_Martins",
    "finetune_gin_pretrain_sweep_dgi_CYP3A4_Veith",
    "finetune_gin_pretrain_sweep_dgi_ogbg-molhiv",
    "finetune_gin_pretrain_sweep_graphcl_BBB_Martins",
    "finetune_gin_pretrain_sweep_graphcl_CYP3A4_Veith",
    "finetune_gin_pretrain_sweep_graphcl_ogbg-molhiv",
    "finetune_gin_pretrain_sweep_vgae_BBB_Martins",
    "finetune_gin_pretrain_sweep_vgae_CYP3A4_Veith",
    "finetune_gin_pretrain_sweep_vgae_ogbg-molhiv",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Process all projects then combine CSVs.")
    p.add_argument("--entity", default="louis-van-langendonck-universitat-polit-cnica-de-catalunya", help="W&B entity.")
    p.add_argument("--train-seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--state", default="finished", help="W&B state filter (empty = all).")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--combined-output", type=Path, default=DEFAULT_COMBINED)
    p.add_argument("--skip-combine", action="store_true", help="Only write per-project CSVs.")
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
    print(f"{'═'*60}\n")

    for i, project in enumerate(projects, start=1):
        print(f"[{i}/{len(projects)}] {project}")
        process_project(
            args.entity,
            project,
            expected_seeds=args.train_seeds,
            state=state or "finished",
            output_dir=args.output_dir,
        )

    if not args.skip_combine:
        print("\nCombining per-project CSVs …")
        combine_processed_csvs(args.output_dir, args.combined_output)

    print(f"\n{'═'*60}\n")


if __name__ == "__main__":
    main()
