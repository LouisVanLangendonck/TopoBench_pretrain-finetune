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

OUTPUTS_BASE = _SCRIPT_DIR / "outputs"
MODELS = ("gin", "gpse_backbone")

# ── W&B finetune projects per model backbone ──────────────────────────────────
# Comment/uncomment individual lines to control which projects are fetched.
# Project naming convention:
#   finetune_{model}_pretrain_sweep_{method}_{dataset}_seedsub
WANDB_PROJECTS: dict[str, list[str]] = {
    # ── GIN backbone ─────────────────────────────────────────────────────────
    "gin": [
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
        # "finetune_gin_pretrain_sweep_graphcl_BBB_Martins_seedsub",
        # "finetune_gin_pretrain_sweep_graphcl_CYP3A4_Veith_seedsub",
        # "finetune_gin_pretrain_sweep_graphcl_IMDB-BINARY_seedsub",
        # "finetune_gin_pretrain_sweep_graphcl_PROTEINS_seedsub",
        # "finetune_gin_pretrain_sweep_graphcl_Clearance_Hepatocyte_AZ_seedsub",
        # "finetune_gin_pretrain_sweep_graphcl_ogbg-molhiv_seedsub",
        # "finetune_gin_pretrain_sweep_bgrl_BBB_Martins_seedsub",
        # "finetune_gin_pretrain_sweep_bgrl_CYP3A4_Veith_seedsub",
        # "finetune_gin_pretrain_sweep_bgrl_IMDB-BINARY_seedsub",
        # "finetune_gin_pretrain_sweep_bgrl_PROTEINS_seedsub",
        # "finetune_gin_pretrain_sweep_bgrl_Clearance_Hepatocyte_AZ_seedsub",
        # "finetune_gin_pretrain_sweep_bgrl_ogbg-molhiv_seedsub",
        # "finetune_gin_pretrain_sweep_dgi_REDDIT-BINARY_seedsub",
        # "finetune_gin_pretrain_sweep_graphcl_REDDIT-BINARY_seedsub",
        # "finetune_gin_pretrain_sweep_bgrl_REDDIT-BINARY_seedsub",
        # "finetune_gin_pretrain_sweep_vgae_REDDIT-BINARY_seedsub",
        "finetune_gin_pretrain_sweep_graphmaev2_REDDIT-BINARY_seedsub",
    ],
    # ── GPSE backbone ─────────────────────────────────────────────────────────
    "gpse_backbone": [
        # "finetune_gpse_backbone_pretrain_sweep_graphmaev2_BBB_Martins_seedsub",
        #"finetune_gpse_backbone_pretrain_sweep_graphmaev2_CYP3A4_Veith_seedsub",
        "finetune_gpse_backbone_pretrain_sweep_graphmaev2_IMDB-BINARY_seedsub",
        # #"finetune_gpse_backbone_pretrain_sweep_graphmaev2_PROTEINS_seedsub",
        # "finetune_gpse_backbone_pretrain_sweep_graphmaev2_Clearance_Hepatocyte_AZ_seedsub",
        #"finetune_gpse_backbone_pretrain_sweep_graphmaev2_ogbg-molhiv_seedsub",
        # "finetune_gpse_backbone_pretrain_sweep_dgi_BBB_Martins_seedsub",
        # "finetune_gpse_backbone_pretrain_sweep_dgi_CYP3A4_Veith_seedsub",
        # "finetune_gpse_backbone_pretrain_sweep_dgi_IMDB-BINARY_seedsub",
        # #"finetune_gpse_backbone_pretrain_sweep_dgi_PROTEINS_seedsub",
        # "finetune_gpse_backbone_pretrain_sweep_dgi_Clearance_Hepatocyte_AZ_seedsub",
        #"finetune_gpse_backbone_pretrain_sweep_dgi_ogbg-molhiv_seedsub",
        # "finetune_gpse_backbone_pretrain_sweep_vgae_BBB_Martins_seedsub",
        # "finetune_gpse_backbone_pretrain_sweep_vgae_CYP3A4_Veith_seedsub",
        # "finetune_gpse_backbone_pretrain_sweep_vgae_IMDB-BINARY_seedsub",
        # # "finetune_gpse_backbone_pretrain_sweep_vgae_PROTEINS_seedsub",
        # "finetune_gpse_backbone_pretrain_sweep_vgae_Clearance_Hepatocyte_AZ_seedsub",
        #"finetune_gpse_backbone_pretrain_sweep_vgae_ogbg-molhiv_seedsub",
        # "finetune_gpse_backbone_pretrain_sweep_graphcl_BBB_Martins_seedsub",
        # "finetune_gpse_backbone_pretrain_sweep_graphcl_CYP3A4_Veith_seedsub",
        # "finetune_gpse_backbone_pretrain_sweep_graphcl_IMDB-BINARY_seedsub",
        # # "finetune_gpse_backbone_pretrain_sweep_graphcl_PROTEINS_seedsub",
        # "finetune_gpse_backbone_pretrain_sweep_graphcl_Clearance_Hepatocyte_AZ_seedsub",
        # "finetune_gpse_backbone_pretrain_sweep_graphcl_ogbg-molhiv_seedsub",
        # "finetune_gpse_backbone_pretrain_sweep_bgrl_BBB_Martins_seedsub",
        # "finetune_gpse_backbone_pretrain_sweep_bgrl_CYP3A4_Veith_seedsub",
        # "finetune_gpse_backbone_pretrain_sweep_bgrl_IMDB-BINARY_seedsub",
        # # "finetune_gpse_backbone_pretrain_sweep_bgrl_PROTEINS_seedsub",
        # "finetune_gpse_backbone_pretrain_sweep_bgrl_Clearance_Hepatocyte_AZ_seedsub",
        # "finetune_gpse_backbone_pretrain_sweep_bgrl_ogbg-molhiv_seedsub",
        # "finetune_gpse_backbone_pretrain_sweep_dgi_REDDIT-BINARY_seedsub",
        # "finetune_gpse_backbone_pretrain_sweep_graphcl_REDDIT-BINARY_seedsub",
        # "finetune_gpse_backbone_pretrain_sweep_bgrl_REDDIT-BINARY_seedsub",
        # "finetune_gpse_backbone_pretrain_sweep_vgae_REDDIT-BINARY_seedsub",
        "finetune_gpse_backbone_pretrain_sweep_graphmaev2_REDDIT-BINARY_seedsub",
    ],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Process all projects then combine CSVs.")
    p.add_argument(
        "--model", default="gin", choices=list(MODELS),
        help="Model backbone to process (determines output subdirectory and project list).",
    )
    p.add_argument("--entity", default="louis-van-langendonck-universitat-polit-cnica-de-catalunya", help="W&B entity.")
    p.add_argument("--train-seeds", nargs="+", type=int, default=[0, 1, 2, 3])
    p.add_argument("--state", default="finished", help="W&B state filter (empty = all).")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Override processed-projects dir (default: outputs/{model}/processed_projects).")
    p.add_argument("--combined-output", type=Path, default=None,
                   help="Override combined CSV path (default: outputs/{model}/aggregated_results.csv).")
    p.add_argument("--skip-combine", action="store_true", help="Only write per-project CSVs.")
    p.add_argument(
        "--select-on-test",
        action="store_true",
        help="Pick best hyperparameters by mean test metric (default: validation at best epoch).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Model-specific output paths
    model_dir      = OUTPUTS_BASE / args.model
    output_dir     = args.output_dir     or model_dir / "processed_projects"
    combined_output = args.combined_output or model_dir / "aggregated_results.csv"

    projects = list(WANDB_PROJECTS.get(args.model, []))
    state = args.state if args.state else None

    if not projects:
        print(f"No projects enabled for model '{args.model}' in WANDB_PROJECTS.")
        print("Edit scripts/plotting/run_all.py and uncomment the relevant lines.")
        return

    print(f"\n{'═'*60}")
    print(f"  Model   : {args.model}")
    print(f"  Projects: {len(projects)}  (from run_all.py WANDB_PROJECTS['{args.model}'])")
    print(f"  Entity  : {args.entity}")
    print(f"  Seeds   : {args.train_seeds}")
    selection = "test" if args.select_on_test else "validation (best_epoch/val/*)"
    print(f"  Select  : {selection}")
    print(f"  Out dir : {output_dir}")
    print(f"{'═'*60}\n")

    for i, project in enumerate(projects, start=1):
        print(f"[{i}/{len(projects)}] {project}")
        process_project(
            args.entity,
            project,
            expected_seeds=args.train_seeds,
            state=state or "finished",
            output_dir=output_dir,
            select_on_test=args.select_on_test,
        )

    if not args.skip_combine:
        print("\nCombining per-project CSVs …")
        combine_processed_csvs(output_dir, combined_output)

    print(f"\n{'═'*60}\n")


if __name__ == "__main__":
    main()
