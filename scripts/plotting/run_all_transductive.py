#!/usr/bin/env python3
"""Process every transductive finetune project in WANDB_PROJECTS_T, then merge CSVs.

Transductive finetune projects follow the naming convention::

    finetune_gpse_backbone_pretrain_sweep_transductive_{method}_{dataset}_seedsub

Edit the list in this file (comment/uncomment lines) to control which projects are
fetched.  All outputs land in ``outputs_transductive/``.

Usage
-----
    python scripts/plotting/run_all_transductive.py --entity <wandb-entity>
    python scripts/plotting/run_all_transductive.py --entity <wandb-entity> --skip-combine
    python scripts/plotting/run_all_transductive.py --entity <wandb-entity> --select-on-test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.plotting.combine_results import combine_processed_csvs
from scripts.plotting.process_project_transductive import (
    OUTPUTS_BASE_T,
    DEFAULT_OUTPUT_DIR_T,
    process_project_transductive,
)

# ── Transductive W&B finetune projects ────────────────────────────────────────
# Convention: finetune_gpse_backbone_pretrain_sweep_transductive_{method}_{dataset}_seedsub
# Comment/uncomment individual lines to control which projects are fetched.
WANDB_PROJECTS_T: list[str] = [
    # ── GraphMAEv2 ──────────────────────────────────────────────────────────
    # "finetune_gpse_backbone_pretrain_sweep_transductive_graphmaev2_cocitation_cora_seedsub",
    # "finetune_gpse_backbone_pretrain_sweep_transductive_graphmaev2_cocitation_pubmed_seedsub",
    # "finetune_gpse_backbone_pretrain_sweep_transductive_graphmaev2_minesweeper_seedsub",
    # "finetune_gpse_backbone_pretrain_sweep_transductive_graphmaev2_roman_empire_seedsub",
    # ── DGI ─────────────────────────────────────────────────────────────────
    # "finetune_gpse_backbone_pretrain_sweep_transductive_dgi_cocitation_cora_seedsub",
    # "finetune_gpse_backbone_pretrain_sweep_transductive_dgi_cocitation_pubmed_seedsub",
    # "finetune_gpse_backbone_pretrain_sweep_transductive_dgi_minesweeper_seedsub",
    # "finetune_gpse_backbone_pretrain_sweep_transductive_dgi_roman_empire_seedsub",
    # ── BGRL ────────────────────────────────────────────────────────────────
    # "finetune_gpse_backbone_pretrain_sweep_transductive_bgrl_cocitation_cora_seedsub",
    # "finetune_gpse_backbone_pretrain_sweep_transductive_bgrl_cocitation_pubmed_seedsub",
    # "finetune_gpse_backbone_pretrain_sweep_transductive_bgrl_minesweeper_seedsub",
    # "finetune_gpse_backbone_pretrain_sweep_transductive_bgrl_roman_empire_seedsub",
    # # ── VGAE ────────────────────────────────────────────────────────────────
    # "finetune_gpse_backbone_pretrain_sweep_transductive_vgae_cocitation_cora_seedsub",
    # "finetune_gpse_backbone_pretrain_sweep_transductive_vgae_cocitation_pubmed_seedsub",
    # "finetune_gpse_backbone_pretrain_sweep_transductive_vgae_minesweeper_seedsub",
    "finetune_gpse_backbone_pretrain_sweep_transductive_vgae_roman_empire_seedsub",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Process all transductive finetune projects then combine CSVs.",
    )
    p.add_argument(
        "--entity",
        default="louis-van-langendonck-universitat-polit-cnica-de-catalunya",
        help="W&B entity.",
    )
    p.add_argument("--train-seeds", nargs="+", type=int, default=[0, 1, 2, 3])
    p.add_argument("--state", default="finished", help="W&B run state filter.")
    p.add_argument(
        "--output-dir", type=Path, default=None,
        help="Override processed-projects dir "
             "(default: outputs_transductive/processed_projects).",
    )
    p.add_argument(
        "--combined-output", type=Path, default=None,
        help="Override combined CSV path "
             "(default: outputs_transductive/aggregated_results.csv).",
    )
    p.add_argument("--skip-combine", action="store_true",
                   help="Only write per-project CSVs, skip the merge step.")
    p.add_argument("--select-on-test", action="store_true",
                   help="Rank hyperparameters by test metric (default: validation).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir      = args.output_dir      or DEFAULT_OUTPUT_DIR_T
    combined_output = args.combined_output or OUTPUTS_BASE_T / "aggregated_results.csv"
    state = args.state if args.state else "finished"

    if not WANDB_PROJECTS_T:
        print("No transductive projects enabled in WANDB_PROJECTS_T.")
        print("Edit scripts/plotting/run_all_transductive.py and uncomment the relevant lines.")
        return

    selection = "test" if args.select_on_test else "validation (best_epoch/val/*)"

    print(f"\n{'═'*60}")
    print(f"  Mode    : transductive (gpse_backbone)")
    print(f"  Projects: {len(WANDB_PROJECTS_T)}")
    print(f"  Entity  : {args.entity}")
    print(f"  Seeds   : {args.train_seeds}")
    print(f"  Select  : {selection}")
    print(f"  Out dir : {output_dir}")
    print(f"{'═'*60}\n")

    for i, project in enumerate(WANDB_PROJECTS_T, start=1):
        print(f"[{i}/{len(WANDB_PROJECTS_T)}] {project}")
        process_project_transductive(
            args.entity,
            project,
            expected_seeds=args.train_seeds,
            state=state,
            output_dir=output_dir,
            select_on_test=args.select_on_test,
        )

    if not args.skip_combine:
        print("\nCombining per-project CSVs …")
        combine_processed_csvs(output_dir, combined_output)

    print(f"\n{'═'*60}\n")


if __name__ == "__main__":
    main()
