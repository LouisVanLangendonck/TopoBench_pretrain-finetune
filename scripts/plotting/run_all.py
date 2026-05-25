#!/usr/bin/env python3
"""Process every project in wandb_projects_list.json, then merge CSVs.

Usage
-----
    python scripts/plotting/run_all.py --entity <wandb-entity>
    python scripts/plotting/run_all.py --entity <wandb-entity> --skip-combine
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.plotting.combine_results import combine_processed_csvs
from scripts.plotting.process_project import process_project

DEFAULT_PROJECTS_JSON = _SCRIPT_DIR / "wandb_projects_list.json"
DEFAULT_OUTPUT_DIR = _SCRIPT_DIR / "outputs" / "processed_projects"
DEFAULT_COMBINED = _SCRIPT_DIR / "outputs" / "aggregated_results.csv"


def load_projects(path: Path) -> list[str]:
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} must be a JSON array of project name strings.")
    return [str(p) for p in data]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Process all projects then combine CSVs.")
    p.add_argument("--entity", required=True, help="W&B entity.")
    p.add_argument(
        "--projects-json", type=Path, default=DEFAULT_PROJECTS_JSON,
        help="JSON list of W&B project names.",
    )
    p.add_argument("--train-seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--state", default="finished", help="W&B state filter (empty = all).")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--combined-output", type=Path, default=DEFAULT_COMBINED)
    p.add_argument("--skip-combine", action="store_true", help="Only write per-project CSVs.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    projects = load_projects(args.projects_json)
    state = args.state if args.state else None

    print(f"\n{'═'*60}")
    print(f"  Projects: {len(projects)}  (from {args.projects_json.name})")
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
