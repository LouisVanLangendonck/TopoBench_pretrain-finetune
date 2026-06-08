#!/usr/bin/env python3
"""Aggregate + select best runs for a single transductive W&B finetune project.

Identical pipeline to ``process_project.py`` except:

  1. **Project naming**.  Transductive projects embed an extra ``transductive_``
     segment between the pretrain-sweep marker and the method name::

         finetune_{model}_pretrain_sweep_transductive_{method}_{dataset}[_seedsub]

     The inductive variant does not have this extra segment.

  2. **Dataset config lookup**.  The dataset config ``data_name`` field (e.g.
     ``Cora``) may differ from the YAML file stem embedded in the project name
     (e.g. ``cocitation_cora``).  This module indexes the monitor-info dict by
     *both* keys so the lookup succeeds regardless.

  3. **Default output directory**: ``outputs_transductive/`` instead of
     ``outputs/``.

Usage
-----
    python scripts/plotting/process_project_transductive.py \\
        --entity <wandb-entity> \\
        --project finetune_gpse_backbone_pretrain_sweep_transductive_graphmaev2_cocitation_cora_seedsub
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.plotting.process_project import (
    _SEEDSUB_SUFFIX,
    _PRETRAIN_SWEEP_MARKER,
    fetch_runs,
    build_raw_table,
    filter_seed_subsample,
    aggregate_seeds,
    select_best_rows,
    sanitize_filename,
)
from scripts.plotting.wb_table import drop_constant_columns

# ── Output paths ──────────────────────────────────────────────────────────────
OUTPUTS_BASE_T = _SCRIPT_DIR / "outputs_transductive"
DEFAULT_OUTPUT_DIR_T = OUTPUTS_BASE_T / "processed_projects"

# The extra path segment that distinguishes transductive from inductive projects.
_TRANSDUCTIVE_SEGMENT = "transductive_"


# ── Transductive-specific helpers ─────────────────────────────────────────────

def extract_transductive_finetune_dataset_name(project: str) -> str | None:
    """Parse the finetuning dataset name from a transductive project name.

    Convention::

        finetune_{model}_pretrain_sweep_transductive_{method}_{dataset}[_seedsub]

    Returns the YAML config file stem (e.g. ``"cocitation_cora"``,
    ``"minesweeper"``), which is also the key used in the transductive DMI.

    Falls back to ``None`` if the expected markers are absent.
    """
    name = project
    if name.endswith(_SEEDSUB_SUFFIX):
        name = name[: -len(_SEEDSUB_SUFFIX)]

    idx = name.find(_PRETRAIN_SWEEP_MARKER)
    if idx == -1:
        return None

    # after_marker: e.g. "transductive_graphmaev2_cocitation_cora"
    after_marker = name[idx + len(_PRETRAIN_SWEEP_MARKER):]
    if after_marker.startswith(_TRANSDUCTIVE_SEGMENT):
        after_marker = after_marker[len(_TRANSDUCTIVE_SEGMENT):]
        # after_marker: e.g. "graphmaev2_cocitation_cora"

    # First token = method name (single word, no underscores); rest = dataset stem.
    sep = after_marker.find("_")
    if sep == -1:
        return None
    return after_marker[sep + 1:]  # e.g. "cocitation_cora"


def load_transductive_dataset_monitor_info(
    config_root: Path | None = None,
) -> dict[str, tuple[str, str]]:
    """Return {key: (task, monitor_metric)} for all graph dataset configs.

    Indexes each config by **both** its ``data_name`` (e.g. ``Cora``) and its
    YAML file stem (e.g. ``cocitation_cora``) so that transductive project names
    (which use the stem) find the right entry.

    Identical to ``load_dataset_monitor_info`` in ``process_project.py`` except
    for the extra stem-based indexing.
    """
    root = config_root or _PROJECT_ROOT
    result: dict[str, tuple[str, str]] = {}
    graph_cfg_dir = root / "configs" / "dataset" / "graph"
    if not graph_cfg_dir.is_dir():
        return result
    for p in graph_cfg_dir.glob("*.yaml"):
        try:
            cfg = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        except Exception:
            continue
        params = cfg.get("parameters", {}) or {}
        loader_params = (cfg.get("loader", {}) or {}).get("parameters", {}) or {}
        data_name = loader_params.get("data_name")
        task = params.get("task")
        monitor_metric = params.get("monitor_metric")
        if not (data_name and task and monitor_metric):
            continue
        info: tuple[str, str] = (str(task), str(monitor_metric))
        result[str(data_name)] = info
        yaml_stem = p.stem
        if yaml_stem != str(data_name):
            result[yaml_stem] = info
    return result


# ── Main pipeline ─────────────────────────────────────────────────────────────

def process_project_transductive(
    entity: str,
    project: str,
    *,
    expected_seeds: list[int] | None = None,
    state: str = "finished",
    output_dir: Path | None = None,
    select_on_test: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full per-project pipeline for transductive finetune projects.

    Returns ``(selected, flagged)`` DataFrames and writes two CSV files into
    *output_dir*.
    """
    seeds = set(expected_seeds or [0, 1, 2, 3])
    dmi = load_transductive_dataset_monitor_info()
    ft_data_name = extract_transductive_finetune_dataset_name(project)
    print(f"  [debug] extracted transductive dataset name: {ft_data_name!r}")
    print(f"  [debug] known datasets in DMI ({len(dmi)}): "
          f"{sorted(dmi.keys())[:10]}{'…' if len(dmi) > 10 else ''}")

    runs = fetch_runs(entity, project, state=state)
    raw = build_raw_table(runs, project)
    if raw.empty:
        return pd.DataFrame(), pd.DataFrame()

    raw = filter_seed_subsample(raw, project)
    if raw.empty:
        return pd.DataFrame(), pd.DataFrame()

    pruned = drop_constant_columns(raw)
    agg, flagged = aggregate_seeds(
        pruned,
        seeds,
        dataset_monitor_info=dmi,
        finetune_data_name=ft_data_name,
        select_on_test=select_on_test,
    )
    selected = select_best_rows(agg, select_on_test=select_on_test)

    out_dir = output_dir or DEFAULT_OUTPUT_DIR_T
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = sanitize_filename(project)
    selected_path = out_dir / f"{stem}.csv"
    flagged_path  = out_dir / f"{stem}_flagged.csv"
    selected.to_csv(selected_path, index=False)
    flagged.to_csv(flagged_path, index=False)
    print(f"  wrote {selected_path}  ({len(selected)} rows)")
    if len(flagged):
        print(f"  wrote {flagged_path}  ({len(flagged)} flagged groups)")

    return selected, flagged


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Process one transductive W&B finetune project to CSV.",
    )
    p.add_argument("--entity", required=True, help="W&B entity.")
    p.add_argument("--project", required=True, help="Transductive finetune W&B project name.")
    p.add_argument("--train-seeds", nargs="+", type=int, default=[0, 1, 2, 3])
    p.add_argument("--state", default="finished", help="W&B run state filter.")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Output directory for CSVs (default: outputs_transductive/processed_projects).")
    p.add_argument("--select-on-test", action="store_true",
                   help="Rank hyperparameters by test metric (default: validation).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or DEFAULT_OUTPUT_DIR_T
    process_project_transductive(
        args.entity,
        args.project,
        expected_seeds=args.train_seeds,
        state=args.state or "finished",
        output_dir=output_dir,
        select_on_test=args.select_on_test,
    )


if __name__ == "__main__":
    main()
