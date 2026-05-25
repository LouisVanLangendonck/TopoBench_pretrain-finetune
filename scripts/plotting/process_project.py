#!/usr/bin/env python3
"""Aggregate + select best runs for a single W&B finetune project.

Steps (only uses runs from this project):
  1. Fetch runs, build a flat table.
  2. Drop constant columns (except dataset / model / pretraining task names).
  3. Group by all remaining hyperparameters except ``ft_train_seed``.
  4. Keep groups with exactly ``N`` seeds; average ``test/*`` metrics.
  5. Per (dataset, model, pretraining task, ft_mode, ft_fraction), keep the row
     with the best mean test monitor metric; drop all other rows.
  6. Write ``scripts/plotting/outputs/processed_projects/<project>.csv``.

Usage
-----
    python scripts/plotting/process_project.py \\
        --entity <wandb-entity> \\
        --project finetune_gin_pretrain_sweep_dgi_MUTAG
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import wandb
import yaml

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.plotting.wb_table import (
    SEED_COLUMN,
    SELECTION_KEY_COLUMNS,
    dedupe_columns,
    drop_constant_columns,
    flatten_config,
    is_hyperparam_column,
    is_metric_column,
    normalize_value,
    order_columns,
)

DEFAULT_OUTPUT_DIR = _SCRIPT_DIR / "outputs" / "processed_projects"

# Project name convention: finetune_{model}_pretrain_sweep_{pretraining_method}_{dataset_name}
_PROJECT_NAME_RE = re.compile(r"^finetune_[^_]+_pretrain_sweep_[^_]+_(.+)$")


def load_dataset_monitor_info(
    config_root: Path | None = None,
) -> dict[str, tuple[str, str]]:
    """Scan configs/dataset/graph/*.yaml and return {data_name: (task, monitor_metric)}.

    This is the ground-truth fallback for when W&B runs didn't log these fields.
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
        if data_name and task and monitor_metric:
            result[str(data_name)] = (str(task), str(monitor_metric))
    return result


def extract_finetune_dataset_name(project: str) -> str | None:
    """Parse the finetuning dataset name out of the W&B project name.

    Convention: finetune_{model}_pretrain_sweep_{pretraining_method}_{dataset_name}
    Examples:
      finetune_gin_pretrain_sweep_graphmaev2_IMDB-BINARY  -> IMDB-BINARY
      finetune_gin_pretrain_sweep_dgi_Clearance_Hepatocyte_AZ  -> Clearance_Hepatocyte_AZ
    """
    m = _PROJECT_NAME_RE.match(project)
    return m.group(1) if m else None


def sanitize_filename(project: str) -> str:
    return re.sub(r"[^\w\-.]+", "_", project)


def fetch_runs(entity: str, project: str, state: str = "finished") -> list[Any]:
    api = wandb.Api()
    path = f"{entity}/{project}"
    filters = {"state": state} if state else None
    runs = list(api.runs(path, filters=filters))
    print(f"  {path}: {len(runs)} run(s)")
    return runs


def run_to_row(run: Any, project: str) -> dict[str, Any]:
    flat = flatten_config(dict(run.config))
    summary = dict(run.summary)

    row: dict[str, Any] = {
        "wandb_project": project,
        "wandb_run_id": run.id,
        "wandb_run_name": run.name,
        "wandb_state": run.state,
    }
    for k, v in flat.items():
        row[k] = normalize_value(v)
    for k, v in summary.items():
        if is_metric_column(k):
            try:
                row[k] = float(v)
            except (TypeError, ValueError):
                row[k] = v
    return row


def build_raw_table(runs: list[Any], project: str) -> pd.DataFrame:
    if not runs:
        return pd.DataFrame()
    return dedupe_columns(pd.DataFrame.from_records([run_to_row(r, project) for r in runs]))


def hyperparam_group_columns(df: pd.DataFrame) -> list[str]:
    return [
        c for c in pd.Index(df.columns).unique()
        if is_hyperparam_column(c) and c != SEED_COLUMN
    ]


def _derive_mode(task: str, metric: str) -> str:
    mode = "max"
    try:
        from topobench.utils.config_resolvers import get_monitor_mode
        mode = get_monitor_mode(task, metric)
    except Exception:
        ml = metric.lower()
        if "loss" in ml or "mae" in ml or "mse" in ml or task == "regression":
            mode = "min"
    return mode


def _resolve_monitor_info(
    finetune_data_name: str | None,
    dataset_monitor_info: dict[str, tuple[str, str]],
) -> tuple[str | None, str | None, str]:
    """Return (task, monitor_metric, mode) for the finetuning evaluation.

    Primary source: local configs/dataset/graph/{data_name}.yaml (authoritative,
    independent of what the W&B run config logged).
    The pretrained_config_* fields in the run describe the PRETRAIN setup, not the
    finetuning evaluation, so we deliberately ignore them here.
    """
    if finetune_data_name:
        info = dataset_monitor_info.get(finetune_data_name)
        if info is not None:
            task_fb, metric_fb = info
            return task_fb, metric_fb, _derive_mode(task_fb, metric_fb)
    return None, None, "max"


def aggregate_seeds(
    df: pd.DataFrame,
    expected_seeds: set[int],
    dataset_monitor_info: dict[str, tuple[str, str]] | None = None,
    finetune_data_name: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (valid aggregated groups, flagged groups)."""
    if df.empty:
        return df, pd.DataFrame()

    _dmi = dataset_monitor_info or {}
    df = dedupe_columns(df)
    test_cols = [c for c in df.columns if c.startswith("test/")]
    group_cols = hyperparam_group_columns(df)

    task, metric, mode = _resolve_monitor_info(finetune_data_name, _dmi)
    print(f"  [debug] finetune_data_name={finetune_data_name!r}  "
          f"monitor → task={task!r}  metric={metric!r}  mode={mode!r}")
    print(f"  [debug] test cols in data: {test_cols}")
    print(f"  [debug] n_groups={df.groupby(group_cols, dropna=False).ngroups}  "
          f"expected_seeds={sorted(expected_seeds)}")

    valid_rows: list[dict[str, Any]] = []
    flagged_rows: list[dict[str, Any]] = []

    for key, grp in df.groupby(group_cols, dropna=False, sort=False):
        if not isinstance(key, tuple):
            key = (key,)
        base = dict(zip(group_cols, key))

        seeds_found: set[int] = set()
        if SEED_COLUMN in grp.columns:
            for s in grp[SEED_COLUMN].dropna().unique():
                try:
                    seeds_found.add(int(s))
                except (TypeError, ValueError):
                    pass

        base["seed_count"] = len(seeds_found)
        base["seeds_found"] = ",".join(str(s) for s in sorted(seeds_found))

        if seeds_found != expected_seeds:
            base["flag_reason"] = "seed_count_mismatch"
            flagged_rows.append(base)
            continue

        row = dict(base)
        row["n_runs"] = len(grp)
        row["monitor_task"] = task
        row["monitor_metric"] = metric
        row["monitor_mode"] = mode
        row["monitor_test_column"] = f"test/{metric}" if metric else None

        for col in test_cols:
            vals = grp[col].dropna()
            if len(vals) == 0:
                continue
            row[f"{col}_mean"] = float(vals.mean())
            row[f"{col}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0

        valid_rows.append(row)

    print(f"  [debug] valid_groups={len(valid_rows)}  flagged_groups={len(flagged_rows)}")
    return pd.DataFrame(valid_rows), pd.DataFrame(flagged_rows)


def select_best_rows(agg: pd.DataFrame) -> pd.DataFrame:
    """One row per selection-key combo with best mean test monitor metric."""
    if agg.empty:
        return agg

    keys = [k for k in SELECTION_KEY_COLUMNS if k in agg.columns]
    if not keys:
        raise ValueError(f"Missing selection columns. Need some of: {SELECTION_KEY_COLUMNS}")

    picked: list[pd.Series] = []
    for _, grp in agg.groupby(keys, dropna=False, sort=False):
        candidates: list[tuple[float, pd.Series]] = []
        for _, row in grp.iterrows():
            test_col = row.get("monitor_test_column")
            if not test_col or (isinstance(test_col, float) and np.isnan(test_col)):
                continue
            mean_col = f"{test_col}_mean"
            if mean_col not in row.index:
                continue
            val = row[mean_col]
            if val is None or (isinstance(val, float) and np.isnan(val)):
                continue
            candidates.append((float(val), row))

        if not candidates:
            # Debug: show why first failing group has no candidates (once only).
            if not picked:
                sample_row = next(grp.iterrows())[1]
                tc = sample_row.get("monitor_test_column")
                mc = f"{tc}_mean" if tc else None
                print(f"  [debug:select] first empty-candidate group: "
                      f"monitor_test_column={tc!r}  "
                      f"mean_col_present={mc in sample_row.index if mc else False}  "
                      f"available_mean_cols={[c for c in sample_row.index if c.endswith('_mean')][:5]}")
            continue

        mode = str(grp["monitor_mode"].iloc[0]) if "monitor_mode" in grp.columns else "max"
        _, best_row = (
            min(candidates, key=lambda x: x[0])
            if mode == "min"
            else max(candidates, key=lambda x: x[0])
        )
        best_row = best_row.copy()
        best_row["selection_score"] = best_row[f"{best_row['monitor_test_column']}_mean"]
        best_row["selected"] = True
        picked.append(best_row)

    if not picked:
        return pd.DataFrame()
    return order_columns(pd.DataFrame(picked).reset_index(drop=True))


def process_project(
    entity: str,
    project: str,
    *,
    expected_seeds: list[int] | None = None,
    state: str = "finished",
    output_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full per-project pipeline. Returns (selected, flagged)."""
    seeds = set(expected_seeds or [0, 1, 2])
    dmi = load_dataset_monitor_info()
    ft_data_name = extract_finetune_dataset_name(project)
    print(f"  [debug] extracted finetune dataset name from project: {ft_data_name!r}")
    print(f"  [debug] known datasets in local configs: {sorted(dmi.keys())}")
    runs = fetch_runs(entity, project, state=state)
    raw = build_raw_table(runs, project)
    if raw.empty:
        return pd.DataFrame(), pd.DataFrame()

    pruned = drop_constant_columns(raw)
    agg, flagged = aggregate_seeds(
        pruned, seeds, dataset_monitor_info=dmi, finetune_data_name=ft_data_name,
    )
    selected = select_best_rows(agg)

    out_dir = output_dir or DEFAULT_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = sanitize_filename(project)
    selected_path = out_dir / f"{stem}.csv"
    flagged_path = out_dir / f"{stem}_flagged.csv"

    selected.to_csv(selected_path, index=False)
    flagged.to_csv(flagged_path, index=False)
    print(f"  wrote {selected_path}  ({len(selected)} rows)")
    if len(flagged):
        print(f"  wrote {flagged_path}  ({len(flagged)} flagged groups)")

    return selected, flagged


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Process one W&B finetune project to CSV.")
    p.add_argument("--entity", required=True, help="W&B entity.")
    p.add_argument("--project", required=True, help="W&B finetune project name.")
    p.add_argument("--train-seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--state", default="finished", help="W&B state filter (empty = all).")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    state = args.state if args.state else None
    process_project(
        args.entity,
        args.project,
        expected_seeds=args.train_seeds,
        state=state or "finished",
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
