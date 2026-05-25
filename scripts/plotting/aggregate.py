"""Aggregate W&B fine-tuning runs into analysis-ready CSV tables."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import wandb

from scripts.plotting.defaults import (
    METHOD_PARAM_SHORT_NAMES,
    METHOD_SWEPT_KEYS,
    NON_HYPERPARAM_COLUMNS,
    NON_HYPERPARAM_PREFIXES,
    SELECTION_GROUP_KEYS,
    SHARED_COLUMN_NAMES,
    SHARED_SWEPT_KEYS,
    parse_pretrain_project,
)

# Optional: reuse monitor helpers when topobench is importable
try:
    from topobench.utils.config_resolvers import get_monitor_mode
except ImportError:  # pragma: no cover

    def get_monitor_mode(task: str, metric: str = "accuracy") -> str:
        metric_lower = metric.lower()
        if "loss" in metric_lower or "mae" in metric_lower or "mse" in metric_lower:
            return "min"
        if task == "regression":
            return "min"
        return "max"


# ──────────────────────────────────────────────────────────────────────────────
# W&B ingestion
# ──────────────────────────────────────────────────────────────────────────────

def fetch_finetune_runs(
    entity: str,
    projects: Iterable[str],
    *,
    state: str = "finished",
) -> list[Any]:
    """Fetch all runs from the given finetune W&B projects."""
    api = wandb.Api()
    runs: list[Any] = []
    for project in projects:
        path = f"{entity}/{project}"
        filters = {"state": state} if state else None
        batch = list(api.runs(path, filters=filters))
        print(f"  {path}: {len(batch)} run(s)")
        runs.extend(batch)
    return runs


def _flatten_config(cfg: dict, prefix: str = "", sep: str = ".") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in (cfg or {}).items():
        key = f"{prefix}{sep}{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(_flatten_config(v, key, sep))
        else:
            out[key] = v
    return out


def _normalize_value(v: Any) -> Any:
    """Make values hashable/comparable for grouping."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    if isinstance(v, (list, dict)):
        return str(v)
    if isinstance(v, np.generic):
        return v.item()
    return v


def _dataset_basename(value: Any) -> str | None:
    if value is None:
        return None
    s = str(value).replace("\\", "/")
    return s.rsplit("/", 1)[-1]


def _extract_monitor_info(row: dict[str, Any]) -> tuple[str | None, str | None, str | None]:
    """Return (task, metric_name, test_column) from flattened pretrained config."""
    task = row.get("pretrained_config_dataset.parameters.task")
    metric = row.get("pretrained_config_dataset.parameters.monitor_metric")
    if task is None or metric is None:
        return None, None, None
    task_s = str(task)
    metric_s = str(metric).split("/")[-1]
    test_col = f"test/{metric_s}"
    return task_s, metric_s, test_col


def run_to_record(run: Any, project: str) -> dict[str, Any]:
    """Convert one W&B run into a flat dict for DataFrame construction."""
    flat_cfg = _flatten_config(dict(run.config))
    summary = dict(run.summary)

    method, dataset_from_proj = parse_pretrain_project(project)
    dataset_cfg = _dataset_basename(
        flat_cfg.get("pretrained_config_dataset")
        or flat_cfg.get("pretrained_config_dataset.loader")
    )

    record: dict[str, Any] = {
        "wandb_project": project,
        "wandb_run_id": run.id,
        "wandb_run_name": run.name,
        "wandb_state": run.state,
        "wandb_group": getattr(run, "group", None),
        "wandb_url": run.url,
        "pretraining_method": method,
        "dataset": dataset_from_proj if dataset_from_proj != "unknown" else dataset_cfg,
    }

    for k, v in flat_cfg.items():
        record[k] = _normalize_value(v)

    # Explicit finetune fields (may duplicate config)
    for key in (
        "ft_mode", "ft_fraction", "ft_pooling", "ft_train_seed",
        "ft_max_epochs", "ft_patience", "ft_seed", "ft_subset_seed",
        "pretrained_run_id", "pretrained_run_name",
        "n_train", "n_val", "n_test", "n_train_full",
        "n_trainable_params", "n_total_params",
    ):
        if key in flat_cfg and key not in record:
            record[key] = _normalize_value(flat_cfg[key])

    task, metric_name, test_col = _extract_monitor_info(record)
    record["monitor_task"] = task
    record["monitor_metric"] = metric_name
    record["monitor_test_column"] = test_col
    if task and metric_name:
        record["monitor_mode"] = get_monitor_mode(task, metric_name)

    for k, v in summary.items():
        if k.startswith("test/") or k.startswith("best_") or k.startswith("best_epoch/"):
            try:
                record[k] = float(v)
            except (TypeError, ValueError):
                record[k] = v

    if record.get("gin_hidden") is None:
        h = flat_cfg.get("pretrained_config_model.feature_encoder.out_channels")
        if h is not None:
            record["gin_hidden"] = _normalize_value(h)
    if record.get("gin_num_layers") is None:
        L = flat_cfg.get("pretrained_config_model.backbone.num_layers")
        if L is not None:
            record["gin_num_layers"] = _normalize_value(L)
    if record.get("weight_decay") is None:
        wd = flat_cfg.get("pretrained_config_optimizer.parameters.weight_decay")
        if wd is not None:
            record["weight_decay"] = _normalize_value(wd)
    if record.get("learning_rate") is None:
        lr = flat_cfg.get("pretrained_config_optimizer.parameters.lr")
        if lr is not None:
            record["learning_rate"] = _normalize_value(lr)

    return record


def build_raw_runs_dataframe(runs: list[Any], project_by_run: dict[str, str]) -> pd.DataFrame:
    """Build a DataFrame with one row per W&B run."""
    records = [
        run_to_record(r, project_by_run.get(r.id, r.project))
        for r in runs
    ]
    if not records:
        return pd.DataFrame()
    return pd.DataFrame.from_records(records)


# ──────────────────────────────────────────────────────────────────────────────
# Hyperparameter grouping & seed aggregation
# ──────────────────────────────────────────────────────────────────────────────

def _is_hyperparam_column(col: str, df: pd.DataFrame) -> bool:
    if col in NON_HYPERPARAM_COLUMNS:
        return False
    if any(col.startswith(p) for p in NON_HYPERPARAM_PREFIXES):
        return False
    if col.endswith("_mean") or col.endswith("_std"):
        return False
    if col.startswith("wandb_"):
        return False
    if col in ("monitor_task", "monitor_metric", "monitor_mode", "monitor_test_column"):
        return False
    # Outcome / bookkeeping
    if col.startswith("pretrained_config_paths."):
        return False
    return True


def grouping_columns(df: pd.DataFrame) -> list[str]:
    """Columns that define a unique hyperparameter setting (excluding train seed)."""
    cols = []
    for c in df.columns:
        if not _is_hyperparam_column(c, df):
            continue
        if c == "ft_train_seed":
            continue
        cols.append(c)
    return cols


def aggregate_seed_groups(
    df: pd.DataFrame,
    expected_seeds: list[int],
    *,
    seed_col: str = "ft_train_seed",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Group by hyperparameters, require exact seed count, aggregate test metrics.

    Returns
    -------
    aggregated : DataFrame
        One row per valid hyperparameter group with ``*_mean`` / ``*_std`` test cols.
    flagged : DataFrame
        Groups that did not have exactly ``len(expected_seeds)`` distinct seeds.
    """
    if df.empty:
        return df, pd.DataFrame()

    expected_set = {int(s) for s in expected_seeds}
    n_expected = len(expected_set)

    test_cols = [c for c in df.columns if c.startswith("test/")]
    group_cols = grouping_columns(df)
    passthrough_cols = [
        c for c in (
            "monitor_task", "monitor_metric", "monitor_test_column", "monitor_mode",
            "pretrained_run_id", "pretrained_run_name", "wandb_project",
        )
        if c in df.columns
    ]

    aggregated_rows: list[dict[str, Any]] = []
    flagged_rows: list[dict[str, Any]] = []

    grouped = df.groupby(group_cols, dropna=False, sort=False)
    for key, grp in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        base = dict(zip(group_cols, key))

        seeds_raw = grp[seed_col].dropna().unique() if seed_col in grp.columns else []
        seeds = {_normalize_value(s) for s in seeds_raw}
        seeds_int = set()
        for s in seeds:
            try:
                seeds_int.add(int(s))
            except (TypeError, ValueError):
                pass

        seed_ok = seeds_int == expected_set
        base["seed_count"] = len(seeds_int)
        base["seed_ok"] = seed_ok
        base["seeds_found"] = ",".join(str(s) for s in sorted(seeds_int))

        if not seed_ok:
            flagged_rows.append({**base, "flag_reason": "seed_count_mismatch"})
            continue

        row = dict(base)
        row["n_runs"] = len(grp)
        for col in passthrough_cols:
            row[col] = grp[col].iloc[0]
        for col in test_cols:
            vals = grp[col].dropna()
            if len(vals) == 0:
                continue
            row[f"{col}_mean"] = float(vals.mean())
            row[f"{col}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0

        aggregated_rows.append(row)

    agg_df = pd.DataFrame(aggregated_rows)
    flag_df = pd.DataFrame(flagged_rows)
    return agg_df, flag_df


# ──────────────────────────────────────────────────────────────────────────────
# Column naming (varied hyperparams only)
# ──────────────────────────────────────────────────────────────────────────────

def detect_varied_columns(
    df: pd.DataFrame,
    *,
    exclude: Iterable[str] | None = None,
) -> list[str]:
    """Return hyperparameter columns with more than one distinct value."""
    exclude_set = set(exclude or ())
    varied = []
    for col in df.columns:
        if col in exclude_set:
            continue
        if not _is_hyperparam_column(col, df):
            continue
        nunique = df[col].nunique(dropna=True)
        if nunique > 1:
            varied.append(col)
    return varied


def _strip_pretrain_prefix(col: str) -> str:
    if col.startswith("pretrained_config_"):
        return col[len("pretrained_config_") :]
    return col


def rename_hyperparam_columns(
    df: pd.DataFrame,
    varied_cols: list[str],
) -> pd.DataFrame:
    """Rename varied columns: shared arch, ``{method}_param_*``, finetune ``ft_*``."""
    rename_map: dict[str, str] = {}
    method_keys = set()
    for m, keys in METHOD_SWEPT_KEYS.items():
        method_keys.update(keys)

    for col in varied_cols:
        if col in rename_map.values():
            continue
        if col in SHARED_COLUMN_NAMES:
            rename_map[col] = SHARED_COLUMN_NAMES[col]
            continue
        if col.startswith("ft_"):
            rename_map[col] = col
            continue
        if col in ("pretraining_method", "dataset", "gin_hidden", "gin_num_layers",
                   "weight_decay", "learning_rate"):
            rename_map[col] = col
            continue

        hydra_key = _strip_pretrain_prefix(col)
        short = METHOD_PARAM_SHORT_NAMES.get(hydra_key)
        if short is None:
            short = hydra_key.replace(".", "_")

        method = None
        if "pretraining_method" in df.columns and len(df) > 0:
            method = str(df["pretraining_method"].iloc[0])
        elif col.startswith("pretrained_config_"):
            for m in METHOD_SWEPT_KEYS:
                if hydra_key in METHOD_SWEPT_KEYS[m]:
                    method = m
                    break

        if hydra_key in method_keys and method:
            rename_map[col] = f"{method}_param_{short}"
        elif hydra_key in SHARED_SWEPT_KEYS:
            rename_map[col] = SHARED_COLUMN_NAMES.get(hydra_key, short)
        elif col.startswith("pretrained_config_"):
            rename_map[col] = f"pretrain_{short}"
        else:
            rename_map[col] = col

    out = df.copy()
    out = out.rename(columns={k: v for k, v in rename_map.items() if k in out.columns})
    return out


def slim_to_varied_hyperparams(
    df: pd.DataFrame,
    varied_cols: list[str],
) -> pd.DataFrame:
    """Keep selection keys, monitor metadata, test metric aggregates, and varied HPs."""
    metric_cols = [c for c in df.columns if c.startswith("test/")]
    meta_cols = [
        "wandb_project", "pretraining_method", "dataset",
        "monitor_metric", "monitor_mode", "monitor_test_column", "monitor_task",
        "seed_count", "seed_ok", "n_runs", "seeds_found",
        "selection_score", "selected",
        "pretrained_run_id", "pretrained_run_name",
    ]
    keep = set(metric_cols) | set(meta_cols) | set(varied_cols)
    keep.update(k for k in SELECTION_GROUP_KEYS if k in df.columns)
    cols_ordered = [c for c in df.columns if c in keep]
    return df[cols_ordered]


# ──────────────────────────────────────────────────────────────────────────────
# Best hyperparameter selection per (model, dataset, method, mode, fraction)
# ──────────────────────────────────────────────────────────────────────────────

def _selection_group_cols(df: pd.DataFrame) -> list[str]:
    present = []
    for k in SELECTION_GROUP_KEYS:
        if k in df.columns:
            present.append(k)
    return present


def select_best_hyperparams(agg_df: pd.DataFrame) -> pd.DataFrame:
    """For each (dataset, method, arch, ft_mode, ft_fraction), keep best monitor mean."""
    if agg_df.empty:
        return agg_df

    group_cols = _selection_group_cols(agg_df)
    if not group_cols:
        raise ValueError(
            f"Selection group columns missing. Expected some of: {SELECTION_GROUP_KEYS}"
        )

    selected_rows: list[pd.Series] = []

    for _, grp in agg_df.groupby(group_cols, dropna=False, sort=False):
        scores: list[tuple[int, float, pd.Series]] = []
        for idx, row in grp.iterrows():
            test_col = row.get("monitor_test_column")
            if test_col is None or (isinstance(test_col, float) and np.isnan(test_col)):
                continue
            mean_col = f"{test_col}_mean"
            if mean_col not in row.index:
                continue
            val = row[mean_col]
            if val is None or (isinstance(val, float) and np.isnan(val)):
                continue
            mode = row.get("monitor_mode", "max")
            if mode is None or (isinstance(mode, float) and np.isnan(mode)):
                task = row.get("monitor_task", "classification")
                metric = row.get("monitor_metric", "accuracy")
                mode = get_monitor_mode(str(task), str(metric))
            scores.append((idx, float(val), row))

        if not scores:
            continue

        mode = str(grp["monitor_mode"].iloc[0]) if "monitor_mode" in grp.columns else "max"
        if mode == "min":
            best_idx, best_val, best_row = min(scores, key=lambda x: x[1])
        else:
            best_idx, best_val, best_row = max(scores, key=lambda x: x[1])

        best_row = best_row.copy()
        best_row["selection_score"] = best_val
        best_row["selected"] = True
        selected_rows.append(best_row)

    if not selected_rows:
        return pd.DataFrame()

    out = pd.DataFrame(selected_rows)
    out = out.reset_index(drop=True)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# End-to-end pipeline
# ──────────────────────────────────────────────────────────────────────────────

def process_finetune_projects(
    entity: str,
    projects: list[str],
    *,
    expected_seeds: list[int] | None = None,
    state: str = "finished",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Full pipeline: fetch → seed aggregate → rename → select best.

    Returns
    -------
    final_df, seed_aggregated_df, flagged_df
    """
    runs = fetch_finetune_runs(entity, projects, state=state)
    project_by_run = {r.id: r.project for r in runs}
    raw = build_raw_runs_dataframe(runs, project_by_run)

    seeds = expected_seeds if expected_seeds is not None else [0, 1, 2]
    agg, flagged = aggregate_seed_groups(raw, seeds)

    varied = detect_varied_columns(agg)
    renamed = rename_hyperparam_columns(agg, varied)
    varied_renamed = [
        c for c in detect_varied_columns(renamed)
        if not c.startswith("test/") and c not in NON_HYPERPARAM_COLUMNS
    ]
    slim = slim_to_varied_hyperparams(renamed, varied_renamed)

    final = select_best_hyperparams(slim)
    return final, renamed, flagged
