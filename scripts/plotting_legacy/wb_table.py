"""Small helpers to turn W&B finetune runs into flat tables (no sweep-specific naming)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# Always keep in the table even when constant across all runs in a project.
ALWAYS_KEEP_COLUMNS: frozenset[str] = frozenset({
    "pretrained_config_dataset.loader.parameters.data_name",
    "pretrained_config_model.model_name",
    "pretrained_config_pretraining.task",
    "pretrained_config_dataset.parameters.task",
    "pretrained_config_dataset.parameters.monitor_metric",
    "ft_mode",
    "ft_fraction",
})

# Left-most columns in per-project / combined CSV outputs.
LEADING_COLUMNS: list[str] = [
    "pretrained_config_dataset.loader.parameters.data_name",
    "pretrained_config_model.model_name",
    "pretrained_config_pretraining.task",
    "ft_mode",
    "ft_fraction",
]

# Identity for best-hyperparameter selection (subset of leading + finetune mode/fraction).
SELECTION_KEY_COLUMNS: list[str] = list(LEADING_COLUMNS)

SEED_COLUMN = "ft_train_seed"

# Dropped before grouping / column pruning (not hyperparameters).
METADATA_COLUMNS: frozenset[str] = frozenset({
    "wandb_run_id",
    "wandb_run_name",
    "wandb_state",
    "wandb_url",
    "wandb_group",
    "wandb_project",
    "pretrained_run_id",
    "pretrained_run_name",
    "ft_subset_seed",
    "n_train",
    "n_val",
    "n_test",
    "n_train_full",
    "n_trainable_params",
    "n_total_params",
})

METRIC_PREFIXES: tuple[str, ...] = ("test/", "val/", "best_", "best_epoch/")


def flatten_config(cfg: dict, prefix: str = "", sep: str = ".") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in (cfg or {}).items():
        key = f"{prefix}{sep}{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(flatten_config(v, key, sep))
        else:
            out[key] = v
    return out


def normalize_value(v: Any) -> Any:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    if isinstance(v, (list, dict)):
        return str(v)
    if isinstance(v, np.generic):
        return v.item()
    return v


def is_metric_column(col: str) -> bool:
    return col.startswith(METRIC_PREFIXES)


def is_hyperparam_column(col: str) -> bool:
    if col in METADATA_COLUMNS or col == SEED_COLUMN:
        return False
    if is_metric_column(col):
        return False
    if col.startswith("wandb_"):
        return False
    if col in (
        "monitor_task",
        "monitor_metric",
        "monitor_mode",
        "monitor_test_column",
        "monitor_selection_column",
        "selection_on",
    ):
        return False
    if col in (
        "seed_count", "seed_ok", "seeds_found", "n_runs",
        "selection_score", "selection_criterion_score", "selected",
    ):
        return False
    if col.startswith("pretrained_config_paths."):
        return False
    return True


def safe_nunique(df: pd.DataFrame, col: str) -> int:
    if col not in df.columns:
        return 0
    block = df.loc[:, col]
    if isinstance(block, pd.DataFrame):
        block = block.bfill(axis=1).iloc[:, 0]
    return int(block.nunique(dropna=True))


def dedupe_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or df.columns.is_unique:
        return df
    out: dict[str, pd.Series] = {}
    for name in pd.Index(df.columns).unique():
        block = df.loc[:, name]
        out[name] = block if isinstance(block, pd.Series) else block.bfill(axis=1).iloc[:, 0]
    return pd.DataFrame(out)


def drop_constant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove columns with a single unique value, except ``ALWAYS_KEEP_COLUMNS``."""
    drop: list[str] = []
    for col in pd.Index(df.columns).unique():
        if col in ALWAYS_KEEP_COLUMNS:
            continue
        if safe_nunique(df, col) <= 1:
            drop.append(col)
    if drop:
        df = df.drop(columns=drop)
    return dedupe_columns(df)


def monitor_info_from_row(row: pd.Series) -> tuple[str | None, str | None, str]:
    task = row.get("pretrained_config_dataset.parameters.task")
    metric = row.get("pretrained_config_dataset.parameters.monitor_metric")
    if metric is None or (isinstance(metric, float) and np.isnan(metric)):
        return None, None, "max"
    metric_s = str(metric).split("/")[-1]
    test_col = f"test/{metric_s}"
    mode = "max"
    if task is not None and not (isinstance(task, float) and np.isnan(task)):
        try:
            from topobench.utils.config_resolvers import get_monitor_mode
            mode = get_monitor_mode(str(task), metric_s)
        except Exception:
            ml = metric_s.lower()
            if "loss" in ml or "mae" in ml or "mse" in ml or str(task) == "regression":
                mode = "min"
    return str(task) if task is not None else None, metric_s, mode


def order_columns(df: pd.DataFrame) -> pd.DataFrame:
    leading = [c for c in LEADING_COLUMNS if c in df.columns]
    metrics = sorted(c for c in df.columns if is_metric_column(c))
    meta = [
        c for c in (
            "selection_score", "selection_criterion_score", "selected", "selection_on",
            "monitor_metric", "monitor_test_column", "monitor_selection_column",
            "monitor_mode", "wandb_project", "n_runs", "seed_count",
        )
        if c in df.columns
    ]
    middle = [
        c for c in df.columns
        if c not in leading and c not in metrics and c not in meta
    ]
    return df[leading + middle + metrics + meta]
