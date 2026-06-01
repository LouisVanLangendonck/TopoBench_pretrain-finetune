"""Column selection and renaming for raw hyperparameter tables."""

from __future__ import annotations

import re

import pandas as pd

from scripts.plotting.wb_table import is_metric_column

IDENTITY_COLUMNS: list[str] = [
    "pretrained_config_dataset.loader.parameters.data_name",
    "pretrained_config_model.model_name",
    "pretrained_config_pretraining.task",
    "ft_mode",
    "ft_fraction",
]

# Kept for traceability; not used in plots.
OPTIONAL_METADATA_COLUMNS: list[str] = [
    "ft_train_seed",
    "wandb_run_id",
    "wandb_project",
]

HYPERPARAM_PREFIX = "hyperparam_"
_MODEL_CONFIG_PREFIX = "pretrained_config_model."


def _hyperparam_short_name(full_col: str) -> str:
    return full_col.split(".")[-1]


def hyperparam_rename_map(columns: pd.Index) -> dict[str, str]:
    """Map source columns → ``hyperparam_*`` names."""
    mapping: dict[str, str] = {}
    identity_model_name = "pretrained_config_model.model_name"
    for col in columns:
        if col.startswith(_MODEL_CONFIG_PREFIX) and col != identity_model_name:
            mapping[col] = f"{HYPERPARAM_PREFIX}{_hyperparam_short_name(col)}"
    if "ft_pooling" in columns:
        mapping["ft_pooling"] = f"{HYPERPARAM_PREFIX}ft_pooling"
    return mapping


def hyperparam_columns(df: pd.DataFrame) -> list[str]:
    return sorted(c for c in df.columns if c.startswith(HYPERPARAM_PREFIX))


def transform_raw_table(df: pd.DataFrame) -> pd.DataFrame:
    """Keep metrics + identity + model hyperparams; rename model config keys."""
    if df.empty:
        return df

    rename_map = hyperparam_rename_map(df.columns)
    metric_cols = [c for c in df.columns if is_metric_column(c)]
    meta_cols = [c for c in OPTIONAL_METADATA_COLUMNS if c in df.columns]
    keep = [
        c for c in IDENTITY_COLUMNS
        if c in df.columns
    ] + meta_cols + list(rename_map.keys()) + metric_cols

    out = df.loc[:, keep].copy()
    out = out.rename(columns=rename_map)
    return order_hyperparam_columns(out)


def order_hyperparam_columns(df: pd.DataFrame) -> pd.DataFrame:
    leading = [c for c in IDENTITY_COLUMNS if c in df.columns]
    meta = [c for c in OPTIONAL_METADATA_COLUMNS if c in df.columns]
    hparams = hyperparam_columns(df)
    metrics = sorted(c for c in df.columns if is_metric_column(c))
    monitor = [
        c for c in (
            "monitor_task", "monitor_metric", "monitor_test_column", "monitor_mode",
        )
        if c in df.columns
    ]
    return df[leading + meta + hparams + metrics + monitor]


def sanitize_filename(project: str) -> str:
    return re.sub(r"[^\w\-.]+", "_", project)
