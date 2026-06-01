"""Share one dataset-level random-init baseline across all pretraining tasks.

Random-init runs depend only on model architecture and finetuning setup, not on
the pretraining method.  New sweeps often omit duplicate random-init runs per
pretraining task; plotting reuses the baseline from whichever pretraining task
still has them for each dataset.
"""

from __future__ import annotations

import pandas as pd

COL_DATASET = "pretrained_config_dataset.loader.parameters.data_name"
COL_PRETRAIN = "pretrained_config_pretraining.task"
COL_FT_MODE = "ft_mode"
COL_FRACTION = "ft_fraction"

RANDOM_INIT_PREFIX = "random-init"


def is_random_init_mode(ft_mode: object) -> bool:
    return str(ft_mode).startswith(RANDOM_INIT_PREFIX)


def find_canonical_pretrain(df: pd.DataFrame, dataset: str) -> str | None:
    """Return a pretraining task that has random-init rows for *dataset*."""
    if df.empty or COL_DATASET not in df.columns:
        return None
    sub = df[df[COL_DATASET].astype(str) == str(dataset)]
    for pretrain in sorted(sub[COL_PRETRAIN].dropna().astype(str).unique()):
        ri = sub[
            (sub[COL_PRETRAIN].astype(str) == pretrain)
            & sub[COL_FT_MODE].map(is_random_init_mode)
        ]
        if not ri.empty:
            return pretrain
    return None


def _baseline_rows(df: pd.DataFrame, dataset: str, canonical_pt: str) -> dict[tuple[str, float], pd.Series]:
    rows = df[
        (df[COL_DATASET].astype(str) == str(dataset))
        & (df[COL_PRETRAIN].astype(str) == canonical_pt)
        & df[COL_FT_MODE].map(is_random_init_mode)
    ]
    baseline: dict[tuple[str, float], pd.Series] = {}
    for _, row in rows.iterrows():
        baseline[(str(row[COL_FT_MODE]), float(row[COL_FRACTION]))] = row
    return baseline


def apply_shared_random_init_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """Copy canonical random-init rows to every pretraining task per dataset."""
    if df.empty or COL_DATASET not in df.columns:
        return df

    out = df.copy()
    for dataset in out[COL_DATASET].dropna().astype(str).unique():
        canonical_pt = find_canonical_pretrain(out, dataset)
        if canonical_pt is None:
            continue

        baseline = _baseline_rows(out, dataset, canonical_pt)
        if not baseline:
            continue

        pretrains = (
            out.loc[out[COL_DATASET].astype(str) == dataset, COL_PRETRAIN]
            .dropna()
            .astype(str)
            .unique()
        )
        for pretrain in pretrains:
            if pretrain == canonical_pt:
                continue
            for (ft_mode, frac), canon_row in baseline.items():
                mask = (
                    (out[COL_DATASET].astype(str) == dataset)
                    & (out[COL_PRETRAIN].astype(str) == pretrain)
                    & (out[COL_FT_MODE].astype(str) == ft_mode)
                    & (out[COL_FRACTION].astype(float) == frac)
                )
                if mask.any():
                    idx = out.index[mask]
                    for col in canon_row.index:
                        if col == COL_PRETRAIN:
                            continue
                        if col in out.columns:
                            out.loc[idx, col] = canon_row[col]
                else:
                    new_row = canon_row.copy()
                    new_row[COL_PRETRAIN] = pretrain
                    out = pd.concat([out, pd.DataFrame([new_row])], ignore_index=True)

    return out


def random_init_runs(
    df: pd.DataFrame,
    dataset: str,
    fraction: float,
    canonical_pretrain: str | None = None,
) -> pd.DataFrame:
    """Individual random-init runs used as the shared baseline for boxplots."""
    pt = canonical_pretrain or find_canonical_pretrain(df, dataset)
    if pt is None:
        return df.iloc[0:0]
    return df[
        (df[COL_DATASET].astype(str) == str(dataset))
        & (df[COL_PRETRAIN].astype(str) == pt)
        & (df[COL_FRACTION].astype(float) == float(fraction))
        & df[COL_FT_MODE].map(is_random_init_mode)
    ].copy()
