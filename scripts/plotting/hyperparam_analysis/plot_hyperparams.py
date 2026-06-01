#!/usr/bin/env python3
"""Boxplot grids showing how each hyperparameter affects the monitor metric.

For each (dataset, pretraining task, ft_fraction) combination, one figure is
saved with one subplot per ``hyperparam_*`` column.  Within each subplot, every
distinct hyperparameter value gets four narrow boxplots in two glued subgroups:

  Linear probe   : from scratch (red) | pretrained (green)
  Full fine-tune : from scratch (red) | pretrained (green)

Each boxplot pools **individual runs** (no seed aggregation).  Random-init
boxplots reuse runs from one canonical pretraining task per dataset (see
``shared_baseline``) so the from-scratch baseline is identical across methods.

Usage
-----
    python scripts/plotting/hyperparam_analysis/plot_hyperparams.py
    python scripts/plotting/hyperparam_analysis/plot_hyperparams.py \\
        --input outputs/all_runs.csv
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.plotting.hyperparam_analysis.columns import hyperparam_columns
from scripts.plotting.shared_baseline import find_canonical_pretrain, is_random_init_mode, random_init_runs

DEFAULT_INPUT = _SCRIPT_DIR / "outputs" / "all_runs.csv"
DEFAULT_OUTPUT_DIR = _SCRIPT_DIR / "outputs" / "figures"

COL_DATASET = "pretrained_config_dataset.loader.parameters.data_name"
COL_PRETRAIN = "pretrained_config_pretraining.task"
COL_FT_MODE = "ft_mode"
COL_FRACTION = "ft_fraction"
COL_MON_COL = "monitor_test_column"
COL_MON_METRIC = "monitor_metric"
COL_MON_MODE = "monitor_mode"

# (scope key, legend label, ft_modes in order: scratch then pretrained)
SCOPE_GROUPS: list[tuple[str, str, tuple[str, str]]] = [
    ("probe", "Linear probe", ("random-init-probe", "finetune-probe")),
    ("full", "Full fine-tune", ("random-init-full", "finetune-full")),
]

COLOR_SCRATCH = "#CC3311"     # red
COLOR_PRETRAINED = "#009933"  # green

PRETRAIN_DISPLAY: dict[str, str] = {
    "graphmaev2": "GraphMAEv2",
    "graphcl": "GraphCL",
    "dgi": "DGI",
    "vgae": "VGAE",
    "bgrl": "BGRL",
}

RC_PARAMS: dict = {
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "0.87",
    "grid.linewidth": 0.6,
}


def _pretrain_label(task: str) -> str:
    return PRETRAIN_DISPLAY.get(str(task), str(task))


def _direction_arrow(mode: str) -> str:
    return "↓" if str(mode).lower() == "min" else "↑"


def _metric_ylabel(metric: str, mode: str) -> str:
    return f"{metric.capitalize()}  {_direction_arrow(mode)}"


def _mode_color(mode: str) -> str:
    return COLOR_PRETRAINED if mode.startswith("finetune") else COLOR_SCRATCH


def _resolve_monitor(sub: pd.DataFrame) -> tuple[str, str, str]:
    metric_col = None
    metric_name = "metric"
    mode = "max"
    if COL_MON_COL in sub.columns:
        vals = sub[COL_MON_COL].dropna()
        if not vals.empty:
            metric_col = str(vals.iloc[0])
    if COL_MON_METRIC in sub.columns:
        vals = sub[COL_MON_METRIC].dropna()
        if not vals.empty:
            metric_name = str(vals.iloc[0])
    if COL_MON_MODE in sub.columns:
        vals = sub[COL_MON_MODE].dropna()
        if not vals.empty:
            mode = str(vals.iloc[0])
    if metric_col is None:
        metric_col = f"test/{metric_name}"
    return metric_col, metric_name, mode


def _category_sort_key(val: object) -> tuple[int, str | float]:
    s = str(val)
    try:
        return (0, float(s))
    except ValueError:
        return (1, s)


def _sorted_categories(series: pd.Series) -> list[str]:
    uniq = series.dropna().unique()
    return [str(v) for v in sorted(uniq, key=_category_sort_key)]


def _box_positions(n_cats: int) -> tuple[list[float], list[tuple[int, int, str]]]:
    """Return x positions and (cat_idx, scope_idx, mode) for each of 4 boxes per category."""
    # Offsets within one category: probe-scratch, probe-pretrained | full-scratch, full-pretrained
    offsets = [-0.27, -0.09, 0.09, 0.27]
    modes = [
        SCOPE_GROUPS[0][2][0],
        SCOPE_GROUPS[0][2][1],
        SCOPE_GROUPS[1][2][0],
        SCOPE_GROUPS[1][2][1],
    ]
    positions: list[float] = []
    meta: list[tuple[int, int, str]] = []
    for cat_idx in range(n_cats):
        base = float(cat_idx)
        for off, mode in zip(offsets, modes):
            positions.append(base + off)
            scope_idx = 0 if "probe" in mode else 1
            meta.append((cat_idx, scope_idx, mode))
    return positions, meta


def _draw_hyperparam_subplot(
    ax: plt.Axes,
    sub: pd.DataFrame,
    hparam_col: str,
    metric_col: str,
    *,
    ri_sub: pd.DataFrame | None = None,
) -> None:
    categories = _sorted_categories(sub[hparam_col])
    if not categories:
        ax.set_visible(False)
        return

    positions, meta = _box_positions(len(categories))
    data: list[list[float]] = []
    colors: list[str] = []

    for cat_idx, _scope_idx, mode in meta:
        cat = categories[cat_idx]
        if is_random_init_mode(mode) and ri_sub is not None and not ri_sub.empty:
            source = ri_sub
        else:
            source = sub
        mask = (
            (source[hparam_col].astype(str) == cat)
            & (source[COL_FT_MODE] == mode)
        )
        vals = source.loc[mask, metric_col].dropna().astype(float).tolist()
        data.append(vals)
        colors.append(_mode_color(mode))

    bp = ax.boxplot(
        data,
        positions=positions,
        widths=0.11,
        patch_artist=True,
        showfliers=True,
        manage_ticks=False,
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
        patch.set_edgecolor("0.25")
        patch.set_linewidth(0.8)
    for med in bp["medians"]:
        med.set_color("0.15")
        med.set_linewidth(1.2)

    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, rotation=25, ha="right")
    ax.set_title(hparam_col.removeprefix("hyperparam_").replace("_", " "), pad=6)

    # Subgroup hints under each category (x in data coords, y in axes coords)
    for cat_idx in range(len(categories)):
        ax.text(
            cat_idx - 0.18, -0.20, "probe", transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=7, color="0.45", clip_on=False,
        )
        ax.text(
            cat_idx + 0.18, -0.20, "full", transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=7, color="0.45", clip_on=False,
        )


def _plot_combo_figure(
    df: pd.DataFrame,
    dataset: str,
    pretrain: str,
    fraction: float,
    hparam_cols: list[str],
    output_dir: Path,
    fmt: str,
) -> Path | None:
    sub = df[
        (df[COL_DATASET] == dataset)
        & (df[COL_PRETRAIN] == pretrain)
        & (df[COL_FRACTION] == fraction)
    ].copy()
    if sub.empty:
        return None

    metric_col, metric_name, mode = _resolve_monitor(sub)
    if metric_col not in sub.columns:
        print(f"  skip {dataset}/{pretrain}/frac={fraction}: missing {metric_col!r}")
        return None

    active_hparams = [
        c for c in hparam_cols
        if c in sub.columns and sub[c].notna().any() and sub[c].nunique(dropna=True) > 1
    ]
    if not active_hparams:
        print(f"  skip {dataset}/{pretrain}/frac={fraction}: no varying hyperparams")
        return None

    canonical_pt = find_canonical_pretrain(df, dataset)
    ri_sub = random_init_runs(df, dataset, fraction, canonical_pt)

    n = len(active_hparams)
    ncols = min(3, n)
    nrows = math.ceil(n / ncols)
    fig_w = 4.2 * ncols
    fig_h = 3.6 * nrows + 0.8

    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)
    fig.subplots_adjust(hspace=0.55, wspace=0.35, top=0.88, bottom=0.12)

    for idx, hcol in enumerate(active_hparams):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        _draw_hyperparam_subplot(ax, sub, hcol, metric_col, ri_sub=ri_sub)
        if c == 0:
            ax.set_ylabel(_metric_ylabel(metric_name, mode))

    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].set_visible(False)

    frac_label = f"{fraction:g}"
    fig.suptitle(
        f"{dataset}  ·  {_pretrain_label(pretrain)}  ·  train fraction {frac_label}",
        fontsize=13,
        y=0.96,
    )

    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=COLOR_SCRATCH, edgecolor="0.25", alpha=0.65, label="From scratch"),
        Patch(facecolor=COLOR_PRETRAINED, edgecolor="0.25", alpha=0.65, label="Pretrained"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.99),
        frameon=False,
        fontsize=9,
    )

    safe_ds = str(dataset).replace("/", "-")
    safe_pt = str(pretrain).replace("/", "-")
    fname = f"hyperparams_{safe_ds}_{safe_pt}_frac{frac_label}.{fmt}"
    out_path = output_dir / fname
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return out_path


def plot_all(
    df: pd.DataFrame,
    output_dir: Path,
    fmt: str = "png",
) -> list[Path]:
    mpl.rcParams.update(RC_PARAMS)
    hparam_cols = hyperparam_columns(df)
    saved: list[Path] = []

    combos = (
        df[[COL_DATASET, COL_PRETRAIN, COL_FRACTION]]
        .drop_duplicates()
        .sort_values([COL_DATASET, COL_PRETRAIN, COL_FRACTION])
    )
    for _, row in combos.iterrows():
        path = _plot_combo_figure(
            df,
            dataset=row[COL_DATASET],
            pretrain=row[COL_PRETRAIN],
            fraction=float(row[COL_FRACTION]),
            hparam_cols=hparam_cols,
            output_dir=output_dir,
            fmt=fmt,
        )
        if path is not None:
            saved.append(path)
            print(f"  wrote {path}")

    return saved


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hyperparameter boxplot grids from raw runs.")
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--fmt", default="png", choices=["png", "pdf", "svg"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(f"Input CSV not found: {args.input}")
    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} runs from {args.input}")
    plot_all(df, args.output_dir, fmt=args.fmt)


if __name__ == "__main__":
    main()
