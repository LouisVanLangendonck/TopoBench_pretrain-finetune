#!/usr/bin/env python3
"""Publication-quality learning-curve plots from aggregated finetuning results.

For each dataset:
  - One figure with one subplot per pretraining method (columns).
  - Each subplot shows one curve per ft_mode as a function of ft_fraction.
  - Points show mean ± stderr (std / sqrt(n_seeds)).
  - Y-axis range is shared across all subplots within a figure so methods are
    directly comparable.
  - Y-axis label carries a ↑ / ↓ arrow to indicate optimisation direction.

Usage
-----
    python scripts/plotting/plot_results.py
    python scripts/plotting/plot_results.py --input path/to/aggregated_results.csv
    python scripts/plotting/plot_results.py --output-dir my_figures/
    python scripts/plotting/plot_results.py --fmt pdf
    python scripts/plotting/plot_results.py --no-curves   # master grid only

Master grid writes ``all_datasets.{fmt}``, plus ``all_datasets_probe`` and
``all_datasets_full`` (probe head vs full fine-tune only).
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

from scripts.plotting.shared_baseline import apply_shared_random_init_baseline

# ── Paths ─────────────────────────────────────────────────────────────────────
_OUTPUTS_BASE     = _SCRIPT_DIR / "outputs"
MODELS            = ("gin", "gpse_backbone")
DEFAULT_INPUT      = None   # resolved at runtime from --model
DEFAULT_OUTPUT_DIR = None   # resolved at runtime from --model

# ── Column names ──────────────────────────────────────────────────────────────
COL_DATASET    = "pretrained_config_dataset.loader.parameters.data_name"
COL_PRETRAIN   = "pretrained_config_pretraining.task"
COL_FT_MODE    = "ft_mode"
COL_FRACTION   = "ft_fraction"
COL_MON_COL    = "monitor_test_column"   # e.g. "test/accuracy"
COL_MON_METRIC = "monitor_metric"        # e.g. "accuracy"
COL_MON_MODE   = "monitor_mode"          # "max" | "min"
COL_SEED_COUNT = "seed_count"

# ── Display order / visual encoding ──────────────────────────────────────────
MODE_ORDER = [
    "finetune-full",
    "finetune-probe",
    "random-init-full",
    "random-init-probe",
]

# Two semantic groups encoded as colour
GROUP_COLORS: dict[str, str] = {
    "pretrained":   "#0077BB",   # blue  – anything starting with "finetune-"
    "from_scratch": "#EE7733",   # orange – anything starting with "random-init-"
}
GROUP_LABELS: dict[str, str] = {
    "pretrained":   "Pretrained",
    "from_scratch": "From scratch",
}

# Probe vs full encoded as line style; probe = solid, full = dashed
TUNE_LINESTYLES: dict[str, str] = {
    "probe": "-",
    "full":  "--",
}
TUNE_LABELS: dict[str, str] = {
    "probe": "Probe head",
    "full":  "Full fine-tune",
}

# Shared marker per group (consistent across subplots)
GROUP_MARKERS: dict[str, str] = {
    "pretrained":   "o",
    "from_scratch": "s",
}

# ── Combined GIN + GPSE colour palette ───────────────────────────────────────
# GIN   → lighter shades; GPSE → darker shades of the same hue families.
_MODEL_GROUP_COLORS: dict[str, dict[str, str]] = {
    "gin": {
        "pretrained":   "#6AACE8",   # light blue
        "from_scratch": "#F5B87A",   # light orange
    },
    "gpse_backbone": {
        "pretrained":   "#004C88",   # dark blue
        "from_scratch": "#BB5500",   # dark orange
    },
}
MODEL_DISPLAY: dict[str, str] = {
    "gin":           "GIN",
    "gpse_backbone": "GPSE",
}


def _combined_mode_color(mode: str, model: str) -> str:
    """Return the colour for a (ft_mode, model) pair in combined plots."""
    palette = _MODEL_GROUP_COLORS.get(model, _MODEL_GROUP_COLORS["gpse_backbone"])
    if mode.startswith("finetune"):
        return palette["pretrained"]
    return palette["from_scratch"]


def _mode_color(mode: str) -> str:
    if mode.startswith("finetune"):
        return GROUP_COLORS["pretrained"]
    return GROUP_COLORS["from_scratch"]


def _mode_linestyle(mode: str) -> str:
    if "probe" in mode:
        return TUNE_LINESTYLES["probe"]
    return TUNE_LINESTYLES["full"]


def _mode_marker(mode: str) -> str:
    if mode.startswith("finetune"):
        return GROUP_MARKERS["pretrained"]
    return GROUP_MARKERS["from_scratch"]


def _filter_modes_by_scope(modes: list[str], scope: str | None) -> list[str]:
    """Keep only probe-head or full fine-tune ``ft_mode`` values when *scope* is set."""
    if scope is None:
        return modes
    if scope == "probe":
        return [m for m in modes if "probe" in m]
    if scope == "full":
        return [m for m in modes if "probe" not in m]
    raise ValueError(f"scope must be 'probe' or 'full', got {scope!r}")

PRETRAIN_DISPLAY: dict[str, str] = {
    "graphmaev2": "GraphMAEv2",
    "graphcl":    "GraphCL",
    "dgi":        "DGI",
    "vgae":       "VGAE",
    "bgrl":       "BGRL",
    "grace":      "GRACE",
    "mvgrl":      "MVGRL",
    "dgmae":      "DGMAE",
}

# Whether the evaluation set uses scaffold splitting (True) or random splitting (False).
# ADME benchmark datasets and ogbg-molhiv are scaffold-split; the rest are random.
SCAFFOLD_SPLIT_DATASETS: dict[str, bool] = {
    "BBB_Martins":             True,   # ADME benchmark
    "Caco2_Wang":              True,   # ADME benchmark
    "Clearance_Hepatocyte_AZ": True,   # ADME benchmark
    "CYP3A4_Veith":            True,   # ADME benchmark
    "ogbg-molhiv":             True,   # OGB scaffold split
    "IMDB-BINARY":             False,
    "REDDIT-BINARY":           False,
    "MUTAG":                   False,
    "ogbg-molbace":            False,
    "PROTEINS":                False,
}

# Dataset domain category, encoded as a float for scatter-plot x-axes.
# 0 = Molecular  ·  1 = Social / Platform  ·  2 = Protein structure
DATASET_DOMAIN: dict[str, float] = {
    "BBB_Martins":             0.0,   # molecular
    "Caco2_Wang":              0.0,   # molecular
    "Clearance_Hepatocyte_AZ": 0.0,   # molecular
    "CYP3A4_Veith":            0.0,   # molecular
    "ogbg-molhiv":             0.0,   # molecular
    "ogbg-molbace":            0.0,   # molecular
    "MUTAG":                   0.0,   # molecular
    "IMDB-BINARY":             1.0,   # social / platform
    "REDDIT-BINARY":           1.0,   # social / platform
    "PROTEINS":                2.0,   # protein structure
}

# ── Matplotlib global style ───────────────────────────────────────────────────
RC_PARAMS: dict = {
    "font.family":        "sans-serif",
    "font.size":          11,
    "axes.titlesize":     12,
    "axes.labelsize":     11,
    "xtick.labelsize":    10,
    "ytick.labelsize":    10,
    "legend.fontsize":    10,
    "legend.framealpha":  0.85,
    "legend.edgecolor":   "0.75",
    "figure.dpi":         150,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.color":         "0.87",
    "grid.linewidth":     0.7,
    "lines.linewidth":    2.0,
    "lines.markersize":   6,
    "errorbar.capsize":   3.5,
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _mean_col(monitor_test_col: str) -> str:
    return f"{monitor_test_col}_mean"


def _std_col(monitor_test_col: str) -> str:
    return f"{monitor_test_col}_std"


def _pretrain_label(task: str) -> str:
    return PRETRAIN_DISPLAY.get(str(task), str(task))


def _direction_arrow(monitor_mode: str) -> str:
    return "↑" if str(monitor_mode).lower() != "min" else "↓"


def _metric_ylabel(metric: str, monitor_mode: str) -> str:
    arrow = _direction_arrow(monitor_mode)
    return f"{metric.capitalize()}  {arrow}"


def _compute_stderr(df_group: pd.DataFrame, std_col: str) -> pd.Series:
    """stderr = std / sqrt(n_seeds)."""
    n = df_group[COL_SEED_COUNT] if COL_SEED_COUNT in df_group.columns else 3
    return df_group[std_col] / np.sqrt(n)


def _y_limits(
    sub_df: pd.DataFrame,
    mean_col: str,
    std_col: str,
    pad_frac: float = 0.08,
) -> tuple[float, float]:
    """Compute y-axis limits across all rows, with padding."""
    if mean_col not in sub_df.columns:
        return (0.0, 1.0)
    n = sub_df[COL_SEED_COUNT] if COL_SEED_COUNT in sub_df.columns else 3
    stderr = sub_df[std_col] / np.sqrt(n) if std_col in sub_df.columns else 0.0
    lo = (sub_df[mean_col] - stderr).min()
    hi = (sub_df[mean_col] + stderr).max()
    rng = hi - lo if hi > lo else 0.1
    return (lo - pad_frac * rng, hi + pad_frac * rng)


# ── Core plotting ─────────────────────────────────────────────────────────────

def plot_dataset(
    df: pd.DataFrame,
    dataset: str,
    output_dir: Path,
    fmt: str = "png",
) -> Path:
    """Create and save the figure for one dataset. Returns the saved path."""
    sub = df[df[COL_DATASET] == dataset].copy()
    pretrain_methods = sorted(sub[COL_PRETRAIN].dropna().unique())
    n_methods = len(pretrain_methods)
    if n_methods == 0:
        raise ValueError(f"No pretraining methods found for dataset {dataset!r}")

    # Resolve monitor info from the first available row (consistent per dataset)
    monitor_test_col: str | None = None
    monitor_metric: str = "metric"
    monitor_mode: str = "max"
    if COL_MON_COL in sub.columns:
        vals = sub[COL_MON_COL].dropna()
        monitor_test_col = vals.iloc[0] if not vals.empty else None
    if COL_MON_METRIC in sub.columns:
        vals = sub[COL_MON_METRIC].dropna()
        monitor_metric = vals.iloc[0] if not vals.empty else "metric"
    if COL_MON_MODE in sub.columns:
        vals = sub[COL_MON_MODE].dropna()
        monitor_mode = vals.iloc[0] if not vals.empty else "max"

    if monitor_test_col is None:
        raise ValueError(f"No monitor_test_column found for dataset {dataset!r}")

    mc = _mean_col(monitor_test_col)
    sc = _std_col(monitor_test_col)
    if mc not in sub.columns:
        raise ValueError(f"Column {mc!r} not found for dataset {dataset!r}")

    # ── Figure layout: one row, wrap to new row only when > 6 methods ────────
    n_cols = min(n_methods, 6)
    n_rows = math.ceil(n_methods / n_cols)
    fig_w = max(3.6 * n_cols, 5.0)
    fig_h = 3.8 * n_rows + 0.6   # extra room for suptitle

    with mpl.rc_context(RC_PARAMS):
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(fig_w, fig_h),
            sharey=True,          # identical y-axis across all subplots
            squeeze=False,
        )

        # Collect global y-limits first so sharey is set correctly
        y_lo, y_hi = _y_limits(sub, mc, sc)

        # Modes actually present (preserve display order)
        modes_present = [m for m in MODE_ORDER if m in sub[COL_FT_MODE].unique()]
        # Fallback: include any mode not in MODE_ORDER
        extra_modes = [m for m in sub[COL_FT_MODE].unique() if m not in MODE_ORDER]
        modes_present += extra_modes

        # ── One subplot per pretraining method ───────────────────────────────
        for idx, task in enumerate(pretrain_methods):
            row_i, col_i = divmod(idx, n_cols)
            ax = axes[row_i][col_i]

            task_df = sub[sub[COL_PRETRAIN] == task]
            fractions_all = sorted(task_df[COL_FRACTION].dropna().unique())

            for mode in modes_present:
                mode_df = task_df[task_df[COL_FT_MODE] == mode].copy()
                if mode_df.empty:
                    continue
                mode_df = mode_df.sort_values(COL_FRACTION)

                x = mode_df[COL_FRACTION].values
                y = mode_df[mc].values
                yerr = _compute_stderr(mode_df, sc).values

                ax.errorbar(
                    x, y,
                    yerr=yerr,
                    color=_mode_color(mode),
                    marker=_mode_marker(mode),
                    linestyle=_mode_linestyle(mode),
                    linewidth=RC_PARAMS["lines.linewidth"],
                    markersize=RC_PARAMS["lines.markersize"],
                    capsize=RC_PARAMS["errorbar.capsize"],
                    capthick=1.2,
                    elinewidth=1.2,
                    zorder=3,
                )

            ax.set_title(_pretrain_label(task), fontweight="bold")
            ax.set_xlabel("Data fraction")
            ax.set_xlim(-0.02, max(fractions_all) + 0.05 if fractions_all else 1.1)

            # x-tick at every fraction value
            if fractions_all:
                ax.set_xticks(fractions_all)
                ax.xaxis.set_major_formatter(
                    matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}")
                )

            ax.set_ylim(y_lo, y_hi)

            # y-label only on leftmost column
            if col_i == 0:
                ax.set_ylabel(_metric_ylabel(str(monitor_metric), str(monitor_mode)))

        # ── Hide unused axes (when n_methods < n_rows * n_cols) ──────────────
        for empty_idx in range(n_methods, n_rows * n_cols):
            row_i, col_i = divmod(empty_idx, n_cols)
            axes[row_i][col_i].set_visible(False)

        # ── Compound legend: colour = init strategy, linestyle = tune scope ──
        from matplotlib.lines import Line2D

        # Determine which groups / tune-scopes are actually present
        groups_present = []
        for gkey, gstart in [("pretrained", "finetune"), ("from_scratch", "random-init")]:
            if any(m.startswith(gstart) for m in modes_present):
                groups_present.append(gkey)
        scopes_present = []
        for skey in ("probe", "full"):
            if any(skey in m for m in modes_present):
                scopes_present.append(skey)

        color_handles = [
            Line2D(
                [0], [0],
                color=GROUP_COLORS[g],
                marker=GROUP_MARKERS[g],
                linestyle="-",
                linewidth=RC_PARAMS["lines.linewidth"],
                markersize=RC_PARAMS["lines.markersize"],
                label=GROUP_LABELS[g],
            )
            for g in groups_present
        ]
        style_handles = [
            Line2D(
                [0], [0],
                color="0.35",
                linestyle=TUNE_LINESTYLES[s],
                linewidth=RC_PARAMS["lines.linewidth"],
                label=TUNE_LABELS[s],
            )
            for s in scopes_present
        ]

        all_handles = color_handles + style_handles
        if all_handles:
            fig.legend(
                handles=all_handles,
                loc="lower center",
                ncol=len(all_handles),
                frameon=True,
                fontsize=RC_PARAMS["legend.fontsize"],
                bbox_to_anchor=(0.5, 0.0),
            )
            fig.tight_layout(rect=(0, 0.10, 1, 0.95))
        else:
            fig.tight_layout(rect=(0, 0, 1, 0.95))

        fig.suptitle(
            f"Dataset: {dataset}",
            fontsize=13,
            fontweight="bold",
            y=0.99,
        )

        # ── Save ──────────────────────────────────────────────────────────────
        output_dir.mkdir(parents=True, exist_ok=True)
        safe_name = dataset.replace("/", "_").replace(" ", "_")
        out_path = output_dir / f"{safe_name}.{fmt}"
        fig.savefig(out_path, bbox_inches="tight", dpi=200 if fmt == "png" else None)
        plt.close(fig)
        print(f"  saved → {out_path}")
        return out_path


# ── Master "all datasets" learning-curve grid ────────────────────────────────

def plot_all_datasets(
    df: pd.DataFrame,
    output_dir: Path,
    fmt: str = "png",
    datasets: list[str] | None = None,
    scope: str | None = None,
) -> Path:
    """One big grid figure with every dataset in a separate row.

    Columns = union of all pretraining methods (sorted).
    Rows    = datasets (one per dataset).
    Y-axis  is shared *within* each row (so all methods for a dataset are
             directly comparable) but *not* across rows (different metrics /
             scales per dataset).
    A single shared legend lives at the bottom of the figure.

    Parameters
    ----------
    scope : ``None`` | ``"probe"`` | ``"full"``
        When set, only plot ``finetune-probe`` / ``random-init-probe`` or the
        corresponding ``*-full`` modes (one curve per init strategy per cell).
    """
    _datasets = datasets or sorted(df[COL_DATASET].dropna().unique())
    _datasets  = [d for d in _datasets if not df[df[COL_DATASET] == d].empty]
    if not _datasets:
        raise ValueError("No datasets to plot in master figure.")

    # ── Global method list (defines columns) ─────────────────────────────────
    all_methods: list[str] = sorted(df[COL_PRETRAIN].dropna().unique())
    n_cols = len(all_methods)
    n_rows = len(_datasets)

    cell_w = 3.2
    cell_h = 2.8
    fig_w  = max(cell_w * n_cols + 0.8, 8.0)
    fig_h  = cell_h * n_rows + 1.2   # extra room for legend

    with mpl.rc_context(RC_PARAMS):
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(fig_w, fig_h),
            sharey="row",   # shared y within each dataset row
            sharex=True,    # shared x across all cells
            squeeze=False,
        )

        # ── Column headers (pretraining method names) ─────────────────────────
        for ci, method in enumerate(all_methods):
            axes[0][ci].set_title(_pretrain_label(method),
                                  fontweight="bold", fontsize=8, pad=4)

        # ── One row per dataset ───────────────────────────────────────────────
        modes_union: set[str] = set()

        for ri, dataset in enumerate(_datasets):
            sub = df[df[COL_DATASET] == dataset].copy()

            # Resolve monitor info for this dataset
            monitor_test_col: str | None = None
            monitor_metric: str = "metric"
            monitor_mode: str = "max"
            if COL_MON_COL in sub.columns:
                v = sub[COL_MON_COL].dropna()
                monitor_test_col = v.iloc[0] if not v.empty else None
            if COL_MON_METRIC in sub.columns:
                v = sub[COL_MON_METRIC].dropna()
                monitor_metric = v.iloc[0] if not v.empty else "metric"
            if COL_MON_MODE in sub.columns:
                v = sub[COL_MON_MODE].dropna()
                monitor_mode = v.iloc[0] if not v.empty else "max"

            if monitor_test_col is None:
                for ci in range(n_cols):
                    axes[ri][ci].set_visible(False)
                continue

            mc = _mean_col(monitor_test_col)
            sc = _std_col(monitor_test_col)
            if mc not in sub.columns:
                for ci in range(n_cols):
                    axes[ri][ci].set_visible(False)
                continue

            y_lo, y_hi = _y_limits(sub, mc, sc)

            modes_present = [m for m in MODE_ORDER if m in sub[COL_FT_MODE].unique()]
            modes_present += [m for m in sub[COL_FT_MODE].unique() if m not in MODE_ORDER]
            modes_present = _filter_modes_by_scope(modes_present, scope)
            modes_union.update(modes_present)

            # ── Row label: dataset name + y-axis metric ───────────────────────
            ylabel_txt = _metric_ylabel(str(monitor_metric), str(monitor_mode))
            axes[ri][0].set_ylabel(ylabel_txt, fontsize=7)

            # Dataset name as a text annotation on the far left
            axes[ri][0].annotate(
                dataset,
                xy=(-0.55, 0.5), xycoords="axes fraction",
                fontsize=7, fontweight="bold",
                ha="right", va="center",
                rotation=0,
                annotation_clip=False,
            )

            # ── Fill each column ──────────────────────────────────────────────
            for ci, method in enumerate(all_methods):
                ax = axes[ri][ci]
                task_df = sub[sub[COL_PRETRAIN] == method]

                if task_df.empty:
                    ax.set_visible(False)
                    continue

                fractions_all = sorted(task_df[COL_FRACTION].dropna().unique())

                for mode in modes_present:
                    mode_df = task_df[task_df[COL_FT_MODE] == mode].sort_values(COL_FRACTION)
                    if mode_df.empty:
                        continue
                    x    = mode_df[COL_FRACTION].values
                    y    = mode_df[mc].values
                    yerr = _compute_stderr(mode_df, sc).values
                    ax.errorbar(
                        x, y, yerr=yerr,
                        color=_mode_color(mode),
                        marker=_mode_marker(mode),
                        linestyle=_mode_linestyle(mode),
                        linewidth=RC_PARAMS["lines.linewidth"],
                        markersize=RC_PARAMS["lines.markersize"] * 0.8,
                        capsize=2, capthick=0.9, elinewidth=0.9,
                        zorder=3,
                    )

                ax.set_ylim(y_lo, y_hi)
                if fractions_all:
                    ax.set_xticks(fractions_all)
                    ax.xaxis.set_major_formatter(
                        matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}")
                    )
                ax.set_xlim(-0.02, max(fractions_all) + 0.05 if fractions_all else 1.1)
                ax.tick_params(labelsize=7)

                # x-label only on the last row
                if ri == n_rows - 1:
                    ax.set_xlabel("Data fraction", fontsize=7)

        # ── Shared legend (same logic as plot_dataset) ────────────────────────
        from matplotlib.lines import Line2D

        all_modes = [m for m in MODE_ORDER if m in modes_union]
        all_modes += [m for m in modes_union if m not in MODE_ORDER]

        groups_present, scopes_present = [], []
        for gkey, gstart in [("pretrained", "finetune"), ("from_scratch", "random-init")]:
            if any(m.startswith(gstart) for m in all_modes):
                groups_present.append(gkey)
        if scope is None:
            for skey in ("probe", "full"):
                if any(skey in m for m in all_modes):
                    scopes_present.append(skey)

        color_handles = [
            Line2D([0], [0], color=GROUP_COLORS[g], marker=GROUP_MARKERS[g],
                   linestyle="-", linewidth=RC_PARAMS["lines.linewidth"],
                   markersize=RC_PARAMS["lines.markersize"], label=GROUP_LABELS[g])
            for g in groups_present
        ]
        style_handles = []
        if scope is None:
            style_handles = [
                Line2D([0], [0], color="0.35", linestyle=TUNE_LINESTYLES[s],
                       linewidth=RC_PARAMS["lines.linewidth"], label=TUNE_LABELS[s])
                for s in scopes_present
            ]
        all_handles = color_handles + style_handles
        if all_handles:
            fig.legend(handles=all_handles, loc="lower center",
                       ncol=len(all_handles), frameon=True,
                       fontsize=RC_PARAMS["legend.fontsize"],
                       bbox_to_anchor=(0.5, 0.0))
            fig.tight_layout(rect=(0, 0.06, 1, 0.97))
        else:
            fig.tight_layout(rect=(0, 0, 1, 0.97))

        title = "Learning curves — all datasets"
        if scope == "probe":
            title += " (probe head)"
        elif scope == "full":
            title += " (full fine-tune)"
        fig.suptitle(title, fontsize=12, fontweight="bold", y=0.995)

        output_dir.mkdir(parents=True, exist_ok=True)
        stem = "all_datasets" if scope is None else f"all_datasets_{scope}"
        out_path = output_dir / f"{stem}.{fmt}"
        fig.savefig(out_path, bbox_inches="tight",
                    dpi=200 if fmt == "png" else None)
        plt.close(fig)
        print(f"  saved → {out_path}")
        return out_path


def plot_all_datasets_percentage(
    df: pd.DataFrame,
    output_dir: Path,
    fmt: str = "png",
    datasets: list[str] | None = None,
    scope: str | None = None,
) -> Path | None:
    """Same grid layout as plot_all_datasets but y-axis rescaled to [0, 100]%.

    100 % = best observed mean result for each dataset (across all settings).
    0   % = worst observed mean result.
    Uncertainties are propagated linearly.  Y-axis is shared within each row
    (all methods for one dataset use the same % scale) but not across rows.

    Parameters
    ----------
    scope : ``None`` | ``"probe"`` | ``"full"``
        When set, only plot probe-head or full fine-tune modes.
    """
    from matplotlib.lines import Line2D

    _datasets = datasets or sorted(df[COL_DATASET].dropna().unique())
    _datasets  = [d for d in _datasets if not df[df[COL_DATASET] == d].empty]
    if not _datasets:
        return None

    all_methods: list[str] = sorted(df[COL_PRETRAIN].dropna().unique())
    n_cols = len(all_methods)
    n_rows = len(_datasets)

    cell_w = 3.2
    cell_h = 2.8
    fig_w  = max(cell_w * n_cols + 0.8, 8.0)
    fig_h  = cell_h * n_rows + 1.2

    with mpl.rc_context(RC_PARAMS):
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(fig_w, fig_h),
            sharey="row",
            sharex=True,
            squeeze=False,
        )

        for ci, method in enumerate(all_methods):
            axes[0][ci].set_title(_pretrain_label(method),
                                  fontweight="bold", fontsize=8, pad=4)

        modes_union: set[str] = set()

        for ri, dataset in enumerate(_datasets):
            sub = df[df[COL_DATASET] == dataset].copy()

            # Resolve normalisation params for this dataset
            norm = _dataset_norm_params(df, dataset)
            if norm is None:
                for ci in range(n_cols):
                    axes[ri][ci].set_visible(False)
                continue
            best_val, worst_val, mc, monitor_mode = norm
            sc = mc.replace("_mean", "_std")

            monitor_metric = (
                sub[COL_MON_METRIC].dropna().iloc[0]
                if COL_MON_METRIC in sub.columns and not sub[COL_MON_METRIC].dropna().empty
                else "metric"
            )

            modes_present = [m for m in MODE_ORDER if m in sub[COL_FT_MODE].unique()]
            modes_present += [m for m in sub[COL_FT_MODE].unique() if m not in MODE_ORDER]
            modes_present = _filter_modes_by_scope(modes_present, scope)
            modes_union.update(modes_present)

            # Compute per-row % y-limits (from all data for this dataset)
            pct_vals, pct_errs = [], []
            for _, row in sub.iterrows():
                v = row.get(mc)
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    pct_vals.append(_to_pct(float(v), best_val, worst_val))
                if sc in sub.columns:
                    s = row.get(sc)
                    n_s = row.get(COL_SEED_COUNT, 3) or 3
                    if s is not None and not (isinstance(s, float) and np.isnan(s)):
                        pct_errs.append(
                            _to_pct_stderr(float(s) / np.sqrt(n_s), best_val, worst_val)
                        )
            pad = 5.0
            lo  = (min(pct_vals) - (max(pct_errs) if pct_errs else 0)) if pct_vals else 0.0
            hi  = (max(pct_vals) + (max(pct_errs) if pct_errs else 0)) if pct_vals else 100.0
            y_lo = max(0.0,   lo - pad)
            y_hi = min(100.0, hi + pad)

            axes[ri][0].set_ylabel("Performance (%)", fontsize=7)
            axes[ri][0].annotate(
                dataset,
                xy=(-0.55, 0.5), xycoords="axes fraction",
                fontsize=7, fontweight="bold",
                ha="right", va="center",
                rotation=0,
                annotation_clip=False,
            )

            for ci, method in enumerate(all_methods):
                ax = axes[ri][ci]
                task_df = sub[sub[COL_PRETRAIN] == method]

                if task_df.empty:
                    ax.set_visible(False)
                    continue

                fractions_all = sorted(task_df[COL_FRACTION].dropna().unique())

                for mode in modes_present:
                    mode_df = task_df[task_df[COL_FT_MODE] == mode].sort_values(COL_FRACTION)
                    if mode_df.empty or mc not in mode_df.columns:
                        continue
                    x = mode_df[COL_FRACTION].values
                    y = np.array([_to_pct(float(v), best_val, worst_val)
                                  for v in mode_df[mc].values])
                    if sc in mode_df.columns:
                        n_s  = (mode_df[COL_SEED_COUNT].values
                                if COL_SEED_COUNT in mode_df.columns
                                else np.full(len(x), 3))
                        yerr = np.array([
                            _to_pct_stderr(float(s) / np.sqrt(max(n, 1)), best_val, worst_val)
                            for s, n in zip(mode_df[sc].values, n_s)
                        ])
                    else:
                        yerr = np.zeros(len(x))

                    ax.errorbar(
                        x, y, yerr=yerr,
                        color=_mode_color(mode),
                        marker=_mode_marker(mode),
                        linestyle=_mode_linestyle(mode),
                        linewidth=RC_PARAMS["lines.linewidth"],
                        markersize=RC_PARAMS["lines.markersize"] * 0.8,
                        capsize=2, capthick=0.9, elinewidth=0.9,
                        zorder=3,
                    )

                ax.set_ylim(y_lo, y_hi)
                if fractions_all:
                    ax.set_xticks(fractions_all)
                    ax.xaxis.set_major_formatter(
                        matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}")
                    )
                ax.set_xlim(-0.02, max(fractions_all) + 0.05 if fractions_all else 1.1)
                ax.tick_params(labelsize=7)

                if ri == n_rows - 1:
                    ax.set_xlabel("Data fraction", fontsize=7)

        # ── Shared legend ─────────────────────────────────────────────────────
        all_modes = [m for m in MODE_ORDER if m in modes_union]
        all_modes += [m for m in modes_union if m not in MODE_ORDER]

        groups_present, scopes_present = [], []
        for gkey, gstart in [("pretrained", "finetune"), ("from_scratch", "random-init")]:
            if any(m.startswith(gstart) for m in all_modes):
                groups_present.append(gkey)
        if scope is None:
            for skey in ("probe", "full"):
                if any(skey in m for m in all_modes):
                    scopes_present.append(skey)

        color_handles = [
            Line2D([0], [0], color=GROUP_COLORS[g], marker=GROUP_MARKERS[g],
                   linestyle="-", linewidth=RC_PARAMS["lines.linewidth"],
                   markersize=RC_PARAMS["lines.markersize"], label=GROUP_LABELS[g])
            for g in groups_present
        ]
        style_handles = []
        if scope is None:
            style_handles = [
                Line2D([0], [0], color="0.35", linestyle=TUNE_LINESTYLES[s],
                       linewidth=RC_PARAMS["lines.linewidth"], label=TUNE_LABELS[s])
                for s in scopes_present
            ]
        all_handles = color_handles + style_handles
        if all_handles:
            fig.legend(handles=all_handles, loc="lower center",
                       ncol=len(all_handles), frameon=True,
                       fontsize=RC_PARAMS["legend.fontsize"],
                       bbox_to_anchor=(0.5, 0.0))
            fig.tight_layout(rect=(0, 0.06, 1, 0.97))
        else:
            fig.tight_layout(rect=(0, 0, 1, 0.97))

        title = "Learning curves (%) — all datasets  ·  100 % = best, 0 % = worst"
        if scope == "probe":
            title += " (probe head)"
        elif scope == "full":
            title += " (full fine-tune)"
        fig.suptitle(title, fontsize=12, fontweight="bold", y=0.995)

        output_dir.mkdir(parents=True, exist_ok=True)
        stem = "all_datasets_pct" if scope is None else f"all_datasets_pct_{scope}"
        out_path = output_dir / f"{stem}.{fmt}"
        fig.savefig(out_path, bbox_inches="tight",
                    dpi=200 if fmt == "png" else None)
        plt.close(fig)
        print(f"  saved → {out_path}")
        return out_path


# ── Percentage-normalisation helpers ─────────────────────────────────────────

METADATA_DIR_DEFAULT = _SCRIPT_DIR.parent / "metadata_analysis" / "outputs" / "metadata_analysis"

# Graph properties extracted from the train-split metadata JSONs.
# Each entry: (json_key, display_label)
# Each structural property is represented twice: once by its per-graph mean
# (how large/connected the graphs are on average) and once by its per-graph std
# (how *variable* graphs in the dataset are for that property).
# The two are kept adjacent so property groups in the 6-per-plot layout stay
# coherent: group 1 = nodes+edges (mean & std), group 2 = density+degree …
GRAPH_PROPERTIES: list[tuple[str, str]] = [
    # ── node / edge counts ─────────────────────────────────────────────────
    ("num_nodes_mean",                "Nodes (mean)"),
    ("num_nodes_std",                 "Nodes (std)"),
    ("num_edges_mean",                "Edges (mean)"),
    ("num_edges_std",                 "Edges (std)"),
    # ── density & degree ───────────────────────────────────────────────────
    ("edge_density_mean",             "Edge density (mean)"),
    ("edge_density_std",              "Edge density (std)"),
    ("avg_degree_mean",               "Avg degree (mean)"),
    ("avg_degree_std",                "Avg degree (std)"),
    ("degree_assortativity_mean",     "Deg. assortativity (mean)"),
    ("degree_assortativity_std",      "Deg. assortativity (std)"),
    ("gini_degree_mean",              "Gini degree (mean)"),
    ("gini_degree_std",               "Gini degree (std)"),
    # ── topology / path structure ──────────────────────────────────────────
    ("pseudo_diameter_mean",          "Pseudo-diameter (mean)"),
    ("pseudo_diameter_std",           "Pseudo-diameter (std)"),
    ("avg_clustering_coef_mean",      "Clustering coef (mean)"),
    ("avg_clustering_coef_std",       "Clustering coef (std)"),
    ("transitivity_mean",             "Transitivity (mean)"),
    ("transitivity_std",              "Transitivity (std)"),
    ("degeneracy_mean",               "Degeneracy (mean)"),
    ("degeneracy_std",                "Degeneracy (std)"),
    # ── spectral ───────────────────────────────────────────────────────────
    ("spectral_gap_mean",             "Spectral gap (mean)"),
    ("spectral_gap_std",              "Spectral gap (std)"),
    ("spectral_radius_mean",          "Spectral radius (mean)"),
    ("spectral_radius_std",           "Spectral radius (std)"),
    ("num_connected_components_mean", "Components (mean)"),
    ("num_connected_components_std",  "Components (std)"),
    # ── dataset-level metadata ─────────────────────────────────────────────
    ("scaffold_split",                "Scaffold split (1=yes, 0=no)"),
    ("dataset_domain",                "Domain (0=mol, 1=social, 2=protein)"),
]
PROPS_PER_PLOT = 6   # 2 rows × 3 cols

# Colors per pretraining method for scatter plots
PRETRAIN_SCATTER_COLORS: dict[str, str] = {
    "graphmaev2": "#2196F3",
    "graphcl":    "#4CAF50",
    "dgi":        "#FF9800",
    "vgae":       "#E91E63",
    "bgrl":       "#9C27B0",
    "grace":      "#00BCD4",
    "mvgrl":      "#795548",
    "dgmae":      "#607D8B",
}
# Marker per dataset for scatter plots (cycles if more datasets than markers)
_SCATTER_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "<", ">"]


def _dataset_norm_params(
    df: pd.DataFrame,
    dataset: str,
) -> tuple[float, float, str, str] | None:
    """Return (best_val, worst_val, mean_col, monitor_mode) for one dataset.

    best_val  = the best  observed mean-metric value across ALL rows for this dataset
    worst_val = the worst observed mean-metric value across ALL rows
    100% ↔ best_val,  0% ↔ worst_val  regardless of optimisation direction.
    """
    sub = df[df[COL_DATASET] == dataset]
    if sub.empty:
        return None
    monitor_mode = (
        sub[COL_MON_MODE].dropna().iloc[0]
        if COL_MON_MODE in sub.columns and not sub[COL_MON_MODE].dropna().empty
        else "max"
    )
    monitor_test_col = (
        sub[COL_MON_COL].dropna().iloc[0]
        if COL_MON_COL in sub.columns and not sub[COL_MON_COL].dropna().empty
        else None
    )
    if monitor_test_col is None:
        return None
    mc = _mean_col(monitor_test_col)
    if mc not in sub.columns:
        return None
    vals = sub[mc].dropna()
    if vals.empty:
        return None
    if str(monitor_mode).lower() == "min":
        best_val  = float(vals.min())
        worst_val = float(vals.max())
    else:
        best_val  = float(vals.max())
        worst_val = float(vals.min())
    return best_val, worst_val, mc, str(monitor_mode)


def _to_pct(val: float, best: float, worst: float) -> float:
    denom = best - worst
    if abs(denom) < 1e-12:
        return 50.0
    return (val - worst) / denom * 100.0


def _to_pct_stderr(stderr: float, best: float, worst: float) -> float:
    """Linear propagation: σ_pct = σ / |best - worst| × 100."""
    denom = abs(best - worst)
    if denom < 1e-12:
        return 0.0
    return stderr / denom * 100.0


# ── Percentage-normalised learning-curve plots ────────────────────────────────

def plot_dataset_percentage(
    df: pd.DataFrame,
    dataset: str,
    output_dir: Path,
    fmt: str = "png",
) -> Path | None:
    """Same structure as plot_dataset but y-axis is rescaled to [0, 100]%.

    100 % = best observed mean result for this dataset (across all settings).
    0   % = worst observed mean result.
    Uncertainties are propagated linearly through the rescaling.
    """
    from matplotlib.lines import Line2D

    norm = _dataset_norm_params(df, dataset)
    if norm is None:
        print(f"  skipping {dataset!r}: cannot compute normalisation")
        return None
    best_val, worst_val, mc, monitor_mode = norm
    sc = _std_col(mc.replace("_mean", "").lstrip("test/") if "_mean" in mc else mc)
    # resolve std col name from mean col: test/accuracy_mean → test/accuracy_std
    sc = mc.replace("_mean", "_std")

    sub = df[df[COL_DATASET] == dataset].copy()
    pretrain_methods = sorted(sub[COL_PRETRAIN].dropna().unique())
    n_methods = len(pretrain_methods)
    if n_methods == 0:
        return None

    monitor_metric = (
        sub[COL_MON_METRIC].dropna().iloc[0] if COL_MON_METRIC in sub.columns else "metric"
    )

    n_cols = min(n_methods, 6)
    n_rows = math.ceil(n_methods / n_cols)
    fig_w  = max(3.6 * n_cols, 5.0)
    fig_h  = 3.8 * n_rows + 0.6

    modes_present = [m for m in MODE_ORDER if m in sub[COL_FT_MODE].unique()]
    modes_present += [m for m in sub[COL_FT_MODE].unique() if m not in MODE_ORDER]

    # Global pct y-limits
    pct_vals, pct_errs = [], []
    if mc in sub.columns:
        for _, row in sub.iterrows():
            v = row.get(mc)
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                pct_vals.append(_to_pct(float(v), best_val, worst_val))
            if sc in sub.columns:
                s = row.get(sc)
                n_s = row.get(COL_SEED_COUNT, 3) or 3
                if s is not None and not (isinstance(s, float) and np.isnan(s)):
                    pct_errs.append(_to_pct_stderr(float(s) / np.sqrt(n_s), best_val, worst_val))
    pad  = 5.0
    lo   = (min(pct_vals) - (max(pct_errs) if pct_errs else 0)) if pct_vals else 0.0
    hi   = (max(pct_vals) + (max(pct_errs) if pct_errs else 0)) if pct_vals else 100.0
    y_lo = max(0.0,   lo - pad)
    y_hi = min(100.0, hi + pad)

    with mpl.rc_context(RC_PARAMS):
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h),
                                 sharey=True, squeeze=False)

        for idx, task in enumerate(pretrain_methods):
            row_i, col_i = divmod(idx, n_cols)
            ax = axes[row_i][col_i]
            task_df = sub[sub[COL_PRETRAIN] == task]
            fracs = sorted(task_df[COL_FRACTION].dropna().unique())

            for mode in modes_present:
                mode_df = task_df[task_df[COL_FT_MODE] == mode].sort_values(COL_FRACTION)
                if mode_df.empty or mc not in mode_df.columns:
                    continue
                x    = mode_df[COL_FRACTION].values
                y    = np.array([_to_pct(float(v), best_val, worst_val)
                                 for v in mode_df[mc].values])
                if sc in mode_df.columns:
                    n_s  = mode_df[COL_SEED_COUNT].values if COL_SEED_COUNT in mode_df.columns else np.full(len(x), 3)
                    yerr = np.array([_to_pct_stderr(float(s) / np.sqrt(max(n, 1)), best_val, worst_val)
                                     for s, n in zip(mode_df[sc].values, n_s)])
                else:
                    yerr = np.zeros(len(x))

                ax.errorbar(x, y, yerr=yerr,
                            color=_mode_color(mode), marker=_mode_marker(mode),
                            linestyle=_mode_linestyle(mode),
                            linewidth=RC_PARAMS["lines.linewidth"],
                            markersize=RC_PARAMS["lines.markersize"],
                            capsize=RC_PARAMS["errorbar.capsize"],
                            capthick=1.2, elinewidth=1.2, zorder=3)

            ax.set_title(_pretrain_label(task), fontweight="bold")
            ax.set_xlabel("Data fraction")
            ax.set_xlim(-0.02, max(fracs) + 0.05 if fracs else 1.1)
            if fracs:
                ax.set_xticks(fracs)
                ax.xaxis.set_major_formatter(
                    matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
            ax.set_ylim(y_lo, y_hi)
            if col_i == 0:
                ax.set_ylabel("Performance (%)")

        for ei in range(n_methods, n_rows * n_cols):
            ri, ci = divmod(ei, n_cols)
            axes[ri][ci].set_visible(False)

        # Compound legend
        groups_present = [g for g, s in [("pretrained", "finetune"), ("from_scratch", "random-init")]
                          if any(m.startswith(s) for m in modes_present)]
        scopes_present = [s for s in ("probe", "full") if any(s in m for m in modes_present)]
        handles = (
            [Line2D([0], [0], color=GROUP_COLORS[g], marker=GROUP_MARKERS[g], linestyle="-",
                    linewidth=RC_PARAMS["lines.linewidth"], markersize=RC_PARAMS["lines.markersize"],
                    label=GROUP_LABELS[g]) for g in groups_present]
            + [Line2D([0], [0], color="0.35", linestyle=TUNE_LINESTYLES[s],
                      linewidth=RC_PARAMS["lines.linewidth"], label=TUNE_LABELS[s])
               for s in scopes_present]
        )
        if handles:
            fig.legend(handles=handles, loc="lower center", ncol=len(handles),
                       frameon=True, fontsize=RC_PARAMS["legend.fontsize"],
                       bbox_to_anchor=(0.5, 0.0))
            fig.tight_layout(rect=(0, 0.10, 1, 0.95))
        else:
            fig.tight_layout(rect=(0, 0, 1, 0.95))

        arrow = _direction_arrow(monitor_mode)
        fig.suptitle(
            f"Dataset: {dataset}  ·  100 % = best {monitor_metric} {arrow},  0 % = worst",
            fontsize=12, fontweight="bold", y=0.99,
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        safe_name = dataset.replace("/", "_").replace(" ", "_")
        out_path  = output_dir / f"{safe_name}.{fmt}"
        fig.savefig(out_path, bbox_inches="tight", dpi=200 if fmt == "png" else None)
        plt.close(fig)
        print(f"  saved → {out_path}")
        return out_path


# ── Property-correlation scatter plots ────────────────────────────────────────

def load_metadata(metadata_dir: Path) -> dict[str, dict]:
    """Load *.json files from metadata_dir. Returns {dataset_name: train_props}.

    Injects ``scaffold_split`` (1.0 / 0.0) from ``SCAFFOLD_SPLIT_DATASETS`` for
    each dataset so it can be used as a correlation variable alongside graph
    structural properties.
    """
    import json
    result: dict[str, dict] = {}
    if not metadata_dir.is_dir():
        return result
    for p in metadata_dir.glob("*.json"):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        train = data.get("train", {})
        props = {k: v for k, v in train.items()
                 if k not in ("num_graphs", "total_num_nodes")}
        props["scaffold_split"] = 1.0 if SCAFFOLD_SPLIT_DATASETS.get(p.stem, False) else 0.0
        props["dataset_domain"] = DATASET_DOMAIN.get(p.stem, 0.0)
        result[p.stem] = props
    return result


def load_metadata_all_splits(metadata_dir: Path) -> dict[str, dict]:
    """Load *.json files, returning all three splits per dataset.

    Returns ``{dataset_name: {"train": {...}, "val": {...}, "test": {...}}}``.
    """
    import json
    result: dict[str, dict] = {}
    if not metadata_dir.is_dir():
        return result
    _skip = {"num_graphs", "total_num_nodes"}
    for p in metadata_dir.glob("*.json"):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        result[p.stem] = {
            split: {k: v for k, v in data.get(split, {}).items() if k not in _skip}
            for split in ("train", "val", "test")
        }
    return result


def _compute_split_shift_metadata(
    metadata_all: dict[str, dict],
) -> tuple[dict[str, dict], dict[str, dict]]:
    """Compute per-property absolute split-mean differences from all-splits metadata.

    Only properties whose key ends with ``_mean`` are included (std-of-property
    and count columns are skipped).  Values are absolute differences so the
    x-axis always represents the magnitude of distribution shift.

    Returns
    -------
    train_val_meta : dict[str, dict]
        ``{dataset: {prop_key: |train_mean - val_mean|}}``.
    val_test_meta : dict[str, dict]
        ``{dataset: {prop_key: |val_mean - test_mean|}}``.
    """
    train_val: dict[str, dict] = {}
    val_test:  dict[str, dict] = {}
    for ds, splits in metadata_all.items():
        train = splits.get("train", {})
        val   = splits.get("val",   {})
        test  = splits.get("test",  {})
        mean_keys = [k for k in set(train) | set(val) | set(test)
                     if k.endswith("_mean")]
        tv, vt = {}, {}
        for key in mean_keys:
            t_v  = train.get(key)
            v_v  = val.get(key)
            te_v = test.get(key)
            if t_v is not None and v_v is not None:
                tv[key] = abs(t_v - v_v)
            if v_v is not None and te_v is not None:
                vt[key] = abs(v_v - te_v)
        if tv:
            train_val[ds] = tv
        if vt:
            val_test[ds] = vt
    return train_val, val_test


def _build_diff_records(
    df: pd.DataFrame,
    datasets: list[str] | None = None,
) -> pd.DataFrame:
    """Build flat records for property-correlation scatter plots.

    For each (dataset, method, fraction, scope):
      diff      = sign · (finetune_mean − random_init_mean)
                  sign = +1 for max-metrics, −1 for min-metrics
                  → positive always means pretraining helped
      norm_diff = diff / σ_dataset
                  σ_dataset = std of ALL metric values for that dataset
                  → dimensionless, comparable across datasets with different
                    metrics, without amplifying noise through min/max scaling
      norm_err  = √(se_ft² + se_ri²) / σ_dataset
                  propagated in quadrature then divided by the same scale

    Using per-dataset std normalisation (rather than min/max range) keeps the
    uncertainty bars proportional to the actual noise level.
    """
    _datasets = datasets or sorted(df[COL_DATASET].dropna().unique())
    records = []

    for ds in _datasets:
        ds_df = df[df[COL_DATASET] == ds]
        if ds_df.empty:
            continue
        monitor_mode = (
            ds_df[COL_MON_MODE].dropna().iloc[0]
            if COL_MON_MODE in ds_df.columns and not ds_df[COL_MON_MODE].dropna().empty
            else "max"
        )
        monitor_test_col = (
            ds_df[COL_MON_COL].dropna().iloc[0]
            if COL_MON_COL in ds_df.columns and not ds_df[COL_MON_COL].dropna().empty
            else None
        )
        if monitor_test_col is None:
            continue
        mc   = _mean_col(monitor_test_col)
        sc   = mc.replace("_mean", "_std")
        sign = -1.0 if str(monitor_mode).lower() == "min" else 1.0

        # σ_dataset: std of all mc values for this dataset (all modes/methods/fractions)
        all_vals = ds_df[mc].dropna() if mc in ds_df.columns else pd.Series(dtype=float)
        sigma_d  = float(all_vals.std(ddof=1)) if len(all_vals) > 1 else 1.0
        if sigma_d < 1e-9:
            sigma_d = 1.0

        for method in ds_df[COL_PRETRAIN].dropna().unique():
            m_df = ds_df[ds_df[COL_PRETRAIN] == method]
            for frac in m_df[COL_FRACTION].dropna().unique():
                f_df = m_df[m_df[COL_FRACTION] == frac]
                for scope in ("probe", "full"):
                    ft_rows = f_df[f_df[COL_FT_MODE] == f"finetune-{scope}"]
                    ri_rows = f_df[f_df[COL_FT_MODE] == f"random-init-{scope}"]
                    if ft_rows.empty or ri_rows.empty or mc not in f_df.columns:
                        continue
                    ft_v = float(ft_rows[mc].iloc[0])
                    ri_v = float(ri_rows[mc].iloc[0])
                    if np.isnan(ft_v) or np.isnan(ri_v):
                        continue

                    diff      = sign * (ft_v - ri_v)
                    norm_diff = diff / sigma_d

                    # Propagate stderr: se = std / sqrt(n)
                    def _se(rows: pd.DataFrame) -> float:
                        if sc not in rows.columns:
                            return 0.0
                        v = float(rows[sc].iloc[0])
                        n = int(rows[COL_SEED_COUNT].iloc[0]) if COL_SEED_COUNT in rows.columns else 3
                        return float(v) / np.sqrt(max(n, 1))

                    norm_err = np.sqrt(_se(ft_rows) ** 2 + _se(ri_rows) ** 2) / sigma_d

                    records.append(dict(
                        dataset=ds, pretrain_method=str(method),
                        fraction=float(frac), scope=scope,
                        diff=diff, norm_diff=norm_diff, norm_err=norm_err,
                    ))
    return pd.DataFrame(records)


def _build_frac_avg_diff_records(diff_df: pd.DataFrame) -> pd.DataFrame:
    """Average norm_diff over fractions for each (dataset, pretrain_method, scope).

    Useful for producing a single correlation point per (dataset, method) pair
    that summarises the pretraining advantage across all data-fraction settings.

    norm_err is propagated in quadrature then divided by K
    (matches the averaging convention used in ``plot_average_improvement``):
        err_avg = sqrt(Σ err_i²) / K
    """
    records = []
    for (ds, method, scope), grp in diff_df.groupby(
        ["dataset", "pretrain_method", "scope"]
    ):
        valid = grp["norm_diff"].notna()
        vals  = grp.loc[valid, "norm_diff"].values
        errs  = grp.loc[valid, "norm_err"].values
        K = len(vals)
        if K == 0:
            continue
        records.append(dict(
            dataset=str(ds),
            pretrain_method=str(method),
            scope=str(scope),
            fraction=np.nan,
            diff=float(grp["diff"].dropna().mean()) if "diff" in grp.columns else np.nan,
            norm_diff=float(vals.mean()),
            norm_err=float(np.sqrt((errs ** 2).sum())) / K,
        ))
    return pd.DataFrame(records)


def plot_property_correlations(
    df: pd.DataFrame,
    metadata: dict[str, dict],
    output_dir: Path,
    fmt: str = "png",
    datasets: list[str] | None = None,
    x_label_suffix: str = "",
    title_extra: str = "",
    fname_prefix: str = "corr",
    pool_fractions: bool = False,
    prebuilt_diff_df: pd.DataFrame | None = None,
) -> None:
    """For each (fraction, scope) and each group of 6 graph properties: one 2×3 figure.

    X-axis : graph property value (or split difference when ``x_label_suffix`` is set).
    Y-axis : sign·(finetune − random_init) / σ_dataset
             [dataset-std-normalised performance difference, ± propagated stderr].
    Each point: one (dataset, pretraining_method) combo.
    Colour  : pretraining method.

    Parameters
    ----------
    x_label_suffix : str
        Appended to each property axis label (e.g. ``" (train−val diff)"``).
    title_extra : str
        Extra text appended to each figure's suptitle.
    fname_prefix : str
        Filename stem prefix; default ``"corr"`` → ``corr_frac…``.
    pool_fractions : bool
        When True, all fractions are combined into a single plot per scope
        (one figure per scope per property group instead of one per fraction).
    prebuilt_diff_df : pd.DataFrame or None
        If provided, skip the internal ``_build_diff_records`` call and use
        this DataFrame directly (e.g. fraction-averaged records).
    """
    _datasets = datasets or sorted(df[COL_DATASET].dropna().unique())

    diff_df = prebuilt_diff_df if prebuilt_diff_df is not None \
              else _build_diff_records(df, _datasets)
    if diff_df.empty:
        print("  no diff records to plot")
        return

    # Collect unique methods for colour assignment
    all_methods = sorted(diff_df["pretrain_method"].unique())
    method_color = {m: PRETRAIN_SCATTER_COLORS.get(m, f"C{i}")
                    for i, m in enumerate(all_methods)}

    # Collect unique datasets for marker assignment
    all_ds = sorted(diff_df["dataset"].unique())
    ds_marker = {d: _SCATTER_MARKERS[i % len(_SCATTER_MARKERS)]
                 for i, d in enumerate(all_ds)}

    # Only keep properties that exist in at least one dataset's metadata dict.
    # This automatically drops _std entries when the metadata only carries _mean
    # keys (e.g. split-shift metadata), avoiding blank panels.
    _avail_keys: set[str] = set()
    for _ds_props in metadata.values():
        _avail_keys.update(_ds_props.keys())
    _active_props = [(k, lbl) for k, lbl in GRAPH_PROPERTIES if k in _avail_keys]
    prop_groups = [_active_props[i:i + PROPS_PER_PLOT]
                   for i in range(0, len(_active_props), PROPS_PER_PLOT)]

    scopes = sorted(diff_df["scope"].unique())
    # When pooling, fractions collapse to a single sentinel None
    fractions = [None] if pool_fractions else sorted(
        diff_df["fraction"].dropna().unique()
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    for frac in fractions:
        for scope in scopes:
            if frac is None:
                subset = diff_df[diff_df["scope"] == scope]
            else:
                subset = diff_df[
                    (diff_df["fraction"] == frac) & (diff_df["scope"] == scope)
                ]
            if subset.empty:
                continue

            scope_label = "Probe" if scope == "probe" else "Full FT"
            frac_label  = "all fractions" if frac is None else f"fraction = {frac:g}"
            fname_frac  = "" if frac is None else f"_frac{frac:g}"

            for gi, prop_group in enumerate(prop_groups):
                n_props = len(prop_group)
                n_pcols = 3
                n_prows = math.ceil(n_props / n_pcols)

                fig_w = 4.5 * n_pcols
                fig_h = 3.8 * n_prows + 1.0

                with mpl.rc_context(RC_PARAMS):
                    fig, axes = plt.subplots(n_prows, n_pcols,
                                             figsize=(fig_w, fig_h),
                                             squeeze=False)

                    for pi, (prop_key, prop_label) in enumerate(prop_group):
                        pr, pc = divmod(pi, n_pcols)
                        ax = axes[pr][pc]
                        ax.axhline(0, color="0.6", linewidth=0.8, linestyle="--", zorder=1)

                        for method in all_methods:
                            m_sub = subset[subset["pretrain_method"] == method]
                            xs, ys, yerrs, labels_ds = [], [], [], []
                            for _, row in m_sub.iterrows():
                                ds = row["dataset"]
                                if ds not in metadata or prop_key not in metadata[ds]:
                                    continue
                                xs.append(metadata[ds][prop_key])
                                ys.append(row["norm_diff"])
                                yerrs.append(row["norm_err"])
                                labels_ds.append(ds)

                            if not xs:
                                continue
                            color = method_color[method]
                            for x, y, ye, ds in zip(xs, ys, yerrs, labels_ds):
                                mkr = ds_marker.get(ds, "o")
                                ax.errorbar(x, y, yerr=ye,
                                            color=color, marker=mkr,
                                            linestyle="none",
                                            markersize=7, capsize=3,
                                            capthick=1.0, elinewidth=1.0,
                                            zorder=3)

                        ax.set_xlabel(prop_label + x_label_suffix, fontsize=9)
                        if pc == 0:
                            ax.set_ylabel("Δ perf. (dataset σ units)", fontsize=9)
                        ax.tick_params(labelsize=8)

                    # Hide unused subplots
                    for ei in range(n_props, n_prows * n_pcols):
                        ri, ci = divmod(ei, n_pcols)
                        axes[ri][ci].set_visible(False)

                    # Method legend
                    from matplotlib.lines import Line2D
                    method_handles = [
                        Line2D([0], [0], color=method_color[m], marker="o",
                               linestyle="none", markersize=6,
                               label=_pretrain_label(m))
                        for m in all_methods
                    ]
                    # Dataset-marker legend
                    ds_handles = [
                        Line2D([0], [0], color="0.3", marker=ds_marker[d],
                               linestyle="none", markersize=6, label=d)
                        for d in all_ds if d in metadata
                    ]
                    all_leg = method_handles + ds_handles
                    if all_leg:
                        fig.legend(handles=all_leg, loc="lower center",
                                   ncol=min(len(all_leg), 5),
                                   fontsize=8, frameon=True,
                                   bbox_to_anchor=(0.5, 0.0))
                        fig.tight_layout(rect=(0, 0.10, 1, 0.95))
                    else:
                        fig.tight_layout(rect=(0, 0, 1, 0.95))

                    _title_extra = f"  ·  {title_extra}" if title_extra else ""
                    fig.suptitle(
                        f"Pretraining advantage vs graph properties{_title_extra}"
                        f"  ·  {scope_label}  ·  {frac_label}",
                        fontsize=12, fontweight="bold", y=0.99,
                    )

                    fname = f"{fname_prefix}{fname_frac}_{scope}_props{gi + 1}.{fmt}"
                    out_path = output_dir / fname
                    fig.savefig(out_path, bbox_inches="tight",
                                dpi=200 if fmt == "png" else None)
                    plt.close(fig)
                    print(f"  saved → {out_path}")


# ── Per-(fraction, scope) grouped bar chart ───────────────────────────────────

_DS_PER_BAR_ROW = 4   # max datasets shown side-by-side before wrapping to next row


def plot_improvement_barplots(
    df: pd.DataFrame,
    output_dir: Path,
    fmt: str = "png",
    datasets: list[str] | None = None,
) -> None:
    """Grouped bar chart per (fraction, scope).

    Datasets are arranged in a multi-row grid (≤ _DS_PER_BAR_ROW per row) so
    every group is wide enough to read.  Each panel is one dataset; bars within
    a panel are coloured by pretraining method.  Y-axis is shared across all
    panels so values are directly comparable.

    Bar height = sign·(finetune − random_init) / σ_dataset.
    Positive → pretraining helped.
    """
    _datasets = datasets or sorted(df[COL_DATASET].dropna().unique())
    diff_df = _build_diff_records(df, _datasets)
    if diff_df.empty:
        print("  no diff records – skipping improvement barplots")
        return

    all_methods = sorted(diff_df["pretrain_method"].unique())
    method_color = {m: PRETRAIN_SCATTER_COLORS.get(m, f"C{i}")
                    for i, m in enumerate(all_methods)}

    fractions = sorted(diff_df["fraction"].unique())
    scopes    = sorted(diff_df["scope"].unique())

    output_dir.mkdir(parents=True, exist_ok=True)

    M       = len(all_methods)
    bar_w   = 0.8 / max(M, 1)
    offsets = [(j - (M - 1) / 2) * bar_w for j in range(M)]
    x_pos   = np.arange(M)   # one bar per method within each panel

    for frac in fractions:
        for scope in scopes:
            subset = diff_df[
                (diff_df["fraction"] == frac) & (diff_df["scope"] == scope)
            ]
            if subset.empty:
                continue

            ds_in_subset = [d for d in _datasets if d in subset["dataset"].values]
            N = len(ds_in_subset)
            if N == 0:
                continue

            scope_label = "Probe" if scope == "probe" else "Full FT"

            # ── Multi-row layout: ≤ _DS_PER_BAR_ROW datasets per row ─────────
            n_cols = min(N, _DS_PER_BAR_ROW)
            n_rows = math.ceil(N / n_cols)

            cell_w = max(1.6 * M, 3.5)
            fig_w  = cell_w * n_cols + 0.5
            fig_h  = 4.2 * n_rows + 1.0   # extra for legend

            with mpl.rc_context(RC_PARAMS):
                fig, axes = plt.subplots(
                    n_rows, n_cols,
                    figsize=(fig_w, fig_h),
                    sharey=True,   # same y-scale across all panels
                    squeeze=False,
                )

                for di, ds in enumerate(ds_in_subset):
                    ri, ci = divmod(di, n_cols)
                    ax = axes[ri][ci]
                    ax.axhline(0, color="0.5", linewidth=0.8, linestyle="--", zorder=1)

                    for ji, method in enumerate(all_methods):
                        row = subset[
                            (subset["dataset"] == ds) &
                            (subset["pretrain_method"] == method)
                        ]
                        y    = float(row["norm_diff"].iloc[0]) if not row.empty else np.nan
                        yerr = float(row["norm_err"].iloc[0])  if not row.empty else 0.0
                        color = method_color[method]
                        ax.bar(ji, y, 0.7, color=color, alpha=0.85, zorder=2)
                        ax.errorbar(ji, y, yerr=yerr,
                                    fmt="none", color="0.2",
                                    capsize=3, capthick=0.8, elinewidth=0.8,
                                    zorder=3)

                    ax.set_title(ds, fontsize=8, fontweight="bold")
                    ax.set_xticks(x_pos)
                    ax.set_xticklabels(
                        [_pretrain_label(m) for m in all_methods],
                        rotation=40, ha="right", fontsize=7,
                    )
                    if ci == 0:
                        ax.set_ylabel(
                            "Δ perf. (dataset σ units)\n[+ve = pretraining helped]",
                            fontsize=8,
                        )
                    ax.tick_params(labelsize=7)

                # Hide unused panels
                for ei in range(N, n_rows * n_cols):
                    ri, ci = divmod(ei, n_cols)
                    axes[ri][ci].set_visible(False)

                # Shared colour legend
                from matplotlib.patches import Patch
                handles = [Patch(color=method_color[m], label=_pretrain_label(m))
                           for m in all_methods]
                fig.legend(handles=handles, loc="lower center",
                           ncol=min(M, 6), fontsize=8,
                           bbox_to_anchor=(0.5, 0.0), frameon=True)

                fig.suptitle(
                    f"Pretraining advantage per dataset  ·  "
                    f"{scope_label}  ·  fraction = {frac:g}",
                    fontsize=11, fontweight="bold",
                )
                fig.tight_layout(rect=(0, 0.08, 1, 0.96))

                fname = f"improvement_bars_frac{frac:g}_{scope}.{fmt}"
                out_path = output_dir / fname
                fig.savefig(out_path, bbox_inches="tight",
                            dpi=200 if fmt == "png" else None)
                plt.close(fig)
                print(f"  saved → {out_path}")


# ── Average improvement across datasets ───────────────────────────────────────

def plot_average_improvement(
    df: pd.DataFrame,
    output_dir: Path,
    fmt: str = "png",
    datasets: list[str] | None = None,
) -> None:
    """Single figure with one subplot per (fraction, scope) combination.

    Each subplot shows, for every pretraining method, the mean normalised
    performance improvement averaged across all datasets.

    mean_improvement(method, frac, scope) = (1/K) Σ_d norm_diff(d, method, …)
    propagated error                      = (1/K) √(Σ_d norm_err(d, …)²)

    Propagating in quadrature (rather than using the cross-dataset std) reflects
    the measurement uncertainty inherited from the seed variance.  Both sources
    of uncertainty are present, but the propagated one is directly connected to
    the underlying data.
    """
    _datasets = datasets or sorted(df[COL_DATASET].dropna().unique())
    diff_df = _build_diff_records(df, _datasets)
    if diff_df.empty:
        print("  no diff records – skipping average-improvement plot")
        return

    all_methods = sorted(diff_df["pretrain_method"].unique())
    method_color = {m: PRETRAIN_SCATTER_COLORS.get(m, f"C{i}")
                    for i, m in enumerate(all_methods)}

    fractions = sorted(diff_df["fraction"].unique())
    scopes    = sorted(diff_df["scope"].unique())

    n_fracs  = len(fractions)
    n_scopes = len(scopes)
    n_cols   = n_fracs
    n_rows   = n_scopes
    fig_w    = max(4.0 * n_cols, 8.0)
    fig_h    = 4.0 * n_rows + 0.6

    M     = len(all_methods)
    x_pos = np.arange(M)

    output_dir.mkdir(parents=True, exist_ok=True)

    with mpl.rc_context(RC_PARAMS):
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(fig_w, fig_h),
                                 squeeze=False,
                                 sharey=False)

        for ri, scope in enumerate(scopes):
            for ci, frac in enumerate(fractions):
                ax = axes[ri][ci]
                ax.axhline(0, color="0.5", linewidth=0.8, linestyle="--", zorder=1)

                means, errs = [], []
                for method in all_methods:
                    m_sub = diff_df[
                        (diff_df["fraction"] == frac) &
                        (diff_df["scope"] == scope) &
                        (diff_df["pretrain_method"] == method)
                    ]
                    vals  = m_sub["norm_diff"].dropna().values
                    nerrs = m_sub["norm_err"].values[~np.isnan(m_sub["norm_diff"].values)]
                    K = len(vals)
                    if K == 0:
                        means.append(np.nan)
                        errs.append(0.0)
                    else:
                        means.append(float(vals.mean()))
                        errs.append(float(np.sqrt((nerrs ** 2).sum())) / K)

                colors = [method_color[m] for m in all_methods]
                ax.bar(x_pos, means, 0.65, color=colors, alpha=0.85, zorder=2)
                ax.errorbar(x_pos, means, yerr=errs,
                            fmt="none", color="0.2",
                            capsize=3, capthick=0.8, elinewidth=0.8, zorder=3)

                scope_label = "Probe" if scope == "probe" else "Full FT"
                ax.set_title(f"{scope_label}  ·  fraction = {frac:g}", fontsize=10)
                ax.set_xticks(x_pos)
                ax.set_xticklabels(
                    [_pretrain_label(m) for m in all_methods],
                    rotation=40, ha="right", fontsize=8,
                )
                if ci == 0:
                    ax.set_ylabel("Mean Δ perf. (dataset σ units)", fontsize=9)

        # Shared colour legend (methods → colours)
        from matplotlib.patches import Patch
        handles = [Patch(color=method_color[m], label=_pretrain_label(m))
                   for m in all_methods]
        fig.legend(handles=handles, loc="lower center",
                   ncol=min(M, 6), fontsize=8,
                   bbox_to_anchor=(0.5, 0.0), frameon=True)

        fig.suptitle(
            "Average pretraining advantage across datasets\n"
            "(dataset-σ-normalised; +ve = pretraining helped)",
            fontsize=12,
        )
        fig.tight_layout(rect=(0, 0.08, 1, 0.96))

        fname = f"average_improvement.{fmt}"
        out_path = output_dir / fname
        fig.savefig(out_path, bbox_inches="tight",
                    dpi=200 if fmt == "png" else None)
        plt.close(fig)
        print(f"  saved → {out_path}")


# ── Per-fraction bar-chart plots ─────────────────────────────────────────────

# Reuse scatter colors for bar charts (one color per pretrain method)
PRETRAIN_BAR_COLORS: dict[str, str] = PRETRAIN_SCATTER_COLORS
FROM_SCRATCH_COLOR                  = "#888888"   # neutral gray for random-init bars
_BAR_DS_PER_ROW                     = 3           # dataset panels per row


def _get_from_scratch_bar(
    sub: pd.DataFrame,
    mc: str,
    sc: str,
    fraction: float,
    scope: str | None,
) -> dict[str, tuple]:
    """Return {scope_key: (mean, stderr)} for random-init modes at *fraction*.

    Values are averaged over all pretrain-method rows (they should be identical
    after ``apply_shared_random_init_baseline`` but averaging is a safe fallback).
    Returns ``(None, None)`` tuples when data is missing.
    """
    result: dict[str, tuple] = {}
    _scopes = ["probe", "full"] if scope is None else [scope]
    for s in _scopes:
        ft_mode = f"random-init-{s}"
        rows = sub[(sub[COL_FRACTION] == fraction) & (sub[COL_FT_MODE] == ft_mode)]
        if rows.empty or mc not in rows.columns:
            result[s] = (None, None)
            continue
        means = rows[mc].dropna()
        if means.empty:
            result[s] = (None, None)
            continue
        mean_v = float(means.mean())
        if sc in rows.columns:
            stds = rows[sc].fillna(0.0).values
            ns   = (rows[COL_SEED_COUNT].values
                    if COL_SEED_COUNT in rows.columns
                    else np.full(len(rows), 3))
            ses  = stds / np.sqrt(np.maximum(ns.astype(float), 1.0))
            se_v = float(np.sqrt((ses ** 2).sum())) / max(len(ses), 1)
        else:
            se_v = 0.0
        result[s] = (mean_v, se_v)
    return result


def _bar_layout(n_sub: int) -> tuple[float, float, np.ndarray]:
    """Return (bar_w, group_w, sub_offsets) for *n_sub* bars per x-group."""
    bar_w      = min(0.38, max(0.16, 0.72 / n_sub))
    group_w    = n_sub * bar_w + 0.22
    sub_offsets = np.array(
        [i * bar_w - (n_sub - 1) * bar_w / 2.0 for i in range(n_sub)]
    )
    return bar_w, group_w, sub_offsets


def plot_all_datasets_bars_frac(
    df: pd.DataFrame,
    output_dir: Path,
    fmt: str = "png",
    datasets: list[str] | None = None,
    fraction: float = 0.01,
    scope: str | None = None,
    pct: bool = False,
) -> Path | None:
    """Bar chart showing performance at a fixed data fraction.

    Layout: one panel per dataset, arranged ≤ ``_BAR_DS_PER_ROW`` per row.
    X-axis  : pretrain methods + one "From Scratch" group.
    Y-axis  : raw metric (or %, when *pct* is True).
    Colors  : one distinct color per pretrain method; gray for From Scratch.
    Scope   : when ``None`` two sub-bars per group (probe=solid, full=hatched);
              otherwise one bar per group.
    Error bars: ±stderr on every bar.

    Parameters
    ----------
    pct : bool
        When True, normalise each dataset's y-axis to [0, 100 %] using the
        same best/worst calibration as ``plot_all_datasets_percentage``.
    """
    from matplotlib.patches import Patch

    _datasets = datasets or sorted(df[COL_DATASET].dropna().unique())
    _datasets  = [d for d in _datasets if not df[df[COL_DATASET] == d].empty]
    if not _datasets:
        return None

    _scopes = ["probe", "full"] if scope is None else [scope]
    n_sub   = len(_scopes)
    bar_w, group_w, sub_offsets = _bar_layout(n_sub)

    N      = len(_datasets)
    n_cols = min(N, _BAR_DS_PER_ROW)
    n_rows = math.ceil(N / n_cols)

    cell_w = max(4.0, 1.5 * (len(df[COL_PRETRAIN].dropna().unique()) + 1))
    cell_h = 3.8
    fig_w  = max(cell_w * n_cols, 6.0)
    fig_h  = cell_h * n_rows + 1.4

    with mpl.rc_context(RC_PARAMS):
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(fig_w, fig_h),
            squeeze=False,
            sharey=False,
        )

        for di, dataset in enumerate(_datasets):
            ri, ci = divmod(di, n_cols)
            ax = axes[ri][ci]

            sub = df[df[COL_DATASET] == dataset].copy()

            # Monitor info
            monitor_test_col: str | None = None
            monitor_metric: str = "metric"
            monitor_mode: str   = "max"
            if COL_MON_COL in sub.columns:
                v = sub[COL_MON_COL].dropna()
                if not v.empty:
                    monitor_test_col = str(v.iloc[0])
            if COL_MON_METRIC in sub.columns:
                v = sub[COL_MON_METRIC].dropna()
                if not v.empty:
                    monitor_metric = str(v.iloc[0])
            if COL_MON_MODE in sub.columns:
                v = sub[COL_MON_MODE].dropna()
                if not v.empty:
                    monitor_mode = str(v.iloc[0])

            if monitor_test_col is None:
                ax.set_visible(False)
                continue

            mc = _mean_col(monitor_test_col)
            sc = mc.replace("_mean", "_std")
            if mc not in sub.columns:
                ax.set_visible(False)
                continue

            # Percentage normalisation params (if pct=True)
            best_val: float | None = None
            worst_val: float | None = None
            if pct:
                norm = _dataset_norm_params(df, dataset)
                if norm is None:
                    ax.set_visible(False)
                    continue
                best_val, worst_val, _, _ = norm

            def _to_y(v: float) -> float:
                return _to_pct(v, best_val, worst_val) if pct else v  # type: ignore[arg-type]

            def _to_ye(se: float) -> float:
                return _to_pct_stderr(se, best_val, worst_val) if pct else se  # type: ignore[arg-type]

            # Methods with finetune data at this fraction
            frac_sub  = sub[sub[COL_FRACTION] == fraction]
            ft_methods = sorted(
                frac_sub[
                    frac_sub[COL_FT_MODE].str.startswith("finetune")
                ][COL_PRETRAIN].dropna().unique()
            )
            if not ft_methods:
                ax.set_visible(False)
                continue

            n_groups  = len(ft_methods) + 1   # +1 for From Scratch
            x_centers = np.arange(n_groups, dtype=float) * group_w
            all_y: list[float] = []

            # ── Pretrain-method bars ──────────────────────────────────────────
            for gi, method in enumerate(ft_methods):
                method_sub = sub[sub[COL_PRETRAIN] == method]
                color = PRETRAIN_BAR_COLORS.get(method, f"C{gi}")
                x_c   = x_centers[gi]

                for si, s in enumerate(_scopes):
                    rows = method_sub[
                        (method_sub[COL_FRACTION] == fraction) &
                        (method_sub[COL_FT_MODE] == f"finetune-{s}")
                    ]
                    if rows.empty or mc not in rows.columns:
                        continue
                    y_raw = float(rows[mc].iloc[0])
                    if np.isnan(y_raw):
                        continue
                    n_s  = int(rows[COL_SEED_COUNT].iloc[0]) if COL_SEED_COUNT in rows.columns else 3
                    se_r = (float(rows[sc].iloc[0]) / np.sqrt(max(n_s, 1))
                            if sc in rows.columns else 0.0)
                    y_v  = _to_y(y_raw)
                    ye_v = _to_ye(se_r)
                    hatch = "////" if s == "full" else None
                    x_pos = x_c + sub_offsets[si]
                    ax.bar(x_pos, y_v, bar_w * 0.92,
                           color=color, alpha=0.85,
                           hatch=hatch,
                           edgecolor="0.95" if hatch is None else "0.4",
                           linewidth=0.4, zorder=2)
                    ax.errorbar(x_pos, y_v, yerr=ye_v,
                                fmt="none", color="0.15",
                                capsize=3, capthick=0.8, elinewidth=0.8, zorder=3)
                    all_y += [y_v + ye_v, y_v - ye_v]

            # ── From Scratch bar (gray, shared) ───────────────────────────────
            fs_data = _get_from_scratch_bar(sub, mc, sc, fraction, scope)
            x_c = x_centers[-1]
            for si, s in enumerate(_scopes):
                mv, se_r = fs_data.get(s, (None, None))
                if mv is None:
                    continue
                y_v  = _to_y(mv)
                ye_v = _to_ye(se_r)
                hatch = "////" if s == "full" else None
                x_pos = x_c + sub_offsets[si]
                ax.bar(x_pos, y_v, bar_w * 0.92,
                       color=FROM_SCRATCH_COLOR, alpha=0.75,
                       hatch=hatch,
                       edgecolor="0.95" if hatch is None else "0.5",
                       linewidth=0.4, zorder=2)
                ax.errorbar(x_pos, y_v, yerr=ye_v,
                            fmt="none", color="0.15",
                            capsize=3, capthick=0.8, elinewidth=0.8, zorder=3)
                all_y += [y_v + ye_v, y_v - ye_v]

            # X-ticks
            ax.set_xticks(x_centers)
            ax.set_xticklabels(
                [_pretrain_label(m) for m in ft_methods] + ["From\nScratch"],
                rotation=35, ha="right", fontsize=7,
            )
            ax.set_title(dataset, fontsize=9, fontweight="bold")

            if pct:
                ax.set_ylabel("Performance (%)", fontsize=8)
            else:
                ax.set_ylabel(_metric_ylabel(str(monitor_metric), str(monitor_mode)), fontsize=8)

            if all_y:
                lo, hi = min(all_y), max(all_y)
                pad = (hi - lo) * 0.12 if hi > lo else abs(hi) * 0.06 + 0.01
                ax.set_ylim(lo - pad, hi + pad * 2)
                if pct:
                    ax.set_ylim(max(0.0, lo - pad), min(100.0, hi + pad * 2))
            ax.tick_params(labelsize=7)
            ax.set_xlim(-group_w * 0.5, x_centers[-1] + group_w * 0.6)

        # ── Hide unused panels ────────────────────────────────────────────────
        for ei in range(N, n_rows * n_cols):
            ri, ci = divmod(ei, n_cols)
            axes[ri][ci].set_visible(False)

        # ── Legend ────────────────────────────────────────────────────────────
        all_methods_in_df = sorted(df[COL_PRETRAIN].dropna().unique())
        method_handles = [
            Patch(facecolor=PRETRAIN_BAR_COLORS.get(m, f"C{i}"), alpha=0.85,
                  label=_pretrain_label(m))
            for i, m in enumerate(all_methods_in_df)
        ]
        fs_handle = Patch(facecolor=FROM_SCRATCH_COLOR, alpha=0.75, label="From Scratch")
        scope_handles: list = []
        if scope is None:
            scope_handles = [
                Patch(facecolor="0.6", hatch=None,   edgecolor="0.4", lw=0.5, label="Probe head"),
                Patch(facecolor="0.6", hatch="////", edgecolor="0.5", lw=0.5, label="Full fine-tune"),
            ]
        all_handles = method_handles + [fs_handle] + scope_handles
        if all_handles:
            fig.legend(handles=all_handles, loc="lower center",
                       ncol=min(len(all_handles), 5),
                       fontsize=8, frameon=True, bbox_to_anchor=(0.5, 0.0))
            fig.tight_layout(rect=(0, 0.10, 1, 0.97))
        else:
            fig.tight_layout(rect=(0, 0, 1, 0.97))

        scope_str = (" (probe head)" if scope == "probe"
                     else " (full fine-tune)" if scope == "full" else "")
        pct_str   = " [%]" if pct else ""
        fig.suptitle(
            f"Performance{pct_str} at fraction = {fraction:g} — all datasets{scope_str}",
            fontsize=12, fontweight="bold", y=0.995,
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        stem_scope = "" if scope is None else f"_{scope}"
        stem_pct   = "_pct" if pct else ""
        stem       = f"all_datasets{stem_pct}{stem_scope}"
        out_path   = output_dir / f"{stem}.{fmt}"
        fig.savefig(out_path, bbox_inches="tight", dpi=200 if fmt == "png" else None)
        plt.close(fig)
        print(f"  saved → {out_path}")
        return out_path


def plot_all_datasets_bars_frac_combined(
    df_gin: pd.DataFrame,
    df_gpse: pd.DataFrame,
    output_dir: Path,
    fmt: str = "png",
    datasets: list[str] | None = None,
    fraction: float = 0.01,
    scope: str | None = None,
    pct: bool = False,
) -> Path | None:
    """Combined GIN + GPSE bar chart at a fixed data fraction.

    Same panel layout as ``plot_all_datasets_bars_frac`` but each x-group has
    sub-bars for both models:
    - scope is not None → 2 sub-bars: GIN (lighter alpha), GPSE (full alpha)
    - scope is None     → 4 sub-bars: GIN-probe, GIN-full, GPSE-probe, GPSE-full
    From Scratch also has one sub-bar per model (×scope).
    GIN bars are drawn with alpha=0.60; GPSE bars with alpha=1.0.
    Full fine-tune bars are additionally hatched with ``////``.

    Parameters
    ----------
    pct : bool
        Normalise y-axis jointly across both models (0 % = worst, 100 % = best).
    """
    from matplotlib.patches import Patch

    gin_ds  = set(df_gin[COL_DATASET].dropna().unique())
    gpse_ds = set(df_gpse[COL_DATASET].dropna().unique())
    common  = sorted(gin_ds & gpse_ds)
    if datasets:
        common = [d for d in common if d in datasets]
    if not common:
        print("  [combined bars] no common datasets – skipping")
        return None

    _scopes = ["probe", "full"] if scope is None else [scope]
    # Sub-bar ordering: (gin,probe), (gin,full), (gpse,probe), (gpse,full) when scope=None;
    # (gin,scope), (gpse,scope) when scope is set.
    sub_keys: list[tuple[str, str]] = []
    for mk in ("gin", "gpse_backbone"):
        for s in _scopes:
            sub_keys.append((mk, s))
    n_sub = len(sub_keys)

    bar_w, group_w, sub_offsets = _bar_layout(n_sub)

    N      = len(common)
    n_cols = min(N, _BAR_DS_PER_ROW)
    n_rows = math.ceil(N / n_cols)

    all_methods_union: list[str] = sorted(
        set(df_gin[COL_PRETRAIN].dropna().unique()) |
        set(df_gpse[COL_PRETRAIN].dropna().unique())
    )
    cell_w = max(4.5, 1.6 * (len(all_methods_union) + 1))
    cell_h = 3.8
    fig_w  = max(cell_w * n_cols, 6.0)
    fig_h  = cell_h * n_rows + 1.6

    with mpl.rc_context(RC_PARAMS):
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(fig_w, fig_h),
            squeeze=False,
            sharey=False,
        )

        for di, dataset in enumerate(common):
            ri, ci = divmod(di, n_cols)
            ax = axes[ri][ci]

            sub_gin  = df_gin[df_gin[COL_DATASET] == dataset].copy()
            sub_gpse = df_gpse[df_gpse[COL_DATASET] == dataset].copy()

            # Monitor info (prefer gin)
            monitor_test_col: str | None = None
            monitor_metric: str = "metric"
            monitor_mode: str   = "max"
            for sub_m in (sub_gin, sub_gpse):
                if monitor_test_col is None and COL_MON_COL in sub_m.columns:
                    v = sub_m[COL_MON_COL].dropna()
                    if not v.empty:
                        monitor_test_col = str(v.iloc[0])
                if COL_MON_METRIC in sub_m.columns:
                    v = sub_m[COL_MON_METRIC].dropna()
                    if not v.empty:
                        monitor_metric = str(v.iloc[0])
                if COL_MON_MODE in sub_m.columns:
                    v = sub_m[COL_MON_MODE].dropna()
                    if not v.empty:
                        monitor_mode = str(v.iloc[0])

            if monitor_test_col is None:
                ax.set_visible(False)
                continue

            mc = _mean_col(monitor_test_col)
            sc = mc.replace("_mean", "_std")

            # Percentage normalisation (joint across both models)
            best_val: float | None = None
            worst_val: float | None = None
            if pct:
                norm = _dataset_norm_params_combined(df_gin, df_gpse, dataset)
                if norm is None:
                    ax.set_visible(False)
                    continue
                best_val, worst_val, _, _ = norm

            def _to_y(v: float) -> float:
                return _to_pct(v, best_val, worst_val) if pct else v  # type: ignore[arg-type]

            def _to_ye(se: float) -> float:
                return _to_pct_stderr(se, best_val, worst_val) if pct else se  # type: ignore[arg-type]

            # Union of methods at this fraction in either model
            ft_methods_set: set[str] = set()
            for df_m in (df_gin, df_gpse):
                sub_m = df_m[df_m[COL_DATASET] == dataset]
                frac_sub = sub_m[sub_m[COL_FRACTION] == fraction]
                ft_methods_set.update(
                    frac_sub[
                        frac_sub[COL_FT_MODE].str.startswith("finetune")
                    ][COL_PRETRAIN].dropna().unique()
                )
            ft_methods = sorted(ft_methods_set)
            if not ft_methods:
                ax.set_visible(False)
                continue

            n_groups  = len(ft_methods) + 1
            x_centers = np.arange(n_groups, dtype=float) * group_w
            all_y: list[float] = []

            # ── Pretrain-method bars ──────────────────────────────────────────
            for gi, method in enumerate(ft_methods):
                color = PRETRAIN_BAR_COLORS.get(method, f"C{gi}")
                x_c   = x_centers[gi]

                for sbi, (model_key, s) in enumerate(sub_keys):
                    df_m  = df_gin if model_key == "gin" else df_gpse
                    sub_m = df_m[df_m[COL_DATASET] == dataset]
                    rows  = sub_m[
                        (sub_m[COL_PRETRAIN] == method) &
                        (sub_m[COL_FRACTION] == fraction) &
                        (sub_m[COL_FT_MODE] == f"finetune-{s}")
                    ]
                    if rows.empty or mc not in rows.columns:
                        continue
                    y_raw = float(rows[mc].iloc[0])
                    if np.isnan(y_raw):
                        continue
                    n_s  = int(rows[COL_SEED_COUNT].iloc[0]) if COL_SEED_COUNT in rows.columns else 3
                    se_r = (float(rows[sc].iloc[0]) / np.sqrt(max(n_s, 1))
                            if sc in rows.columns else 0.0)
                    y_v  = _to_y(y_raw)
                    ye_v = _to_ye(se_r)
                    alpha = 0.55 if model_key == "gin" else 1.0
                    hatch = "////" if s == "full" else None
                    x_pos = x_c + sub_offsets[sbi]
                    ax.bar(x_pos, y_v, bar_w * 0.92,
                           color=color, alpha=alpha,
                           hatch=hatch,
                           edgecolor="0.95" if hatch is None else "0.4",
                           linewidth=0.4, zorder=2)
                    ax.errorbar(x_pos, y_v, yerr=ye_v,
                                fmt="none", color="0.15",
                                capsize=2, capthick=0.7, elinewidth=0.7, zorder=3)
                    all_y += [y_v + ye_v, y_v - ye_v]

            # ── From Scratch bars (gray, per model) ───────────────────────────
            x_c = x_centers[-1]
            for sbi, (model_key, s) in enumerate(sub_keys):
                df_m  = df_gin if model_key == "gin" else df_gpse
                sub_m = df_m[df_m[COL_DATASET] == dataset]
                fs_d  = _get_from_scratch_bar(sub_m, mc, sc, fraction, s)
                mv, se_r = fs_d.get(s, (None, None))
                if mv is None:
                    continue
                y_v  = _to_y(mv)
                ye_v = _to_ye(se_r)
                alpha = 0.55 if model_key == "gin" else 1.0
                hatch = "////" if s == "full" else None
                x_pos = x_c + sub_offsets[sbi]
                ax.bar(x_pos, y_v, bar_w * 0.92,
                       color=FROM_SCRATCH_COLOR, alpha=alpha,
                       hatch=hatch,
                       edgecolor="0.95" if hatch is None else "0.5",
                       linewidth=0.4, zorder=2)
                ax.errorbar(x_pos, y_v, yerr=ye_v,
                            fmt="none", color="0.15",
                            capsize=2, capthick=0.7, elinewidth=0.7, zorder=3)
                all_y += [y_v + ye_v, y_v - ye_v]

            # Axis formatting
            ax.set_xticks(x_centers)
            ax.set_xticklabels(
                [_pretrain_label(m) for m in ft_methods] + ["From\nScratch"],
                rotation=35, ha="right", fontsize=7,
            )
            ax.set_title(dataset, fontsize=9, fontweight="bold")

            if pct:
                ax.set_ylabel("Performance (%)", fontsize=8)
            else:
                ax.set_ylabel(_metric_ylabel(str(monitor_metric), str(monitor_mode)), fontsize=8)

            if all_y:
                lo, hi = min(all_y), max(all_y)
                pad = (hi - lo) * 0.12 if hi > lo else abs(hi) * 0.06 + 0.01
                lo_lim = lo - pad
                hi_lim = hi + pad * 2
                if pct:
                    lo_lim = max(0.0, lo_lim)
                    hi_lim = min(100.0, hi_lim)
                ax.set_ylim(lo_lim, hi_lim)
            ax.tick_params(labelsize=7)
            ax.set_xlim(-group_w * 0.5, x_centers[-1] + group_w * 0.6)

        # ── Hide unused panels ────────────────────────────────────────────────
        for ei in range(N, n_rows * n_cols):
            ri, ci = divmod(ei, n_cols)
            axes[ri][ci].set_visible(False)

        # ── Legend ────────────────────────────────────────────────────────────
        method_handles = [
            Patch(facecolor=PRETRAIN_BAR_COLORS.get(m, f"C{i}"), alpha=0.85,
                  label=_pretrain_label(m))
            for i, m in enumerate(all_methods_union)
        ]
        fs_handle    = Patch(facecolor=FROM_SCRATCH_COLOR, alpha=0.8, label="From Scratch")
        model_handles = [
            Patch(facecolor="0.55", alpha=0.55, label="GIN (lighter = lower alpha)"),
            Patch(facecolor="0.35", alpha=1.00, label="GPSE (solid)"),
        ]
        scope_handles: list = []
        if scope is None:
            scope_handles = [
                Patch(facecolor="0.6", hatch=None,   edgecolor="0.4", lw=0.5, label="Probe head"),
                Patch(facecolor="0.6", hatch="////", edgecolor="0.5", lw=0.5, label="Full fine-tune"),
            ]
        all_handles = method_handles + [fs_handle] + model_handles + scope_handles
        if all_handles:
            fig.legend(handles=all_handles, loc="lower center",
                       ncol=min(len(all_handles), 5),
                       fontsize=8, frameon=True, bbox_to_anchor=(0.5, 0.0))
            fig.tight_layout(rect=(0, 0.10, 1, 0.97))
        else:
            fig.tight_layout(rect=(0, 0, 1, 0.97))

        scope_str = (" (probe head)" if scope == "probe"
                     else " (full fine-tune)" if scope == "full" else "")
        pct_str   = " [%]" if pct else ""
        fig.suptitle(
            f"Performance{pct_str} at fraction = {fraction:g}"
            f" — GIN & GPSE (common datasets){scope_str}",
            fontsize=12, fontweight="bold", y=0.995,
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        stem_scope = "" if scope is None else f"_{scope}"
        stem_pct   = "_pct" if pct else ""
        stem       = f"all_datasets{stem_pct}{stem_scope}_gin_and_gpse"
        out_path   = output_dir / f"{stem}.{fmt}"
        fig.savefig(out_path, bbox_inches="tight", dpi=200 if fmt == "png" else None)
        plt.close(fig)
        print(f"  saved → {out_path}")
        return out_path


def _generate_frac_bars(
    df: pd.DataFrame,
    output_dir: Path,
    fmt: str = "png",
    datasets: list[str] | None = None,
) -> None:
    """Generate bar-chart figures for every fraction present in *df*.

    Saves into sub-directories: ``<output_dir>/all_datasets_bars_frac_{frac:g}/``
    Each sub-directory gets six files:
      all_datasets.png, all_datasets_probe.png, all_datasets_full.png
      all_datasets_pct.png, all_datasets_pct_probe.png, all_datasets_pct_full.png
    """
    fractions = sorted(df[COL_FRACTION].dropna().unique())
    for frac in fractions:
        frac_dir = output_dir / f"all_datasets_bars_frac_{frac:g}"
        print(f"  fraction = {frac:g}  →  {frac_dir}")
        for scope in (None, "probe", "full"):
            for use_pct in (False, True):
                lbl = ("combined" if scope is None else scope) + (" pct" if use_pct else "")
                try:
                    plot_all_datasets_bars_frac(
                        df, frac_dir, fmt=fmt, datasets=datasets,
                        fraction=frac, scope=scope, pct=use_pct,
                    )
                except Exception as exc:
                    print(f"  ERROR bars frac={frac:g} ({lbl}): {exc}")


def _generate_frac_bars_combined(
    df_gin: pd.DataFrame,
    df_gpse: pd.DataFrame,
    output_dir: Path,
    fmt: str = "png",
    datasets: list[str] | None = None,
) -> None:
    """Same as ``_generate_frac_bars`` but for the combined GIN + GPSE plots."""
    fractions = sorted(
        set(df_gin[COL_FRACTION].dropna().tolist()) |
        set(df_gpse[COL_FRACTION].dropna().tolist())
    )
    for frac in fractions:
        frac_dir = output_dir / f"all_datasets_bars_frac_{frac:g}"
        print(f"  [combined] fraction = {frac:g}  →  {frac_dir}")
        for scope in (None, "probe", "full"):
            for use_pct in (False, True):
                lbl = ("combined" if scope is None else scope) + (" pct" if use_pct else "")
                try:
                    plot_all_datasets_bars_frac_combined(
                        df_gin, df_gpse, frac_dir, fmt=fmt, datasets=datasets,
                        fraction=frac, scope=scope, pct=use_pct,
                    )
                except Exception as exc:
                    print(f"  ERROR combined bars frac={frac:g} ({lbl}): {exc}")


# ── Combined GIN + GPSE learning-curve grids ─────────────────────────────────

def _dataset_norm_params_combined(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    dataset: str,
) -> tuple[float, float, str, str] | None:
    """Compute (best_val, worst_val, mean_col, monitor_mode) for one dataset
    across two DataFrames (e.g. gin and gpse_backbone).

    best_val / worst_val span the union of all metric values observed in both
    models, so 100 % and 0 % are calibrated jointly across models.
    """
    def _resolve(df: pd.DataFrame):
        sub = df[df[COL_DATASET] == dataset]
        if sub.empty:
            return None, None, None
        mode = (sub[COL_MON_MODE].dropna().iloc[0]
                if COL_MON_MODE in sub.columns and not sub[COL_MON_MODE].dropna().empty
                else "max")
        col  = (sub[COL_MON_COL].dropna().iloc[0]
                if COL_MON_COL in sub.columns and not sub[COL_MON_COL].dropna().empty
                else None)
        return mode, col, sub

    mode_a, col_a, sub_a = _resolve(df_a)
    mode_b, col_b, sub_b = _resolve(df_b)

    monitor_mode     = mode_a or mode_b or "max"
    monitor_test_col = col_a or col_b
    if monitor_test_col is None:
        return None

    mc = _mean_col(monitor_test_col)
    all_vals: list[float] = []
    for sub in (sub_a, sub_b):
        if sub is not None and mc in sub.columns:
            all_vals.extend(float(v) for v in sub[mc].dropna())
    if not all_vals:
        return None

    s = pd.Series(all_vals, dtype=float)
    if str(monitor_mode).lower() == "min":
        best_val, worst_val = float(s.min()), float(s.max())
    else:
        best_val, worst_val = float(s.max()), float(s.min())
    return best_val, worst_val, mc, str(monitor_mode)


def plot_all_datasets_combined(
    df_gin: pd.DataFrame,
    df_gpse: pd.DataFrame,
    output_dir: Path,
    fmt: str = "png",
    datasets: list[str] | None = None,
    scope: str | None = None,
) -> Path | None:
    """Master grid showing GIN and GPSE backbone results in the same cells.

    Only datasets present in *both* DataFrames are included.
    GIN uses lighter colours; GPSE uses darker colours.
    Columns = union of all pretraining methods; rows = common datasets.
    Y-axis is shared within each row (so both models share the same scale).

    Parameters
    ----------
    scope : ``None`` | ``"probe"`` | ``"full"``
        When set, only probe-head or full fine-tune modes are shown.
    """
    from matplotlib.lines import Line2D

    gin_ds  = set(df_gin[COL_DATASET].dropna().unique())
    gpse_ds = set(df_gpse[COL_DATASET].dropna().unique())
    common  = sorted(gin_ds & gpse_ds)
    if datasets:
        common = [d for d in common if d in datasets]
    if not common:
        print("  [combined] no common datasets found – skipping combined plot")
        return None

    all_methods: list[str] = sorted(
        set(df_gin[COL_PRETRAIN].dropna().unique()) |
        set(df_gpse[COL_PRETRAIN].dropna().unique())
    )
    n_cols = len(all_methods)
    n_rows = len(common)

    cell_w = 3.2
    cell_h = 2.8
    fig_w  = max(cell_w * n_cols + 0.8, 8.0)
    fig_h  = cell_h * n_rows + 1.4

    with mpl.rc_context(RC_PARAMS):
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(fig_w, fig_h),
            sharey="row",
            sharex=True,
            squeeze=False,
        )

        for ci, method in enumerate(all_methods):
            axes[0][ci].set_title(
                _pretrain_label(method), fontweight="bold", fontsize=8, pad=4
            )

        modes_union: set[str] = set()

        for ri, dataset in enumerate(common):
            sub_gin  = df_gin[df_gin[COL_DATASET] == dataset].copy()
            sub_gpse = df_gpse[df_gpse[COL_DATASET] == dataset].copy()

            # Resolve monitor info – prefer gin, fall back to gpse
            monitor_test_col: str | None = None
            monitor_metric: str = "metric"
            monitor_mode: str = "max"
            for sub in (sub_gin, sub_gpse):
                if monitor_test_col is None and COL_MON_COL in sub.columns:
                    v = sub[COL_MON_COL].dropna()
                    if not v.empty:
                        monitor_test_col = str(v.iloc[0])
                if COL_MON_METRIC in sub.columns:
                    v = sub[COL_MON_METRIC].dropna()
                    if not v.empty:
                        monitor_metric = str(v.iloc[0])
                if COL_MON_MODE in sub.columns:
                    v = sub[COL_MON_MODE].dropna()
                    if not v.empty:
                        monitor_mode = str(v.iloc[0])

            if monitor_test_col is None:
                for ci in range(n_cols):
                    axes[ri][ci].set_visible(False)
                continue

            mc = _mean_col(monitor_test_col)
            sc = mc.replace("_mean", "_std")

            # Y-limits: span of both models' data for this dataset
            subs_with_mc = [s for s in (sub_gin, sub_gpse) if mc in s.columns]
            if not subs_with_mc:
                for ci in range(n_cols):
                    axes[ri][ci].set_visible(False)
                continue
            all_sub = pd.concat(subs_with_mc, ignore_index=True)
            y_lo, y_hi = _y_limits(all_sub, mc, sc)

            # Modes present in either model
            modes_either = (
                set(sub_gin[COL_FT_MODE].unique()) |
                set(sub_gpse[COL_FT_MODE].unique())
            )
            modes_present = [m for m in MODE_ORDER if m in modes_either]
            modes_present += [m for m in modes_either if m not in MODE_ORDER]
            modes_present = _filter_modes_by_scope(modes_present, scope)
            modes_union.update(modes_present)

            ylabel_txt = _metric_ylabel(str(monitor_metric), str(monitor_mode))
            axes[ri][0].set_ylabel(ylabel_txt, fontsize=7)
            axes[ri][0].annotate(
                dataset,
                xy=(-0.55, 0.5), xycoords="axes fraction",
                fontsize=7, fontweight="bold",
                ha="right", va="center",
                rotation=0,
                annotation_clip=False,
            )

            for ci, method in enumerate(all_methods):
                ax = axes[ri][ci]
                gin_task  = sub_gin[sub_gin[COL_PRETRAIN] == method]
                gpse_task = sub_gpse[sub_gpse[COL_PRETRAIN] == method]

                if gin_task.empty and gpse_task.empty:
                    ax.set_visible(False)
                    continue

                fractions_all: list[float] = sorted(
                    set(gin_task[COL_FRACTION].dropna().tolist()) |
                    set(gpse_task[COL_FRACTION].dropna().tolist())
                )

                for model_key, task_df in (("gin", gin_task), ("gpse_backbone", gpse_task)):
                    if task_df.empty or mc not in task_df.columns:
                        continue
                    for mode in modes_present:
                        mode_df = task_df[task_df[COL_FT_MODE] == mode].sort_values(COL_FRACTION)
                        if mode_df.empty:
                            continue
                        x    = mode_df[COL_FRACTION].values
                        y    = mode_df[mc].values
                        yerr = _compute_stderr(mode_df, sc).values
                        ax.errorbar(
                            x, y, yerr=yerr,
                            color=_combined_mode_color(mode, model_key),
                            marker=_mode_marker(mode),
                            linestyle=_mode_linestyle(mode),
                            linewidth=RC_PARAMS["lines.linewidth"],
                            markersize=RC_PARAMS["lines.markersize"] * 0.8,
                            capsize=2, capthick=0.9, elinewidth=0.9,
                            zorder=3,
                        )

                ax.set_ylim(y_lo, y_hi)
                if fractions_all:
                    ax.set_xticks(fractions_all)
                    ax.xaxis.set_major_formatter(
                        matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}")
                    )
                ax.set_xlim(-0.02, max(fractions_all) + 0.05 if fractions_all else 1.1)
                ax.tick_params(labelsize=7)
                if ri == n_rows - 1:
                    ax.set_xlabel("Data fraction", fontsize=7)

        # ── Legend ────────────────────────────────────────────────────────────
        all_modes = [m for m in MODE_ORDER if m in modes_union]
        all_modes += [m for m in modes_union if m not in MODE_ORDER]

        groups_present = []
        for gkey, gstart in [("pretrained", "finetune"), ("from_scratch", "random-init")]:
            if any(m.startswith(gstart) for m in all_modes):
                groups_present.append(gkey)

        model_color_handles = []
        for model_key in ("gin", "gpse_backbone"):
            for gkey in groups_present:
                color  = _MODEL_GROUP_COLORS[model_key][gkey]
                marker = GROUP_MARKERS[gkey]
                label  = f"{MODEL_DISPLAY[model_key]} – {GROUP_LABELS[gkey]}"
                model_color_handles.append(
                    Line2D(
                        [0], [0], color=color, marker=marker,
                        linestyle="-", linewidth=RC_PARAMS["lines.linewidth"],
                        markersize=RC_PARAMS["lines.markersize"], label=label,
                    )
                )

        scopes_present = []
        if scope is None:
            for skey in ("probe", "full"):
                if any(skey in m for m in all_modes):
                    scopes_present.append(skey)

        style_handles = [
            Line2D([0], [0], color="0.35", linestyle=TUNE_LINESTYLES[s],
                   linewidth=RC_PARAMS["lines.linewidth"], label=TUNE_LABELS[s])
            for s in scopes_present
        ]

        all_handles = model_color_handles + style_handles
        if all_handles:
            fig.legend(
                handles=all_handles, loc="lower center",
                ncol=min(len(all_handles), 4), frameon=True,
                fontsize=RC_PARAMS["legend.fontsize"],
                bbox_to_anchor=(0.5, 0.0),
            )
            fig.tight_layout(rect=(0, 0.08, 1, 0.97))
        else:
            fig.tight_layout(rect=(0, 0, 1, 0.97))

        title = "Learning curves — GIN & GPSE (common datasets)"
        if scope == "probe":
            title += " (probe head)"
        elif scope == "full":
            title += " (full fine-tune)"
        fig.suptitle(title, fontsize=12, fontweight="bold", y=0.995)

        output_dir.mkdir(parents=True, exist_ok=True)
        stem = (
            "all_datasets_gin_and_gpse" if scope is None
            else f"all_datasets_{scope}_gin_and_gpse"
        )
        out_path = output_dir / f"{stem}.{fmt}"
        fig.savefig(out_path, bbox_inches="tight", dpi=200 if fmt == "png" else None)
        plt.close(fig)
        print(f"  saved → {out_path}")
        return out_path


def plot_all_datasets_percentage_combined(
    df_gin: pd.DataFrame,
    df_gpse: pd.DataFrame,
    output_dir: Path,
    fmt: str = "png",
    datasets: list[str] | None = None,
    scope: str | None = None,
) -> Path | None:
    """Same layout as plot_all_datasets_combined but y-axis is [0, 100 %].

    100 % = best mean result observed across *both* models for each dataset.
    0   % = worst mean result observed across *both* models.

    Calibrating the scale jointly means the two models are directly comparable
    on the same percentage axis per dataset.

    Parameters
    ----------
    scope : ``None`` | ``"probe"`` | ``"full"``
        When set, only probe-head or full fine-tune modes are shown.
    """
    from matplotlib.lines import Line2D

    gin_ds  = set(df_gin[COL_DATASET].dropna().unique())
    gpse_ds = set(df_gpse[COL_DATASET].dropna().unique())
    common  = sorted(gin_ds & gpse_ds)
    if datasets:
        common = [d for d in common if d in datasets]
    if not common:
        print("  [combined pct] no common datasets found – skipping")
        return None

    all_methods: list[str] = sorted(
        set(df_gin[COL_PRETRAIN].dropna().unique()) |
        set(df_gpse[COL_PRETRAIN].dropna().unique())
    )
    n_cols = len(all_methods)
    n_rows = len(common)

    cell_w = 3.2
    cell_h = 2.8
    fig_w  = max(cell_w * n_cols + 0.8, 8.0)
    fig_h  = cell_h * n_rows + 1.4

    with mpl.rc_context(RC_PARAMS):
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(fig_w, fig_h),
            sharey="row",
            sharex=True,
            squeeze=False,
        )

        for ci, method in enumerate(all_methods):
            axes[0][ci].set_title(
                _pretrain_label(method), fontweight="bold", fontsize=8, pad=4
            )

        modes_union: set[str] = set()

        for ri, dataset in enumerate(common):
            sub_gin  = df_gin[df_gin[COL_DATASET] == dataset].copy()
            sub_gpse = df_gpse[df_gpse[COL_DATASET] == dataset].copy()

            # Combined normalisation (best/worst across BOTH models)
            norm = _dataset_norm_params_combined(df_gin, df_gpse, dataset)
            if norm is None:
                for ci in range(n_cols):
                    axes[ri][ci].set_visible(False)
                continue
            best_val, worst_val, mc, monitor_mode = norm
            sc = mc.replace("_mean", "_std")

            monitor_metric = "metric"
            for sub in (sub_gin, sub_gpse):
                if COL_MON_METRIC in sub.columns:
                    v = sub[COL_MON_METRIC].dropna()
                    if not v.empty:
                        monitor_metric = str(v.iloc[0])
                        break

            modes_either = (
                set(sub_gin[COL_FT_MODE].unique()) |
                set(sub_gpse[COL_FT_MODE].unique())
            )
            modes_present = [m for m in MODE_ORDER if m in modes_either]
            modes_present += [m for m in modes_either if m not in MODE_ORDER]
            modes_present = _filter_modes_by_scope(modes_present, scope)
            modes_union.update(modes_present)

            # Per-row % y-limits from both models
            pct_vals, pct_errs = [], []
            for sub in (sub_gin, sub_gpse):
                if mc not in sub.columns:
                    continue
                for _, row in sub.iterrows():
                    v = row.get(mc)
                    if v is not None and not (isinstance(v, float) and np.isnan(v)):
                        pct_vals.append(_to_pct(float(v), best_val, worst_val))
                    if sc in sub.columns:
                        s = row.get(sc)
                        n_s = row.get(COL_SEED_COUNT, 3) or 3
                        if s is not None and not (isinstance(s, float) and np.isnan(s)):
                            pct_errs.append(
                                _to_pct_stderr(float(s) / np.sqrt(n_s), best_val, worst_val)
                            )
            pad  = 5.0
            lo   = (min(pct_vals) - (max(pct_errs) if pct_errs else 0)) if pct_vals else 0.0
            hi   = (max(pct_vals) + (max(pct_errs) if pct_errs else 0)) if pct_vals else 100.0
            y_lo = max(0.0,   lo - pad)
            y_hi = min(100.0, hi + pad)

            axes[ri][0].set_ylabel("Performance (%)", fontsize=7)
            axes[ri][0].annotate(
                dataset,
                xy=(-0.55, 0.5), xycoords="axes fraction",
                fontsize=7, fontweight="bold",
                ha="right", va="center",
                rotation=0,
                annotation_clip=False,
            )

            for ci, method in enumerate(all_methods):
                ax = axes[ri][ci]
                gin_task  = sub_gin[sub_gin[COL_PRETRAIN] == method]
                gpse_task = sub_gpse[sub_gpse[COL_PRETRAIN] == method]

                if gin_task.empty and gpse_task.empty:
                    ax.set_visible(False)
                    continue

                fractions_all: list[float] = sorted(
                    set(gin_task[COL_FRACTION].dropna().tolist()) |
                    set(gpse_task[COL_FRACTION].dropna().tolist())
                )

                for model_key, task_df in (("gin", gin_task), ("gpse_backbone", gpse_task)):
                    if task_df.empty or mc not in task_df.columns:
                        continue
                    for mode in modes_present:
                        mode_df = task_df[task_df[COL_FT_MODE] == mode].sort_values(COL_FRACTION)
                        if mode_df.empty:
                            continue
                        x    = mode_df[COL_FRACTION].values
                        y    = np.array([_to_pct(float(v), best_val, worst_val)
                                         for v in mode_df[mc].values])
                        if sc in mode_df.columns:
                            n_s  = (mode_df[COL_SEED_COUNT].values
                                    if COL_SEED_COUNT in mode_df.columns
                                    else np.full(len(x), 3))
                            yerr = np.array([
                                _to_pct_stderr(float(s) / np.sqrt(max(n, 1)), best_val, worst_val)
                                for s, n in zip(mode_df[sc].values, n_s)
                            ])
                        else:
                            yerr = np.zeros(len(x))
                        ax.errorbar(
                            x, y, yerr=yerr,
                            color=_combined_mode_color(mode, model_key),
                            marker=_mode_marker(mode),
                            linestyle=_mode_linestyle(mode),
                            linewidth=RC_PARAMS["lines.linewidth"],
                            markersize=RC_PARAMS["lines.markersize"] * 0.8,
                            capsize=2, capthick=0.9, elinewidth=0.9,
                            zorder=3,
                        )

                ax.set_ylim(y_lo, y_hi)
                if fractions_all:
                    ax.set_xticks(fractions_all)
                    ax.xaxis.set_major_formatter(
                        matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}")
                    )
                ax.set_xlim(-0.02, max(fractions_all) + 0.05 if fractions_all else 1.1)
                ax.tick_params(labelsize=7)
                if ri == n_rows - 1:
                    ax.set_xlabel("Data fraction", fontsize=7)

        # ── Legend ────────────────────────────────────────────────────────────
        all_modes = [m for m in MODE_ORDER if m in modes_union]
        all_modes += [m for m in modes_union if m not in MODE_ORDER]

        groups_present = []
        for gkey, gstart in [("pretrained", "finetune"), ("from_scratch", "random-init")]:
            if any(m.startswith(gstart) for m in all_modes):
                groups_present.append(gkey)

        model_color_handles = []
        for model_key in ("gin", "gpse_backbone"):
            for gkey in groups_present:
                color  = _MODEL_GROUP_COLORS[model_key][gkey]
                marker = GROUP_MARKERS[gkey]
                label  = f"{MODEL_DISPLAY[model_key]} – {GROUP_LABELS[gkey]}"
                model_color_handles.append(
                    Line2D(
                        [0], [0], color=color, marker=marker,
                        linestyle="-", linewidth=RC_PARAMS["lines.linewidth"],
                        markersize=RC_PARAMS["lines.markersize"], label=label,
                    )
                )

        scopes_present = []
        if scope is None:
            for skey in ("probe", "full"):
                if any(skey in m for m in all_modes):
                    scopes_present.append(skey)

        style_handles = [
            Line2D([0], [0], color="0.35", linestyle=TUNE_LINESTYLES[s],
                   linewidth=RC_PARAMS["lines.linewidth"], label=TUNE_LABELS[s])
            for s in scopes_present
        ]

        all_handles = model_color_handles + style_handles
        if all_handles:
            fig.legend(
                handles=all_handles, loc="lower center",
                ncol=min(len(all_handles), 4), frameon=True,
                fontsize=RC_PARAMS["legend.fontsize"],
                bbox_to_anchor=(0.5, 0.0),
            )
            fig.tight_layout(rect=(0, 0.08, 1, 0.97))
        else:
            fig.tight_layout(rect=(0, 0, 1, 0.97))

        title = (
            "Learning curves (%) — GIN & GPSE (common datasets)"
            "  ·  100 % = best across both models, 0 % = worst"
        )
        if scope == "probe":
            title += " (probe head)"
        elif scope == "full":
            title += " (full fine-tune)"
        fig.suptitle(title, fontsize=11, fontweight="bold", y=0.995)

        output_dir.mkdir(parents=True, exist_ok=True)
        stem = (
            "all_datasets_pct_gin_and_gpse" if scope is None
            else f"all_datasets_pct_{scope}_gin_and_gpse"
        )
        out_path = output_dir / f"{stem}.{fmt}"
        fig.savefig(out_path, bbox_inches="tight", dpi=200 if fmt == "png" else None)
        plt.close(fig)
        print(f"  saved → {out_path}")
        return out_path


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot aggregated finetuning results.")
    p.add_argument(
        "--model", default="gin", choices=list(MODELS),
        help="Model backbone whose results to plot. Determines default input CSV and "
             "output directory (outputs/{model}/). Default: gin.",
    )
    p.add_argument(
        "--input", type=Path, default=None,
        help="Path to aggregated_results.csv (default: outputs/{model}/aggregated_results.csv).",
    )
    p.add_argument(
        "--output-dir", type=Path, default=None,
        help="Directory to write figures into (default: outputs/{model}/figures/).",
    )
    p.add_argument(
        "--fmt", default="png", choices=["png", "pdf", "svg"],
        help="Output format (default: png).",
    )
    p.add_argument(
        "--datasets", nargs="+", default=None,
        help="Restrict to specific dataset names (default: all).",
    )
    p.add_argument(
        "--no-curves", action="store_true",
        help="Skip per-dataset learning-curve figures.",
    )
    p.add_argument(
        "--no-master", action="store_true",
        help="Skip the combined all-datasets master learning-curve figure.",
    )
    p.add_argument(
        "--no-pct-master", action="store_true",
        help="Skip the combined all-datasets percentage master learning-curve figure.",
    )
    p.add_argument(
        "--no-percentage", action="store_true",
        help="Skip percentage-normalised learning-curve figures.",
    )
    p.add_argument(
        "--no-correlations", action="store_true",
        help="Skip property-correlation scatter figures.",
    )
    p.add_argument(
        "--no-shift-correlations", action="store_true",
        help="Skip train-val and val-test split-shift correlation scatter figures.",
    )
    p.add_argument(
        "--no-frac-avg-correlations", action="store_true",
        help="Skip the fraction-averaged correlation scatter figures.",
    )
    p.add_argument(
        "--no-barplots", action="store_true",
        help="Skip per-(fraction, scope) grouped improvement bar charts.",
    )
    p.add_argument(
        "--no-avg-improvement", action="store_true",
        help="Skip the average-improvement summary figure.",
    )
    p.add_argument(
        "--metadata-dir", type=Path, default=METADATA_DIR_DEFAULT,
        help="Directory containing per-dataset metadata JSON files.",
    )
    p.add_argument(
        "--combined-only", action="store_true",
        help=(
            "Skip all per-model plots and only generate the combined GIN + GPSE "
            "summarising figures (requires both outputs/gin/aggregated_results.csv "
            "and outputs/gpse_backbone/aggregated_results.csv to exist)."
        ),
    )
    p.add_argument(
        "--no-bars-frac", action="store_true",
        help="Skip per-fraction bar-chart figures (all_datasets_bars_frac_*/).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── --combined-only: skip single-model work entirely ─────────────────────
    if args.combined_only:
        _gin_csv  = _OUTPUTS_BASE / "gin"           / "aggregated_results.csv"
        _gpse_csv = _OUTPUTS_BASE / "gpse_backbone" / "aggregated_results.csv"
        _combined_dir = _OUTPUTS_BASE / "combined" / "figures"
        print("Mode     : combined-only (GIN + GPSE)")
        print(f"GIN CSV  : {_gin_csv}")
        print(f"GPSE CSV : {_gpse_csv}")
        print(f"Out dir  : {_combined_dir}")
        for p in (_gin_csv, _gpse_csv):
            if not p.exists():
                raise FileNotFoundError(f"Required CSV not found: {p}")
        _df_gin  = apply_shared_random_init_baseline(pd.read_csv(_gin_csv))
        _df_gpse = apply_shared_random_init_baseline(pd.read_csv(_gpse_csv))
        print(f"Loaded {len(_df_gin)} GIN rows, {len(_df_gpse)} GPSE rows")
        print("\n── Combined GIN + GPSE summarising plots ──")
        for _scope in (None, "probe", "full"):
            _label = "combined" if _scope is None else _scope
            try:
                plot_all_datasets_combined(
                    _df_gin, _df_gpse, _combined_dir,
                    fmt=args.fmt, scope=_scope,
                )
            except Exception as exc:
                print(f"  ERROR combined learning curves ({_label}): {exc}")
            try:
                plot_all_datasets_percentage_combined(
                    _df_gin, _df_gpse, _combined_dir,
                    fmt=args.fmt, scope=_scope,
                )
            except Exception as exc:
                print(f"  ERROR combined pct ({_label}): {exc}")
        if not args.no_bars_frac:
            print("\n── Combined per-fraction bar charts ──")
            try:
                _generate_frac_bars_combined(_df_gin, _df_gpse, _combined_dir, fmt=args.fmt)
            except Exception as exc:
                print(f"  ERROR combined bars: {exc}")
        print(f"\nDone. Figures in: {_combined_dir}")
        return

    # Resolve model-specific paths (explicit args take precedence)
    _model_dir = _OUTPUTS_BASE / args.model
    input_path = args.input      or _model_dir / "aggregated_results.csv"
    output_dir = args.output_dir or _model_dir / "figures"

    print(f"Model    : {args.model}")
    print(f"Input    : {input_path}")
    print(f"Out dir  : {output_dir}")

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} rows from {input_path}")
    df = apply_shared_random_init_baseline(df)

    if COL_DATASET not in df.columns:
        raise ValueError(
            f"Expected column {COL_DATASET!r} not found. "
            f"Available columns: {list(df.columns)}"
        )

    datasets = args.datasets or sorted(df[COL_DATASET].dropna().unique())

    if not args.no_curves:
        print(f"\n── Learning-curve plots ({len(datasets)} dataset(s)) ──")
        for dataset in datasets:
            print(f"[{dataset}]")
            try:
                plot_dataset(df, dataset, output_dir, fmt=args.fmt)
            except Exception as exc:
                print(f"  ERROR: {exc}")

    if not args.no_master:
        print(f"\n── Master all-datasets learning-curve figure(s) ──")
        for master_scope in (None, "probe", "full"):
            label = "combined" if master_scope is None else master_scope
            try:
                plot_all_datasets(
                    df, output_dir, fmt=args.fmt,
                    datasets=datasets, scope=master_scope,
                )
            except Exception as exc:
                print(f"  ERROR ({label}): {exc}")

    if not args.no_pct_master:
        print(f"\n── Master all-datasets percentage learning-curve figure(s) ──")
        for master_scope in (None, "probe", "full"):
            label = "combined" if master_scope is None else master_scope
            try:
                plot_all_datasets_percentage(
                    df, output_dir, fmt=args.fmt,
                    datasets=datasets, scope=master_scope,
                )
            except Exception as exc:
                print(f"  ERROR ({label}): {exc}")

    # ── Combined GIN + GPSE master plots (generated alongside the per-model ones)
    _gin_csv  = _OUTPUTS_BASE / "gin"           / "aggregated_results.csv"
    _gpse_csv = _OUTPUTS_BASE / "gpse_backbone" / "aggregated_results.csv"
    _df_gin:  pd.DataFrame | None = None
    _df_gpse: pd.DataFrame | None = None
    _combined_dir = _OUTPUTS_BASE / "combined" / "figures"
    if _gin_csv.exists() and _gpse_csv.exists():
        print("\n── Combined GIN + GPSE — master figures ──")
        _df_gin  = apply_shared_random_init_baseline(pd.read_csv(_gin_csv))
        _df_gpse = apply_shared_random_init_baseline(pd.read_csv(_gpse_csv))
        if not args.no_master:
            for _scope in (None, "probe", "full"):
                _label = "combined" if _scope is None else _scope
                try:
                    plot_all_datasets_combined(
                        _df_gin, _df_gpse, _combined_dir,
                        fmt=args.fmt, scope=_scope,
                    )
                except Exception as exc:
                    print(f"  ERROR combined learning curves ({_label}): {exc}")
        if not args.no_pct_master:
            for _scope in (None, "probe", "full"):
                _label = "combined" if _scope is None else _scope
                try:
                    plot_all_datasets_percentage_combined(
                        _df_gin, _df_gpse, _combined_dir,
                        fmt=args.fmt, scope=_scope,
                    )
                except Exception as exc:
                    print(f"  ERROR combined pct ({_label}): {exc}")
        if not args.no_bars_frac:
            print("\n── Combined per-fraction bar charts ──")
            try:
                _generate_frac_bars_combined(_df_gin, _df_gpse, _combined_dir, fmt=args.fmt)
            except Exception as exc:
                print(f"  ERROR combined bars: {exc}")
    else:
        _missing = [str(p) for p in (_gin_csv, _gpse_csv) if not p.exists()]
        print(f"\n  Skipping combined GIN+GPSE plots (missing: {', '.join(_missing)})")

    if not args.no_bars_frac:
        n_fracs = len(df[COL_FRACTION].dropna().unique())
        print(f"\n── Per-fraction bar charts ({n_fracs} fraction(s) × 6 figures each) ──")
        try:
            _generate_frac_bars(df, output_dir, fmt=args.fmt, datasets=datasets)
        except Exception as exc:
            print(f"  ERROR bars: {exc}")

    pct_dir = output_dir / "percentage_results"

    if not args.no_percentage:
        print(f"\n── Percentage-normalised learning curves ({len(datasets)} dataset(s)) ──")
        for dataset in datasets:
            print(f"[{dataset}]")
            try:
                plot_dataset_percentage(df, dataset, pct_dir, fmt=args.fmt)
            except Exception as exc:
                print(f"  ERROR: {exc}")

    # Load metadata once for all analyses that need it
    need_metadata = not args.no_correlations or not args.no_shift_correlations
    metadata: dict = {}
    metadata_all_splits: dict = {}
    if need_metadata:
        metadata = load_metadata(args.metadata_dir)
        if not metadata:
            print(f"\n  WARNING: no metadata JSON files found in {args.metadata_dir}")

    corr_dir = pct_dir / "correlations"
    n_groups = math.ceil(len(GRAPH_PROPERTIES) / PROPS_PER_PLOT)

    if not args.no_correlations and metadata:
        n_figs = 2 * n_groups   # 2 scopes, fractions pooled
        print(f"\n── Property-correlation plots (fractions pooled, {n_figs} figure(s)) ──")
        try:
            plot_property_correlations(
                df, metadata, corr_dir,
                fmt=args.fmt, datasets=datasets,
                pool_fractions=True,
            )
        except Exception as exc:
            print(f"  ERROR: {exc}")

        if not args.no_frac_avg_correlations:
            frac_avg_dir = corr_dir / "frac-averaged"
            print(f"\n── Property-correlation plots (fraction-averaged, {n_figs} figure(s)) ──")
            try:
                diff_df_corr = _build_diff_records(df, datasets)
                frac_avg_df  = _build_frac_avg_diff_records(diff_df_corr)
                plot_property_correlations(
                    df, metadata, frac_avg_dir,
                    fmt=args.fmt, datasets=datasets,
                    pool_fractions=True,
                    prebuilt_diff_df=frac_avg_df,
                    title_extra="fraction-averaged",
                    fname_prefix="corr_frac_avg",
                )
                for _method in sorted(frac_avg_df["pretrain_method"].unique()):
                    _m_label = _pretrain_label(_method)
                    _m_df = frac_avg_df[frac_avg_df["pretrain_method"] == _method]
                    plot_property_correlations(
                        df, metadata, frac_avg_dir / _method,
                        fmt=args.fmt, datasets=datasets,
                        pool_fractions=True,
                        prebuilt_diff_df=_m_df,
                        title_extra=f"fraction-averaged · {_m_label}",
                        fname_prefix="corr_frac_avg",
                    )
            except Exception as exc:
                print(f"  ERROR: {exc}")

    if not args.no_shift_correlations:
        metadata_all_splits = load_metadata_all_splits(args.metadata_dir)
        if not metadata_all_splits:
            print(f"\n  WARNING: no metadata JSON files found for shift correlations in {args.metadata_dir}")
        else:
            tv_meta, vt_meta = _compute_split_shift_metadata(metadata_all_splits)
            n_figs = 2 * n_groups

            if tv_meta:
                tv_dir = corr_dir / "train-val-shift-correlations"
                print(f"\n── Train-val shift correlation plots (fractions pooled, {n_figs} figure(s)) ──")
                try:
                    plot_property_correlations(
                        df, tv_meta, tv_dir,
                        fmt=args.fmt, datasets=datasets,
                        pool_fractions=True,
                        x_label_suffix=" |train−val|",
                        title_extra="train−val property shift (mean, absolute)",
                        fname_prefix="corr_tv_shift",
                    )
                except Exception as exc:
                    print(f"  ERROR: {exc}")

                if not args.no_frac_avg_correlations:
                    tv_avg_dir = tv_dir / "frac-averaged"
                    print(f"\n── Train-val shift correlation plots (fraction-averaged, {n_figs} figure(s)) ──")
                    try:
                        diff_df_tv  = _build_diff_records(df, datasets)
                        frac_avg_tv = _build_frac_avg_diff_records(diff_df_tv)
                        plot_property_correlations(
                            df, tv_meta, tv_avg_dir,
                            fmt=args.fmt, datasets=datasets,
                            pool_fractions=True,
                            prebuilt_diff_df=frac_avg_tv,
                            x_label_suffix=" |train−val|",
                            title_extra="train−val shift · fraction-averaged",
                            fname_prefix="corr_tv_shift_frac_avg",
                        )
                        for _method in sorted(frac_avg_tv["pretrain_method"].unique()):
                            _m_label = _pretrain_label(_method)
                            _m_df = frac_avg_tv[frac_avg_tv["pretrain_method"] == _method]
                            plot_property_correlations(
                                df, tv_meta, tv_avg_dir / _method,
                                fmt=args.fmt, datasets=datasets,
                                pool_fractions=True,
                                prebuilt_diff_df=_m_df,
                                x_label_suffix=" |train−val|",
                                title_extra=f"train−val shift · fraction-averaged · {_m_label}",
                                fname_prefix="corr_tv_shift_frac_avg",
                            )
                    except Exception as exc:
                        print(f"  ERROR: {exc}")

            if vt_meta:
                vt_dir = corr_dir / "val-test-shift-correlations"
                print(f"\n── Val-test shift correlation plots (fractions pooled, {n_figs} figure(s)) ──")
                try:
                    plot_property_correlations(
                        df, vt_meta, vt_dir,
                        fmt=args.fmt, datasets=datasets,
                        pool_fractions=True,
                        x_label_suffix=" |val−test|",
                        title_extra="val−test property shift (mean, absolute)",
                        fname_prefix="corr_vt_shift",
                    )
                except Exception as exc:
                    print(f"  ERROR: {exc}")

                if not args.no_frac_avg_correlations:
                    vt_avg_dir = vt_dir / "frac-averaged"
                    print(f"\n── Val-test shift correlation plots (fraction-averaged, {n_figs} figure(s)) ──")
                    try:
                        diff_df_vt  = _build_diff_records(df, datasets)
                        frac_avg_vt = _build_frac_avg_diff_records(diff_df_vt)
                        plot_property_correlations(
                            df, vt_meta, vt_avg_dir,
                            fmt=args.fmt, datasets=datasets,
                            pool_fractions=True,
                            prebuilt_diff_df=frac_avg_vt,
                            x_label_suffix=" |val−test|",
                            title_extra="val−test shift · fraction-averaged",
                            fname_prefix="corr_vt_shift_frac_avg",
                        )
                        for _method in sorted(frac_avg_vt["pretrain_method"].unique()):
                            _m_label = _pretrain_label(_method)
                            _m_df = frac_avg_vt[frac_avg_vt["pretrain_method"] == _method]
                            plot_property_correlations(
                                df, vt_meta, vt_avg_dir / _method,
                                fmt=args.fmt, datasets=datasets,
                                pool_fractions=True,
                                prebuilt_diff_df=_m_df,
                                x_label_suffix=" |val−test|",
                                title_extra=f"val−test shift · fraction-averaged · {_m_label}",
                                fname_prefix="corr_vt_shift_frac_avg",
                            )
                    except Exception as exc:
                        print(f"  ERROR: {exc}")

    if not args.no_barplots:
        bars_dir  = output_dir / "improvement_bars"
        fractions = sorted(df[COL_FRACTION].dropna().unique())
        n_figs    = len(fractions) * 2
        print(f"\n── Improvement bar charts ({n_figs} figure(s)) ──")
        try:
            plot_improvement_barplots(df, bars_dir,
                                      fmt=args.fmt, datasets=datasets)
        except Exception as exc:
            print(f"  ERROR: {exc}")

    if not args.no_avg_improvement:
        avg_dir = output_dir / "improvement_bars"
        print("\n── Average-improvement summary figure ──")
        try:
            plot_average_improvement(df, avg_dir,
                                     fmt=args.fmt, datasets=datasets)
        except Exception as exc:
            print(f"  ERROR: {exc}")

    print(f"\nDone. Figures in: {output_dir}")


if __name__ == "__main__":
    main()
