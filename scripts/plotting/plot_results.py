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
DEFAULT_INPUT     = _SCRIPT_DIR / "outputs" / "aggregated_results.csv"
DEFAULT_OUTPUT_DIR = _SCRIPT_DIR / "outputs" / "figures"

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
    """Load *.json files from metadata_dir. Returns {dataset_name: train_props}."""
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
        result[p.stem] = {k: v for k, v in train.items()
                          if k not in ("num_graphs", "total_num_nodes")}
    return result


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


def plot_property_correlations(
    df: pd.DataFrame,
    metadata: dict[str, dict],
    output_dir: Path,
    fmt: str = "png",
    datasets: list[str] | None = None,
) -> None:
    """For each (fraction, scope) and each group of 6 graph properties: one 2×3 figure.

    X-axis : graph property value (mean or std, from train-split metadata).
    Y-axis : sign·(finetune − random_init) / σ_dataset
             [dataset-std-normalised performance difference, ± propagated stderr].
    Each point: one (dataset, pretraining_method) combo.
    Colour  : pretraining method.
    """
    _datasets = datasets or sorted(df[COL_DATASET].dropna().unique())

    diff_df = _build_diff_records(df, _datasets)
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

    # Property groups
    prop_groups = [GRAPH_PROPERTIES[i:i + PROPS_PER_PLOT]
                   for i in range(0, len(GRAPH_PROPERTIES), PROPS_PER_PLOT)]

    fractions = sorted(diff_df["fraction"].unique())
    scopes    = sorted(diff_df["scope"].unique())

    output_dir.mkdir(parents=True, exist_ok=True)

    for frac in fractions:
        for scope in scopes:
            subset = diff_df[(diff_df["fraction"] == frac) & (diff_df["scope"] == scope)]
            if subset.empty:
                continue

            scope_label = "Probe" if scope == "probe" else "Full FT"

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

                        ax.set_xlabel(prop_label, fontsize=9)
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

                    fig.suptitle(
                        f"Pretraining advantage vs graph properties  "
                        f"·  {scope_label}  ·  fraction = {frac:g}",
                        fontsize=12, fontweight="bold", y=0.99,
                    )

                    fname = f"corr_frac{frac:g}_{scope}_props{gi + 1}.{fmt}"
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


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot aggregated finetuning results.")
    p.add_argument(
        "--input", type=Path, default=DEFAULT_INPUT,
        help="Path to aggregated_results.csv",
    )
    p.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help="Directory to write figures into.",
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
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} rows from {args.input}")
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
                plot_dataset(df, dataset, args.output_dir, fmt=args.fmt)
            except Exception as exc:
                print(f"  ERROR: {exc}")

    if not args.no_master:
        print(f"\n── Master all-datasets learning-curve figure(s) ──")
        for master_scope in (None, "probe", "full"):
            label = "combined" if master_scope is None else master_scope
            try:
                plot_all_datasets(
                    df, args.output_dir, fmt=args.fmt,
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
                    df, args.output_dir, fmt=args.fmt,
                    datasets=datasets, scope=master_scope,
                )
            except Exception as exc:
                print(f"  ERROR ({label}): {exc}")

    pct_dir = args.output_dir / "percentage_results"

    if not args.no_percentage:
        print(f"\n── Percentage-normalised learning curves ({len(datasets)} dataset(s)) ──")
        for dataset in datasets:
            print(f"[{dataset}]")
            try:
                plot_dataset_percentage(df, dataset, pct_dir, fmt=args.fmt)
            except Exception as exc:
                print(f"  ERROR: {exc}")

    # Load metadata once for all analyses that need it
    metadata: dict = {}
    if not args.no_correlations:
        metadata = load_metadata(args.metadata_dir)
        if not metadata:
            print(f"\n  WARNING: no metadata JSON files found in {args.metadata_dir}")

    if not args.no_correlations and metadata:
        corr_dir  = pct_dir / "correlations"
        fractions = sorted(df[COL_FRACTION].dropna().unique())
        n_groups  = math.ceil(len(GRAPH_PROPERTIES) / PROPS_PER_PLOT)
        n_figs    = len(fractions) * 2 * n_groups
        print(f"\n── Property-correlation plots ({n_figs} figure(s)) ──")
        try:
            plot_property_correlations(df, metadata, corr_dir,
                                       fmt=args.fmt, datasets=datasets)
        except Exception as exc:
            print(f"  ERROR: {exc}")

    if not args.no_barplots:
        bars_dir  = args.output_dir / "improvement_bars"
        fractions = sorted(df[COL_FRACTION].dropna().unique())
        n_figs    = len(fractions) * 2
        print(f"\n── Improvement bar charts ({n_figs} figure(s)) ──")
        try:
            plot_improvement_barplots(df, bars_dir,
                                      fmt=args.fmt, datasets=datasets)
        except Exception as exc:
            print(f"  ERROR: {exc}")

    if not args.no_avg_improvement:
        avg_dir = args.output_dir / "improvement_bars"
        print("\n── Average-improvement summary figure ──")
        try:
            plot_average_improvement(df, avg_dir,
                                     fmt=args.fmt, datasets=datasets)
        except Exception as exc:
            print(f"  ERROR: {exc}")

    print(f"\nDone. Figures in: {args.output_dir}")


if __name__ == "__main__":
    main()
