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
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
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
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} rows from {args.input}")

    if COL_DATASET not in df.columns:
        raise ValueError(
            f"Expected column {COL_DATASET!r} not found. "
            f"Available columns: {list(df.columns)}"
        )

    datasets = args.datasets or sorted(df[COL_DATASET].dropna().unique())
    print(f"Plotting {len(datasets)} dataset(s): {datasets}\n")

    for dataset in datasets:
        print(f"[{dataset}]")
        try:
            plot_dataset(df, dataset, args.output_dir, fmt=args.fmt)
        except Exception as exc:
            print(f"  ERROR: {exc}")

    print(f"\nDone. Figures in: {args.output_dir}")


if __name__ == "__main__":
    main()
