#!/usr/bin/env python3
"""run_baseline_analysis.py

Reads finished runs from the ``gpse_backbone_supervised_sweep`` W&B project,
identifies the best hyperparameter configuration per dataset (by val metric),
and:

1. **JSON update** – writes ``GPSE_backbone_hyperparam_best_test`` into each
   dataset's existing metadata JSON file (the files produced by run_analysis.py).

2. **Plots** – for each varied hyperparameter (hidden_channels, num_layers,
   weight_decay, lr) produces a bar chart showing how often each value was the
   best setting across datasets.  Saved as
   ``<output_dir>/baseline_best_hyperparam_counts.png``.

Usage
-----
    python scripts/metadata_analysis/run_baseline_analysis.py
    python scripts/metadata_analysis/run_baseline_analysis.py \\
        --entity  my-entity \\
        --project gpse_backbone_supervised_sweep \\
        --output-dir scripts/metadata_analysis/outputs
    # Only regenerate plots, skip JSON updates:
    python scripts/metadata_analysis/run_baseline_analysis.py --no-json-update
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_SCRIPT_DIR   = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parents[1]

# ── Metric direction ──────────────────────────────────────────────────────────
# Substrings that indicate a metric where LOWER is better.
_LOWER_IS_BETTER_SUBSTRINGS = {"mae", "mse", "rmse", "loss", "cross_entropy", "error"}

def _higher_is_better(monitor_metric: str, mode: str | None = None) -> bool:
    """True when a larger value of *monitor_metric* is better."""
    if mode is not None:
        return mode.strip().lower() == "max"
    ml = monitor_metric.lower()
    return not any(s in ml for s in _LOWER_IS_BETTER_SUBSTRINGS)


# ── Varied-param metadata ─────────────────────────────────────────────────────
# Only these params are included in the bar-chart plots (not pretraining/seed).
_PLOT_PARAMS: dict[str, str] = {
    "varied_param_hidden_channels": "hidden_channels",
    "varied_param_num_layers":      "num_layers",
    "varied_param_weight_decay":    "weight_decay",
    "varied_param_lr":              "lr",
}


def _try_numeric(s: str):
    """Try to parse *s* as int or float; return original string on failure."""
    try:
        iv = int(s)
        return iv
    except (ValueError, TypeError):
        pass
    try:
        return float(s)
    except (ValueError, TypeError):
        return s


def _sort_key(s: str):
    """Sort key that places numeric strings before non-numeric strings."""
    v = _try_numeric(s)
    return (0, v) if isinstance(v, (int, float)) else (1, s)


# ─────────────────────────────────────────────────────────────────────────────
# W&B data fetching
# ─────────────────────────────────────────────────────────────────────────────

def fetch_runs(entity: str, project: str) -> list[dict]:
    """Download all *finished* runs from the W&B project.

    Returns a list of plain dicts, one per run, with:
        dataset, monitor_metric, higher_better,
        val_score, test_score, run_id,
        varied_param_hidden_channels, varied_param_num_layers,
        varied_param_weight_decay, varied_param_lr, varied_param_data_seed
    """
    import wandb

    api  = wandb.Api(timeout=120)
    path = f"{entity}/{project}"
    print(f"  Fetching runs from {path} …")
    raw_runs = api.runs(path, filters={"state": "finished"})

    records: list[dict] = []
    skipped = 0

    for run in raw_runs:
        cfg     = run.config
        summary = run.summary._json_dict

        # ── Dataset name ──────────────────────────────────────────────────────
        dataset = cfg.get("varied_param_dataset")
        if dataset is None:
            # Fall back to nested config (runs before varied_param_* was added)
            try:
                dataset = (
                    cfg["dataset"]["loader"]["parameters"]["data_name"]
                )
            except (KeyError, TypeError):
                skipped += 1
                continue
        dataset = str(dataset)

        # ── Monitor metric & direction ─────────────────────────────────────────
        try:
            monitor_metric = cfg["dataset"]["parameters"]["monitor_metric"]
        except (KeyError, TypeError):
            monitor_metric = "auroc"  # sensible default

        try:
            mode = cfg["callbacks"]["early_stopping"]["mode"]
        except (KeyError, TypeError):
            mode = None

        higher = _higher_is_better(monitor_metric, mode)

        # ── Performance scores ────────────────────────────────────────────────
        val_key  = f"best_epoch/val/{monitor_metric}"
        test_key = f"test_best_rerun/{monitor_metric}"

        val_score = summary.get(val_key)
        if val_score is None:
            skipped += 1
            continue  # run didn't produce metrics — skip

        test_score = summary.get(test_key)

        record: dict = {
            "run_id":          run.id,
            "run_name":        run.name,
            "dataset":         dataset,
            "monitor_metric":  monitor_metric,
            "higher_better":   higher,
            "val_score":       float(val_score),
            "test_score":      float(test_score) if test_score is not None else None,
        }

        # ── Varied hyperparameters ────────────────────────────────────────────
        for key in list(_PLOT_PARAMS) + ["varied_param_data_seed"]:
            record[key] = cfg.get(key)

        records.append(record)

    print(f"  → {len(records)} valid run(s)  ({skipped} skipped — no metrics)")
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Best-run selection
# ─────────────────────────────────────────────────────────────────────────────

def select_best_per_dataset(records: list[dict]) -> dict[str, dict]:
    """Return the best-val run for each dataset."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        grouped[r["dataset"]].append(r)

    best: dict[str, dict] = {}
    for dataset, runs in grouped.items():
        higher = runs[0]["higher_better"]
        best[dataset] = (
            max(runs, key=lambda r: r["val_score"])
            if higher
            else min(runs, key=lambda r: r["val_score"])
        )
    return best


# ─────────────────────────────────────────────────────────────────────────────
# JSON update
# ─────────────────────────────────────────────────────────────────────────────

def update_json_files(best_per_dataset: dict[str, dict], output_dir: Path) -> None:
    """Write ``GPSE_backbone_hyperparam_best_test`` into each dataset JSON."""
    updated, missing = 0, 0
    for dataset, best in best_per_dataset.items():
        json_path = output_dir / f"{dataset}.json"
        if not json_path.exists():
            print(f"  WARNING: {json_path} not found — skipping.")
            missing += 1
            continue

        with open(json_path) as fh:
            data = json.load(fh)

        m = best["monitor_metric"]
        ts = best["test_score"]
        data["GPSE_backbone_hyperparam_best_test"] = {
            "test_score":          ts,
            "val_score":           best["val_score"],
            "monitor_metric":      m,
            "higher_better":       best["higher_better"],
            "best_run_id":         best["run_id"],
            "best_run_name":       best["run_name"],
            "best_hidden_channels": _try_numeric(str(best.get("varied_param_hidden_channels", ""))),
            "best_num_layers":      _try_numeric(str(best.get("varied_param_num_layers", ""))),
            "best_weight_decay":    _try_numeric(str(best.get("varied_param_weight_decay", ""))),
            "best_lr":              _try_numeric(str(best.get("varied_param_lr", ""))),
        }

        with open(json_path, "w") as fh:
            json.dump(data, fh, indent=2)

        ts_str = f"{ts:.4f}" if ts is not None else "N/A"
        print(
            f"  Updated {json_path.name:<35s}  "
            f"val_{m}={best['val_score']:.4f}  test_{m}={ts_str}"
        )
        updated += 1

    print(f"  → {updated} JSON(s) updated, {missing} missing.")


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_best_hyperparam_counts(
    best_per_dataset: dict[str, dict],
    all_records:      list[dict],
    output_dir:       Path,
) -> None:
    """Bar charts: for each varied hyperparam, # datasets where each value was best."""

    # Collect all unique values per param seen across ALL runs
    all_values: dict[str, set[str]] = {k: set() for k in _PLOT_PARAMS}
    for r in all_records:
        for key in _PLOT_PARAMS:
            v = r.get(key)
            if v is not None:
                all_values[key].add(str(v))

    # Count best occurrences
    best_counts: dict[str, dict[str, int]] = {k: defaultdict(int) for k in _PLOT_PARAMS}
    for _ds, best in best_per_dataset.items():
        for key in _PLOT_PARAMS:
            v = best.get(key)
            if v is not None:
                best_counts[key][str(v)] += 1

    # Only plot params that have at least one value
    active_params = [(k, lbl) for k, lbl in _PLOT_PARAMS.items() if all_values[k]]
    if not active_params:
        print("  WARNING: no varied-param data found — skipping plots.")
        return

    n_params   = len(active_params)
    n_datasets = len(best_per_dataset)
    palette    = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    fig, axes = plt.subplots(
        1, n_params,
        figsize=(max(4.5, 4.0 * n_params), 5),
        squeeze=False,
    )
    axes = axes[0]

    for ax, color, (key, label) in zip(axes, palette, active_params):
        values = sorted(all_values[key], key=_sort_key)
        counts = [best_counts[key].get(v, 0) for v in values]

        bars = ax.bar(
            range(len(values)), counts,
            color=color, edgecolor="white", linewidth=0.8, zorder=3,
        )

        # Annotate bars
        for bar, cnt in zip(bars, counts):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.08,
                str(cnt),
                ha="center", va="bottom",
                fontsize=11, fontweight="bold",
            )

        ax.set_xticks(range(len(values)))
        ax.set_xticklabels(values, rotation=30, ha="right", fontsize=10)
        ax.set_title(label, fontsize=12, fontweight="bold", pad=8)
        ax.set_xlabel(label, fontsize=10)
        ax.set_ylim(0, n_datasets + 1.2)
        ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.set_ylabel("# datasets where best" if ax is axes[0] else "", fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.3, linestyle="--", zorder=0)

    fig.suptitle(
        f"GPSE backbone supervised: best hyperparameter setting per dataset\n"
        f"(N = {n_datasets} dataset{'s' if n_datasets != 1 else ''},"
        f"  total runs = {len(all_records)})",
        fontsize=12, y=1.03,
    )
    plt.tight_layout()

    out_path = output_dir / "baseline_best_hyperparam_counts.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Analyse gpse_backbone_supervised_sweep W&B results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--entity",
        default="louis-van-langendonck-universitat-polit-cnica-de-catalunya",
        help="W&B entity.",
    )
    parser.add_argument(
        "--project",
        default="gpse_backbone_supervised_sweep",
        help="W&B project name.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(_SCRIPT_DIR / "outputs"),
        help="Directory with metadata JSONs; PNGs are also written here.",
    )
    parser.add_argument(
        "--no-json-update",
        action="store_true",
        help="Skip updating JSON files (only regenerate plots).",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args       = _parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 62)
    print("  GPSE Backbone Supervised — Baseline Analysis")
    print("=" * 62)
    print(f"  Entity    : {args.entity}")
    print(f"  Project   : {args.project}")
    print(f"  Output    : {output_dir}")
    print("=" * 62)

    # ── Fetch ─────────────────────────────────────────────────────────────────
    records = fetch_runs(args.entity, args.project)
    if not records:
        print("ERROR: no valid runs found.", file=sys.stderr)
        sys.exit(1)

    # ── Select best per dataset ───────────────────────────────────────────────
    best_per_dataset = select_best_per_dataset(records)

    print(f"\nBest config per dataset ({len(best_per_dataset)} datasets):")
    print(f"  {'Dataset':<35s}  {'val':>8s}  {'test':>8s}  h    L   wd       lr")
    print("  " + "-" * 72)
    for ds, best in sorted(best_per_dataset.items()):
        ts_str = f"{best['test_score']:.4f}" if best["test_score"] is not None else "  N/A "
        print(
            f"  {ds:<35s}  {best['val_score']:>8.4f}  {ts_str:>8s}"
            f"  {str(best.get('varied_param_hidden_channels','?')):<4s}"
            f" {str(best.get('varied_param_num_layers','?')):<3s}"
            f" {str(best.get('varied_param_weight_decay','?')):<8s}"
            f" {str(best.get('varied_param_lr','?'))}"
        )

    # ── Update JSONs ──────────────────────────────────────────────────────────
    if not args.no_json_update:
        print("\nUpdating metadata JSON files …")
        update_json_files(best_per_dataset, output_dir)

    # ── Plot ──────────────────────────────────────────────────────────────────
    print("\nGenerating plots …")
    plot_best_hyperparam_counts(best_per_dataset, records, output_dir)

    print("\n✔ Done.")


if __name__ == "__main__":
    main()
