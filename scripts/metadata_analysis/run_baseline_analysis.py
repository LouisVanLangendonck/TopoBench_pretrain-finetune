#!/usr/bin/env python3
"""run_baseline_analysis.py

Reads finished runs from the GPSE backbone supervised W&B sweep project(s),
identifies the best hyperparameter configuration per dataset (by val metric),
and:

1. **JSON update** – writes ``GPSE_backbone_hyperparam_best_test`` into each
   dataset's metadata JSON file (the files produced by run_analysis.py).
   Missing JSONs are created with minimal metadata; existing entries are left
   untouched.

2. **Plots** – for each varied hyperparameter (hidden_channels, num_layers,
   weight_decay, lr) produces a bar chart showing how often each value was the
   best setting across datasets.  Saved as
   ``<output_dir>/baseline_best_hyperparam_counts[_transductive].png``.

Supported sweeps
----------------
* **inductive**    – ``gpse_backbone_supervised_sweep`` (graph-level datasets)
* **transductive** – ``gpse_backbone_supervised_sweep_transductive``
                     (cocitation_cora, cocitation_pubmed, minesweeper, roman_empire)

Usage
-----
    python scripts/metadata_analysis/run_baseline_analysis.py
    python scripts/metadata_analysis/run_baseline_analysis.py --setting both
    python scripts/metadata_analysis/run_baseline_analysis.py \\
        --entity  my-entity \\
        --project gpse_backbone_supervised_sweep \\
        --output-dir scripts/metadata_analysis/outputs
    python scripts/metadata_analysis/run_baseline_analysis.py --setting transductive
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

# ── Sweep presets ─────────────────────────────────────────────────────────────
_SWEEP_PRESETS: dict[str, dict[str, str]] = {
    "inductive": {
        "project":    "gpse_backbone_supervised_sweep",
        "plot_stem":  "baseline_best_hyperparam_counts",
        "plot_title": "GPSE backbone supervised (inductive)",
    },
    "transductive": {
        "project":    "gpse_backbone_supervised_sweep_transductive",
        "plot_stem":  "baseline_best_hyperparam_counts_transductive",
        "plot_title": "GPSE backbone supervised (transductive)",
    },
}

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


def _step(msg: str) -> None:
    """Print a visible phase banner."""
    print(f"\n▶ {msg}")


def _progress(current: int, total: int, label: str) -> str:
    """Return a short ``[i/n]`` progress prefix."""
    return f"[{current}/{total}] {label}"


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
    print(f"  Connecting to W&B: {path}")
    print("  Filter: state=finished")
    raw_runs = api.runs(path, filters={"state": "finished"})
    print("  Downloading run configs and summaries …")

    records: list[dict] = []
    skipped_no_dataset = 0
    skipped_no_metrics = 0
    n_seen = 0

    for run in raw_runs:
        n_seen += 1
        if n_seen == 1 or n_seen % 25 == 0:
            print(f"    … processed {n_seen} run(s) so far "
                  f"({len(records)} valid, {skipped_no_dataset + skipped_no_metrics} skipped)")

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
                skipped_no_dataset += 1
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
            skipped_no_metrics += 1
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

    skipped = skipped_no_dataset + skipped_no_metrics
    print(f"  Finished scanning {n_seen} run(s)")
    print(f"  → {len(records)} valid run(s)")
    if skipped_no_dataset:
        print(f"    · {skipped_no_dataset} skipped — could not resolve dataset name")
    if skipped_no_metrics:
        print(f"    · {skipped_no_metrics} skipped — missing val metric in summary")
    if skipped == 0:
        print("    · 0 skipped")
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Best-run selection
# ─────────────────────────────────────────────────────────────────────────────

def select_best_per_dataset(records: list[dict]) -> dict[str, dict]:
    """Return the best-val run for each dataset."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        grouped[r["dataset"]].append(r)

    print(f"  Grouped into {len(grouped)} dataset(s):")
    for dataset in sorted(grouped):
        n = len(grouped[dataset])
        metric = grouped[dataset][0]["monitor_metric"]
        direction = "↑ higher" if grouped[dataset][0]["higher_better"] else "↓ lower"
        print(f"    · {dataset}: {n} run(s), metric={metric} ({direction} is better)")

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

_JSON_KEY = "GPSE_backbone_hyperparam_best_test"


def _build_hyperparam_entry(best: dict) -> dict:
    """Build the ``GPSE_backbone_hyperparam_best_test`` payload from a best run."""
    m = best["monitor_metric"]
    return {
        "test_score":          best["test_score"],
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


def update_json_files(best_per_dataset: dict[str, dict], output_dir: Path) -> None:
    """Write ``GPSE_backbone_hyperparam_best_test`` into each dataset JSON.

    - Creates the JSON file if it does not exist yet (minimal metadata only).
    - Skips files that already contain ``GPSE_backbone_hyperparam_best_test``.
    """
    updated, created, skipped = 0, 0, 0
    datasets = sorted(best_per_dataset)
    n_total = len(datasets)
    print(f"  Output directory: {output_dir.resolve()}")
    print(f"  Processing {n_total} dataset JSON(s) …")

    for i, dataset in enumerate(datasets, 1):
        best = best_per_dataset[dataset]
        json_path = output_dir / f"{dataset}.json"
        prefix = _progress(i, n_total, dataset)

        if json_path.exists():
            with open(json_path) as fh:
                data = json.load(fh)
            if _JSON_KEY in data:
                print(f"  {prefix}  SKIP  — {_JSON_KEY} already in {json_path.name}")
                skipped += 1
                continue
            print(f"  {prefix}  UPDATE — adding {_JSON_KEY} to existing {json_path.name}")
            action = "Updated"
        else:
            print(f"  {prefix}  CREATE — new file {json_path.name} (minimal metadata)")
            data = {"dataset": f"graph/{dataset}"}
            data_seed = best.get("varied_param_data_seed")
            if data_seed is not None:
                data["data_seed"] = _try_numeric(str(data_seed))
            action = "Created"

        entry = _build_hyperparam_entry(best)
        data[_JSON_KEY] = entry

        with open(json_path, "w") as fh:
            json.dump(data, fh, indent=2)

        m = best["monitor_metric"]
        ts = best["test_score"]
        ts_str = f"{ts:.4f}" if ts is not None else "N/A"
        print(
            f"           ✔ {action}  val_{m}={best['val_score']:.4f}  "
            f"test_{m}={ts_str}  run={best['run_name']}"
        )
        if action == "Created":
            created += 1
        else:
            updated += 1

    print(f"  JSON summary: {updated} updated, {created} created, {skipped} skipped (already present).")


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_best_hyperparam_counts(
    best_per_dataset: dict[str, dict],
    all_records:      list[dict],
    output_dir:       Path,
    *,
    plot_stem:  str = "baseline_best_hyperparam_counts",
    plot_title: str = "GPSE backbone supervised",
) -> None:
    """Bar charts: for each varied hyperparam, # datasets where each value was best."""

    print(f"  Building plot: {plot_stem}.png")
    print(f"    · {len(best_per_dataset)} dataset(s), {len(all_records)} total run(s)")

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

    print(f"    · plotting {len(active_params)} hyperparameter(s): "
          f"{', '.join(lbl for _, lbl in active_params)}")

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
        f"{plot_title}: best hyperparameter setting per dataset\n"
        f"(N = {n_datasets} dataset{'s' if n_datasets != 1 else ''},"
        f"  total runs = {len(all_records)})",
        fontsize=12, y=1.03,
    )
    plt.tight_layout()

    out_path = output_dir / f"{plot_stem}.png"
    print(f"  Saving figure …")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✔ Saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_sweep_jobs(args) -> list[dict[str, str]]:
    """Return the list of sweep jobs to run from CLI args."""
    if args.project is not None:
        return [{
            "setting":    "custom",
            "project":    args.project,
            "plot_stem":  "baseline_best_hyperparam_counts",
            "plot_title": f"GPSE backbone supervised ({args.project})",
        }]

    setting = args.setting
    if setting == "both":
        return [_SWEEP_PRESETS[k] | {"setting": k} for k in ("inductive", "transductive")]
    return [_SWEEP_PRESETS[setting] | {"setting": setting}]


def _analyse_sweep(
    entity: str,
    job: dict[str, str],
    output_dir: Path,
    *,
    update_json: bool,
    job_index: int = 1,
    job_total: int = 1,
) -> bool:
    """Run the full analysis pipeline for one W&B project.  Returns True on success."""
    project    = job["project"]
    plot_stem  = job["plot_stem"]
    plot_title = job["plot_title"]
    setting    = job["setting"]

    print("\n" + "=" * 62)
    print(f"  Sweep {job_index}/{job_total}: {setting}")
    print("=" * 62)
    print(f"  Entity    : {entity}")
    print(f"  Project   : {project}")
    print(f"  Output    : {output_dir.resolve()}")
    print(f"  JSON mode : {'update/create' if update_json else 'disabled (--no-json-update)'}")
    print("=" * 62)

    _step("Step 1/4 — Fetch W&B runs")
    records = fetch_runs(entity, project)
    if not records:
        print(f"  ✗ No valid runs found for {project} — skipping this sweep.", file=sys.stderr)
        return False

    _step("Step 2/4 — Select best config per dataset (by val metric)")
    best_per_dataset = select_best_per_dataset(records)

    print("\n  Best runs:")
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

    if update_json:
        _step("Step 3/4 — Update metadata JSON files")
        update_json_files(best_per_dataset, output_dir)
    else:
        _step("Step 3/4 — Update metadata JSON files (skipped)")

    _step("Step 4/4 — Generate hyperparameter count plots")
    plot_best_hyperparam_counts(
        best_per_dataset,
        records,
        output_dir,
        plot_stem=plot_stem,
        plot_title=plot_title,
    )
    print(f"\n  ✔ Sweep {job_index}/{job_total} ({setting}) complete.")
    return True


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Analyse GPSE backbone supervised sweep W&B results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--entity",
        default="louis-van-langendonck-universitat-polit-cnica-de-catalunya",
        help="W&B entity.",
    )
    parser.add_argument(
        "--setting",
        choices=["inductive", "transductive", "both"],
        default="both",
        help=(
            "Which supervised sweep to analyse.  "
            "'both' runs inductive + transductive (default).  "
            "Ignored when --project is set."
        ),
    )
    parser.add_argument(
        "--project",
        default=None,
        help="Override W&B project name (disables --setting).",
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

    jobs     = _resolve_sweep_jobs(args)
    n_jobs   = len(jobs)
    n_ok     = 0
    n_skip   = 0

    print("=" * 62)
    print("  GPSE Backbone Supervised — Baseline Analysis")
    print("=" * 62)
    print(f"  Entity      : {args.entity}")
    print(f"  Setting     : {args.setting if args.project is None else 'custom (--project)'}")
    print(f"  Output dir  : {output_dir.resolve()}")
    print(f"  JSON update : {'yes' if not args.no_json_update else 'no (--no-json-update)'}")
    print(f"  Sweeps      : {n_jobs}")
    for i, job in enumerate(jobs, 1):
        print(f"    {i}. {job['setting']} → {job['project']}")
    print("=" * 62)

    for i, job in enumerate(jobs, 1):
        if _analyse_sweep(
            args.entity,
            job,
            output_dir,
            update_json=not args.no_json_update,
            job_index=i,
            job_total=n_jobs,
        ):
            n_ok += 1
        else:
            n_skip += 1

    print("\n" + "=" * 62)
    if n_ok == 0:
        print("  ✗ FAILED — no valid runs found for any requested sweep.")
        print("=" * 62)
        sys.exit(1)

    print(f"  ✔ All done — {n_ok}/{n_jobs} sweep(s) analysed"
          + (f", {n_skip} skipped" if n_skip else ""))
    print("=" * 62)


if __name__ == "__main__":
    main()
