#!/usr/bin/env python3
"""Feature and structural importance analysis via gpse_backbone ablations.

For each dataset listed in the config YAML the script runs four supervised
training experiments with a fresh ``gpse_backbone``:

  1. Baseline          – real features + real graph structure
  2. Random features   – i.i.d. Gaussian node features, real structure
  3. Shuffled edges    – real features, randomly rewired graph
  4. Both              – Gaussian features + randomly rewired graph (pure noise)

The test-set performance for each run (using the dataset's own monitor metric)
is saved to the JSON output file.  Feature and structural importance scores
are then computed as:

    worst = both_ablation_performance  (least informative)
    best  = baseline_performance       (most informative)

    PERCENTAGE(x) = (x − worst) / (best − worst)

    task_feature_importance    = 1 − PERCENTAGE(random_features_performance)
    task_structural_importance = 1 − PERCENTAGE(shuffled_edges_performance)

Additive / incremental behaviour
---------------------------------
Results are *merged* into existing JSON files under
``scripts/metadata_analysis/outputs/<DatasetName>.json`` (same location as
run_analysis.py).  Keys already present in the file that are not re-computed
in this run are preserved, so you can run run_analysis.py and run_importance.py
independently and still accumulate a single consistent JSON per dataset.

Usage
-----
    # Default config:
    python scripts/metadata_analysis/run_importance.py

    # Override output directory:
    python scripts/metadata_analysis/run_importance.py --output-dir path/to/out

    # Single dataset:
    python scripts/metadata_analysis/run_importance.py --datasets graph/MUTAG

    # Dry-run (list what would be processed, then exit):
    python scripts/metadata_analysis/run_importance.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import yaml

# ── project root on sys.path ─────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Local imports (same directory)
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from compute_importance import compute_dataset_depth, compute_dataset_importance  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Merge helper  (mirrors run_analysis._merge_result)
# ─────────────────────────────────────────────────────────────────────────────

def _merge_result(existing: dict, new: dict) -> dict:
    """Merge *new* top-level keys into *existing*, preserving un-touched keys.

    - Standard metadata (dataset, model, pretraining, data_seed): refreshed.
    - Split dicts (train / val / test): per-key merged (existing wins for keys
      absent from *new*, new wins otherwise).
    - Any other top-level key in *new* (e.g. importance scores): written
      directly into *merged*, overriding the existing value.
    """
    merged = dict(existing)
    _metadata = {"dataset", "model", "pretraining", "data_seed"}
    _splits = {"train", "val", "test"}
    _handled = _metadata | _splits

    for k in _metadata:
        if k in new:
            merged[k] = new[k]
    for split in _splits:
        if split in new:
            merged[split] = {**existing.get(split, {}), **new[split]}
    for k, v in new.items():
        if k not in _handled:
            merged[k] = v
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run gpse_backbone ablations to compute feature/structural importance.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        default=str(_SCRIPT_DIR / "config.yaml"),
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override output_dir from the config.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        metavar="DATASET",
        help="Override datasets list (e.g. graph/IMDB-BINARY graph/MUTAG).",
    )
    parser.add_argument(
        "--data-seed",
        type=int,
        default=None,
        help="Override data_seed from config.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Override device (e.g. cpu, cuda:0).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be processed, then exit.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)

    # ── Load YAML config ──────────────────────────────────────────────────────
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)

    imp_cfg = cfg.get("importance_analysis", {})

    datasets   = args.datasets  or cfg.get("datasets", [])
    data_seed  = args.data_seed if args.data_seed is not None else cfg.get("data_seed", 0)
    output_dir = Path(
        args.output_dir
        or cfg.get("output_dir", str(_SCRIPT_DIR / "outputs"))
    )

    run_feat_struct = imp_cfg.get("feature_and_structural_importance", True)
    run_task_depth  = imp_cfg.get("task_depth", False)

    model         = imp_cfg.get("model",       "graph/gpse_backbone")
    pretraining   = imp_cfg.get("pretraining", "none")
    max_epochs    = imp_cfg.get("max_epochs",  100)
    patience      = imp_cfg.get("patience",    20)
    device        = args.device or imp_cfg.get("device", "cpu")
    train_seed    = imp_cfg.get("train_seed",  42)
    ablation_seed = imp_cfg.get("ablation_seed", 42)
    max_layers    = imp_cfg.get("task_depth_max_layers", 8)
    structural_feature_datasets = set(
        ds.split("/")[-1]
        for ds in imp_cfg.get("structural_feature_datasets", ["IMDB-BINARY", "IMDB-MULTI", "REDDIT-BINARY"])
    )

    train_cfg = {
        "max_epochs": max_epochs,
        "patience":   patience,
        "seed":       train_seed,
        "device":     device,
    }

    if not datasets:
        print("ERROR: no datasets specified.", file=sys.stderr)
        sys.exit(1)

    if not run_feat_struct and not run_task_depth:
        print("WARNING: both feature_and_structural_importance and task_depth are disabled — nothing to do.", file=sys.stderr)
        return

    print("=" * 60)
    print("  Importance Analysis")
    print("=" * 60)
    print(f"  Config    : {config_path}")
    print(f"  Model     : {model}")
    print(f"  Pretrain  : {pretraining}")
    print(f"  Data seed : {data_seed}")
    print(f"  Train seed: {train_seed}")
    print(f"  Device    : {device}")
    print(f"  Epochs    : {max_epochs}  patience={patience}")
    print(f"  Output    : {output_dir}")
    print(f"  Analyses  : feat+struct={run_feat_struct}  task_depth={run_task_depth}" +
          (f"  (max_layers={max_layers})" if run_task_depth else ""))
    if run_feat_struct:
        print(f"  Skip ablations for: {sorted(structural_feature_datasets)}")
    print(f"  Datasets  : {len(datasets)}")
    for ds in datasets:
        print(f"    - {ds}")
    print("=" * 60)

    if args.dry_run:
        print("Dry run — exiting without processing.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Per-dataset loop ──────────────────────────────────────────────────────
    successes, failures = [], []

    for i, dataset in enumerate(datasets, 1):
        dataset_name = dataset.split("/")[-1]
        print(f"\n[{i}/{len(datasets)}] {dataset}")
        print("-" * 60)
        t0 = time.perf_counter()

        result: dict = {}
        dataset_failed = False

        # ── Feature / structural importance ───────────────────────────────────
        if run_feat_struct:
            try:
                r = compute_dataset_importance(
                    dataset=dataset,
                    model=model,
                    pretraining=pretraining,
                    data_seed=data_seed,
                    train_cfg=train_cfg,
                    structural_feature_datasets=structural_feature_datasets,
                    ablation_seed=ablation_seed,
                )
                result.update(r)
                print(
                    f"  feat+struct → "
                    f"feature_importance={r.get('task_feature_importance')}  "
                    f"structural_importance={r.get('task_structural_importance')}"
                )
            except Exception as exc:
                print(f"  ERROR (feat+struct) {dataset}: {exc}")
                traceback.print_exc()
                failures.append((dataset, f"feat+struct: {exc}"))
                dataset_failed = True

        # ── Task depth ────────────────────────────────────────────────────────
        if run_task_depth:
            try:
                r = compute_dataset_depth(
                    dataset=dataset,
                    model=model,
                    pretraining=pretraining,
                    data_seed=data_seed,
                    train_cfg=train_cfg,
                    max_layers=max_layers,
                    ablation_seed=ablation_seed,
                )
                result.update(r)
                print(f"  task_depth  → {r.get('task_depth')}")
            except Exception as exc:
                print(f"  ERROR (task_depth) {dataset}: {exc}")
                traceback.print_exc()
                if not dataset_failed:
                    failures.append((dataset, f"task_depth: {exc}"))
                dataset_failed = True

        if not result:
            continue

        print(f"  Total elapsed: {time.perf_counter() - t0:.1f}s")

        # ── Merge into existing JSON ──────────────────────────────────────────
        out_path = output_dir / f"{dataset_name}.json"
        if out_path.exists():
            try:
                with open(out_path) as fh:
                    existing = json.load(fh)
                result = _merge_result(existing, result)
                action = "Updated (merged)"
            except Exception as exc:
                print(
                    f"  WARNING: could not read {out_path} — will overwrite. ({exc})"
                )
                action = "Overwritten (read error)"
        else:
            action = "Created"

        with open(out_path, "w") as fh:
            json.dump(result, fh, indent=2)
        print(f"  {action} → {out_path}")
        if not dataset_failed:
            successes.append(dataset)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  Done.  {len(successes)} succeeded, {len(failures)} failed.")
    if failures:
        print("  Failed datasets:")
        for ds, err in failures:
            print(f"    - {ds}: {err}")
    print("=" * 60)


if __name__ == "__main__":
    main()
