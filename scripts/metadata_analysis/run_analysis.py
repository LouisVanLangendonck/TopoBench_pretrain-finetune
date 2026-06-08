#!/usr/bin/env python3
"""Dataset metadata analysis script.

For each dataset listed in the config YAML the script:
  1. Loads the dataset via the same Hydra pipeline used during pretraining
     (model=graph/gin + pretraining=dgi by default), ensuring identical
     pre_transforms (e.g. one-hot degree features for IMDB-BINARY, OGB atom
     features for molecular datasets).
  2. Applies the same train/val/test split (same seed) as the pretraining runs.
  3. Computes graph-level metadata features for each split separately.
  4. Saves results to ``<output_dir>/<DatasetName>.json``.

Incremental / additive runs
----------------------------
Features toggled to ``false`` in the config are skipped entirely — they are
**not** recomputed and their existing values in the output JSON are preserved.
This means you can run the script multiple times with different subsets of
features enabled and the JSON will accumulate all results without wasting time
re-running features that were already computed.

Concretely, when an output file already exists:
  - Each split dict (train / val / test) is updated with the newly computed
    keys only.  Pre-existing keys that were not enabled this run are kept as-is.
  - Top-level metadata (dataset, model, pretraining, data_seed) is refreshed.

Default output directory: ``scripts/metadata_analysis/outputs/<DatasetName>.json``

Usage
-----
    # Default config (scripts/metadata_analysis/config.yaml):
    python scripts/metadata_analysis/run_analysis.py

    # Custom config path:
    python scripts/metadata_analysis/run_analysis.py --config path/to/config.yaml

    # Override output directory:
    python scripts/metadata_analysis/run_analysis.py --output-dir outputs/my_analysis

    # Analyse a single dataset without editing the YAML:
    python scripts/metadata_analysis/run_analysis.py --datasets graph/IMDB-BINARY graph/MUTAG

    # Dry-run (show which datasets would be processed, then exit):
    python scripts/metadata_analysis/run_analysis.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import yaml

# ── Project root on sys.path ──────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from compute_metrics import compute_split_features  # noqa: E402  (local import)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset loading (mirrors scripts/finetuning/utils.py::build_datamodule)
# ─────────────────────────────────────────────────────────────────────────────

def _compose_cfg(dataset: str, model: str, pretraining: str, data_seed: int):
    """Compose the Hydra DictConfig for a given (dataset, model, pretraining)."""
    import rootutils
    import hydra
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    rootutils.setup_root(_PROJECT_ROOT, indicator=".project-root", pythonpath=True)

    from topobench.utils.config_resolvers import register_all_resolvers
    register_all_resolvers()

    config_dir = str(_PROJECT_ROOT / "configs")
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=config_dir, version_base="1.3", job_name="metadata"):
        cfg = compose(
            config_name="run",
            overrides=[
                f"dataset={dataset}",
                f"model={model}",
                f"pretraining={pretraining}",
                f"++dataset.split_params.data_seed={data_seed}",
                # Silence W&B / trainer noise during data loading
                "logger=[]",
            ],
        )
    return cfg


def _load_splits(dataset: str, model: str, pretraining: str, data_seed: int):
    """Return (train_list, val_list, test_list) of PyG Data objects.

    The data is loaded and pre-transformed exactly as it would be during
    pretraining (cached on disk under the same hash-based directory).
    """
    import hydra
    from topobench.data.preprocessor import PreProcessor

    cfg = _compose_cfg(dataset, model, pretraining, data_seed)

    loader = hydra.utils.instantiate(cfg.dataset.loader)
    raw_dataset, dataset_dir = loader.load()

    preprocessor = PreProcessor(raw_dataset, dataset_dir, cfg.get("transforms"))
    train_ds, val_ds, test_ds = preprocessor.load_dataset_splits(cfg.dataset.split_params)

    # DataloadDataset stores graphs in .data_lst
    train_list = train_ds.data_lst
    val_list   = val_ds.data_lst
    test_list  = test_ds.data_lst

    return train_list, val_list, test_list


# ─────────────────────────────────────────────────────────────────────────────
# Per-dataset analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyse_dataset(
    dataset: str,
    model: str,
    pretraining: str,
    data_seed: int,
    features_config: dict,
) -> dict:
    """Run the full metadata analysis for one dataset.

    Returns
    -------
    dict
        ``{"dataset": ..., "train": {...}, "val": {...}, "test": {...}}``
    """
    t0 = time.perf_counter()
    print(f"  Loading splits …")
    train_list, val_list, test_list = _load_splits(dataset, model, pretraining, data_seed)
    print(
        f"  Splits loaded in {time.perf_counter() - t0:.1f}s  "
        f"(train={len(train_list)}, val={len(val_list)}, test={len(test_list)})"
    )

    result: dict = {
        "dataset": dataset,
        "model": model,
        "pretraining": pretraining,
        "data_seed": data_seed,
    }

    for split_name, data_list in [("train", train_list), ("val", val_list), ("test", test_list)]:
        t1 = time.perf_counter()
        print(f"  Computing {split_name} features …")
        result[split_name] = compute_split_features(
            data_list,
            features_config,
            split_name=split_name,
            verbose=True,
        )
        print(f"  {split_name} done in {time.perf_counter() - t1:.1f}s")

    print(f"  Total elapsed: {time.perf_counter() - t0:.1f}s")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Result merging
# ─────────────────────────────────────────────────────────────────────────────

def _merge_result(existing: dict, new: dict) -> dict:
    """Merge newly-computed features into an existing result dict.

    Rules
    -----
    - Top-level metadata keys (dataset, model, pretraining, data_seed) are
      always refreshed from *new*.
    - Each split sub-dict (train / val / test) is merged with
      ``existing_split | new_split``: keys present in *new* overwrite the
      corresponding key in *existing*, but keys that only exist in *existing*
      (i.e. features that were disabled / skipped this run) are kept intact.
    - Any other top-level keys that appear in *new* but are not splits or
      standard metadata (e.g. importance scores added by run_importance.py)
      are written directly into *merged*, overriding the existing value.
    - All top-level keys in *existing* that are absent from *new* are
      preserved unchanged.
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
        description="Compute graph metadata features for all listed datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        default=str(_SCRIPT_DIR / "config.yaml"),
        help="Path to the YAML config file (default: scripts/metadata_analysis/config.yaml).",
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
        help="Override datasets list from config (e.g. graph/IMDB-BINARY graph/MUTAG).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override model from config (default: graph/gin).",
    )
    parser.add_argument(
        "--pretraining",
        default=None,
        help="Override pretraining method from config (default: dgi).",
    )
    parser.add_argument(
        "--data-seed",
        type=int,
        default=None,
        help="Override data_seed from config.",
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

    datasets    = args.datasets    or cfg.get("datasets", [])
    model       = args.model       or cfg.get("model", "graph/gin")
    pretraining = args.pretraining or cfg.get("pretraining", "dgi")
    data_seed   = args.data_seed   if args.data_seed is not None else cfg.get("data_seed", 0)
    output_dir  = Path(args.output_dir or cfg.get("output_dir", str(_SCRIPT_DIR / "outputs")))
    features    = cfg.get("features", {})

    if not datasets:
        print("ERROR: no datasets specified in config or via --datasets.", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print("  Dataset Metadata Analysis")
    print("=" * 60)
    print(f"  Config    : {config_path}")
    print(f"  Model     : {model}")
    print(f"  Pretrain  : {pretraining}")
    print(f"  Data seed : {data_seed}")
    print(f"  Output    : {output_dir}")
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

        try:
            result = analyse_dataset(
                dataset=dataset,
                model=model,
                pretraining=pretraining,
                data_seed=data_seed,
                features_config=features,
            )
        except Exception as exc:
            print(f"  ERROR processing {dataset}: {exc}")
            traceback.print_exc()
            failures.append((dataset, str(exc)))
            continue

        out_path = output_dir / f"{dataset_name}.json"
        if out_path.exists():
            try:
                with open(out_path) as fh:
                    existing = json.load(fh)
                result = _merge_result(existing, result)
                action = "Updated (merged)"
            except Exception as exc:
                print(f"  WARNING: could not read {out_path} — will overwrite. ({exc})")
                action = "Overwritten (read error)"
        else:
            action = "Created"
        with open(out_path, "w") as fh:
            json.dump(result, fh, indent=2)
        print(f"  {action} → {out_path}")
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
