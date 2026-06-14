"""Few-shot fine-tuning of a pretrained TopoBench model from W&B.

Loads the best pretrained checkpoint, replaces the pretraining wrapper + readout
with a plain GNNWrapper + DownstreamReadOut, and runs four fine-tuning strategies
at three training-data fractions.

Fine-tuning modes
-----------------
    finetune-full      pretrained weights, train all parameters
    finetune-probe     pretrained weights, freeze encoder, train readout only
    random-init-full   randomise all weights, train all parameters
    random-init-probe  randomise all weights, freeze encoder, train readout only

Usage
-----
    python scripts/finetuning/main.py \\
        --project  gin_pretrain_sweep_graphmaev2_BBB_Martins \\
        --entity   <wandb-entity> \\
        [--run-id  <run-id>]           # omit → uses first (oldest) run
        [--fractions 0.05 0.5 1.0]    # training-data fractions
        [--modes finetune-full finetune-probe random-init-full random-init-probe]
        [--pooling mean]               # readout graph pooling: mean | sum | max
        [--max-epochs 100]
        [--patience 20]
        [--device cuda:0]
        [--seed 42]
        [--pretrain-eval]              # also re-evaluate pretraining metrics
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.finetuning.utils import (
    FINETUNE_MODES,
    apply_finetuning_mode,
    build_datamodule,
    build_downstream_model,
    compose_cfg,
    evaluate,
    extract_downstream_overrides,
    fetch_run,
    get_checkpoint_path,
    get_downstream_monitor,
    get_hydra_overrides,
    load_model,
    make_subset_datamodule,
    run_finetune_experiment,
)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Few-shot fine-tuning of a pretrained TopoBench model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--project",  default="gpse_backbone_pretrain_sweep_graphmaev2_COLLAB")
    p.add_argument("--entity",   default="louis-van-langendonck-universitat-polit-cnica-de-catalunya")
    p.add_argument("--run-id",   default=None, dest="run_id",
                   help="W&B run ID. Omit to use the first run in the project.")
    p.add_argument("--fractions", nargs="+", type=float, default=[0.05],
                   metavar="F", help="Fractions of training data to use.")
    p.add_argument("--modes", nargs="+", default=FINETUNE_MODES,
                   choices=FINETUNE_MODES, metavar="MODE",
                   help="Fine-tuning modes to run.")
    p.add_argument("--poolings", nargs="+", default=["mean", "sum"],
                   choices=["mean", "sum", "max"], metavar="POOL",
                   help="Graph pooling options for the downstream readout.")
    p.add_argument("--max-epochs",   type=int, default=100, dest="max_epochs")
    p.add_argument("--patience",     type=int, default=20)
    p.add_argument("--seed",         type=int, default=42,
                   help="Fixed seed for few-shot subset selection (data, not training).")
    p.add_argument("--train-seeds",  nargs="+", type=int, default=[0, 1, 2],
                   dest="train_seeds", metavar="S",
                   help="Seeds for model init + training randomness (one repeat per seed).")
    p.add_argument("--seed-subsample", action="store_true", dest="seed_subsample",
                   help="When set, each train-seed independently resamples the training "
                        "fraction instead of all seeds sharing the same fixed subset. "
                        "Corresponds to seed_subsample=true in sweep_config.yaml.")
    p.add_argument("--batch-size",   type=int, default=None, dest="batch_size",
                   help="Override batch size for fine-tuning (default: inherit from pretrain run).")
    p.add_argument("--device",
                   default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no-pretrain-eval", action="store_true", dest="skip_pretrain_eval",
                   help="Skip the pretraining sanity-check evaluation.")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _make_header(title: str, width: int = 70) -> str:
    pad = max(0, width - len(title) - 2)
    return f"{'═' * (pad // 2)} {title} {'═' * (pad - pad // 2)}"


def _print_results_table(results: dict[str, dict[str, float]]) -> None:
    """Pretty-print a comparison table of all experiment metrics."""
    if not results:
        return

    # Collect and sort metric names; strip leading "test/" prefix for display
    all_keys = sorted({k for v in results.values() for k in v})
    col_names = [k.replace("test/", "") for k in all_keys]
    col_w = max(12, *(len(c) + 2 for c in col_names))
    row_w = 40

    header = f"{'Experiment':<{row_w}}" + "".join(f"{c:>{col_w}}" for c in col_names)
    sep = "─" * len(header)

    print(f"\n{sep}")
    print(header)
    print(sep)
    for exp, metrics in results.items():
        row = f"{exp:<{row_w}}" + "".join(
            f"{metrics.get(k, float('nan')):>{col_w}.4f}" for k in all_keys
        )
        print(row)
    print(sep)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    print(_make_header(f"{args.entity}/{args.project}"))
    print(f"  run     : {args.run_id or '(first run)'}")
    print(f"  device  : {device}")
    print(f"  modes    : {args.modes}")
    print(f"  fractions: {args.fractions}")
    print(f"  poolings : {args.poolings}")
    print(f"  epochs  : {args.max_epochs}  patience={args.patience}")
    print(f"  seed_subsample: {args.seed_subsample}")

    # ── 1. W&B run + checkpoint ───────────────────────────────────────────────
    run = fetch_run(args.project, args.entity, args.run_id)
    pretrain_overrides = get_hydra_overrides(run)
    ckpt_path = get_checkpoint_path(run)

    print(f"\n  pretrain overrides ({len(pretrain_overrides)}):")
    for o in pretrain_overrides:
        print(f"    {o}")

    # ── 2. Reconstruct pretraining config + model ─────────────────────────────
    print("\n[Composing pretraining config...]")
    pretrain_cfg = compose_cfg(pretrain_overrides)

    print("\n[Loading pretrained model...]")
    pretrained_model = load_model(pretrain_cfg, ckpt_path, device)

    # Sanity-check: re-evaluate the pretrained model on its own pretraining task
    # to confirm loaded weights reproduce the W&B-reported best metrics.
    if not args.skip_pretrain_eval:
        print("\n[Building pretraining datamodule for sanity check...]")
        pretrain_dm = build_datamodule(pretrain_cfg)
        print("\n[Evaluating pretrained model on pretraining task...]")
        pretrain_metrics = evaluate(pretrained_model, pretrain_dm, device, seed=args.seed)
        print("\n  pretrain test metrics (local):")
        for k, v in sorted(pretrain_metrics.items()):
            print(f"    {k}: {v:.4f}")
        summary = dict(run.summary)
        ref = {k: v for k, v in sorted(summary.items()) if "test_best_rerun" in k}
        if ref:
            print("  W&B reported best (reference):")
            for k, v in ref.items():
                print(f"    {k}: {v}")

    # ── 3. Downstream config + datamodule ─────────────────────────────────────
    print("\n[Composing downstream (supervised) config...]")
    downstream_overrides = extract_downstream_overrides(pretrain_overrides)
    print(f"  downstream overrides ({len(downstream_overrides)}):")
    for o in downstream_overrides:
        print(f"    {o}")

    downstream_cfg = compose_cfg(downstream_overrides)
    monitor_metric, monitor_mode = get_downstream_monitor(downstream_cfg)
    print(f"  monitor: {monitor_metric}  (mode={monitor_mode})")

    print("\n[Building downstream datamodule...]")
    datamodule = build_datamodule(downstream_cfg)
    n_train_full = len(datamodule.dataset_train)
    n_val = len(datamodule.dataset_val) if datamodule.dataset_val is not None else 0
    n_test = len(datamodule.dataset_test) if datamodule.dataset_test is not None else 0
    print(f"  splits: train={n_train_full}  val={n_val}  test={n_test}")

    # ── 4. Run all (mode × fraction × pooling × train_seed) experiments ──────
    # seed_subsample controls whether each train_seed draws a different random
    # subset (True) or all seeds share the same fixed subset (False / legacy).
    results: dict[str, dict[str, float]] = {}

    for mode in args.modes:
        for frac in args.fractions:
            for pooling in args.poolings:
                # When seed_subsample is off: build subset once per (frac, pooling)
                # so the same data is used for every train_seed (legacy behaviour).
                if not args.seed_subsample:
                    subset_dm = make_subset_datamodule(
                        datamodule, frac, batch_size=args.batch_size, seed=args.seed
                    )
                    n_train = len(subset_dm.dataset_train)

                for train_seed in args.train_seeds:
                    # When seed_subsample is on: each train_seed draws its own
                    # independent random subsample of the training fraction.
                    if args.seed_subsample:
                        subset_seed = train_seed
                        subset_dm = make_subset_datamodule(
                            datamodule, frac, batch_size=args.batch_size, seed=subset_seed
                        )
                        n_train = len(subset_dm.dataset_train)
                    else:
                        subset_seed = args.seed

                    exp_key = f"{mode} @ {int(frac * 100):3d}% [{pooling}] s{train_seed}"
                    print(f"\n{'─' * 60}")
                    print(f"  {exp_key}")
                    print(f"{'─' * 60}")

                    random.seed(train_seed)
                    np.random.seed(train_seed)
                    torch.manual_seed(train_seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(train_seed)

                    model = build_downstream_model(pretrained_model, downstream_cfg, pooling)
                    apply_finetuning_mode(model, mode)

                    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
                    n_total = sum(p.numel() for p in model.parameters())
                    print(f"  params    : {n_trainable:,} / {n_total:,} trainable")
                    print(f"  train size: {n_train}  ({frac * 100:.0f}% of {n_train_full})")

                    test_metrics, fit_metrics = run_finetune_experiment(
                        model=model,
                        datamodule=subset_dm,
                        device=device,
                        monitor_metric=monitor_metric,
                        monitor_mode=monitor_mode,
                        max_epochs=args.max_epochs,
                        patience=args.patience,
                        seed=train_seed,
                    )
                    results[exp_key] = test_metrics
                    for k, v in sorted(test_metrics.items()):
                        print(f"  {k}: {v:.4f}")
                    if fit_metrics:
                        print("  (best epoch train/val)")
                        for k, v in sorted(fit_metrics.items()):
                            print(f"    {k}: {v:.4f}")

    # ── 5. Summary table ──────────────────────────────────────────────────────
    print(f"\n{_make_header('RESULTS  ' + run.name)}")
    _print_results_table(results)
    print("\nDone.")


if __name__ == "__main__":
    main()
