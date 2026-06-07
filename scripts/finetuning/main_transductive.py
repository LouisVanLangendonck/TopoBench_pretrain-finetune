"""Few-shot fine-tuning of a pretrained TopoBench model — TRANSDUCTIVE setting.

Loads the best pretrained checkpoint, replaces the pretraining wrapper + readout
with a plain GNNWrapper + DownstreamReadOut, and runs fine-tuning strategies
at several labelled-training-node fractions.

Transductive vs inductive difference
--------------------------------------
In the inductive case (main.py) "fractions" refers to a random subset of
training *graphs*.  Here, the dataset is a single large graph with node-level
train / val / test masks (e.g. Cora, PubMed, Roman-Empire, Minesweeper).
"Fractions" therefore means a random subset of training *nodes* whose labels
are exposed to the model.  The full graph is always used for message passing.
Val and test evaluation is always on the full held-out node sets.

Fine-tuning modes
-----------------
    finetune-full      pretrained weights, train all parameters
    finetune-probe     pretrained weights, freeze encoder, train readout only
    random-init-full   randomise all weights, train all parameters
    random-init-probe  randomise all weights, freeze encoder, train readout only

Usage
-----
    python scripts/finetuning/main_transductive.py \\
        --project  gpse_backbone_pretrain_sweep_transductive_graphmaev2_cocitation_cora \\
        --entity   <wandb-entity> \\
        [--run-id  <run-id>]           # omit → uses first (oldest) run
        [--fractions 0.01 0.05 0.15]  # fraction of labelled training nodes
        [--modes finetune-full finetune-probe random-init-full random-init-probe]
        [--max-epochs 400]
        [--patience 10]
        [--device cuda:0]
        [--seed 42]
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
    count_transductive_nodes,
    evaluate,
    extract_downstream_overrides,
    fetch_run,
    get_checkpoint_path,
    get_downstream_monitor,
    get_hydra_overrides,
    load_model,
    make_subset_transductive_datamodule,
    run_finetune_experiment,
)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Few-shot transductive fine-tuning of a pretrained TopoBench model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--project",  default="gpse_backbone_pretrain_sweep_transductive_graphmaev2_cocitation_cora")
    p.add_argument("--entity",   default="louis-van-langendonck-universitat-polit-cnica-de-catalunya")
    p.add_argument("--run-id",   default=None, dest="run_id",
                   help="W&B run ID. Omit to use the first run in the project.")
    p.add_argument("--fractions", nargs="+", type=float, default=[0.01, 0.05, 0.15],
                   metavar="F",
                   help="Fractions of labelled training NODES to use (transductive few-shot).")
    p.add_argument("--modes", nargs="+", default=FINETUNE_MODES,
                   choices=FINETUNE_MODES, metavar="MODE",
                   help="Fine-tuning modes to run.")
    p.add_argument("--poolings", nargs="+", default=["mean"],
                   choices=["mean", "sum", "max"], metavar="POOL",
                   help="Graph pooling option passed to DownstreamReadOut. "
                        "For node-level transductive tasks this has no effect "
                        "on predictions but is kept for API consistency.")
    p.add_argument("--max-epochs",   type=int, default=400, dest="max_epochs")
    p.add_argument("--patience",     type=int, default=10)
    p.add_argument("--seed",         type=int, default=42,
                   help="Base RNG seed (not used for subsample selection when "
                        "seed_subsample is on).")
    p.add_argument("--train-seeds",  nargs="+", type=int, default=[0, 1, 2, 3],
                   dest="train_seeds", metavar="S",
                   help="Training seeds.  Each seed independently resamples the "
                        "labelled training-node subset (seed_subsample=true).")
    p.add_argument("--batch-size",   type=int, default=None, dest="batch_size",
                   help="Override batch size (default: inherit from pretrain config; "
                        "transductive datasets use batch_size=1).")
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
    if not results:
        return
    all_keys = sorted({k for v in results.values() for k in v})
    col_names = [k.replace("test/", "") for k in all_keys]
    col_w = max(12, *(len(c) + 2 for c in col_names))
    row_w = 44
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
    print(f"  run      : {args.run_id or '(first run)'}")
    print(f"  device   : {device}")
    print(f"  modes    : {args.modes}")
    print(f"  fractions: {args.fractions}  (labelled training-node fractions)")
    print(f"  poolings : {args.poolings}")
    print(f"  epochs   : {args.max_epochs}  patience={args.patience}")
    print(f"  seeds    : {args.train_seeds}  (each independently resamples nodes)")

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

    # For transductive datasets: report node counts from masks, not graph counts.
    n_train_full = count_transductive_nodes(datamodule.dataset_train, "train_mask")
    n_val  = count_transductive_nodes(datamodule.dataset_val,  "val_mask")
    n_test = count_transductive_nodes(datamodule.dataset_test, "test_mask")
    print(f"  node splits: train={n_train_full}  val={n_val}  test={n_test}")
    print(f"  (full graph has {len(datamodule.dataset_train)} training-batch item(s))")

    # ── 4. Run all (mode × fraction × pooling × train_seed) experiments ──────
    # In transductive mode seed_subsample is always True:
    # each train_seed independently resamples the labelled node subset.
    results: dict[str, dict[str, float]] = {}

    for mode in args.modes:
        for frac in args.fractions:
            for pooling in args.poolings:
                for train_seed in args.train_seeds:
                    # Each train_seed draws its own random subset of training nodes.
                    subset_dm = make_subset_transductive_datamodule(
                        datamodule, frac,
                        batch_size=args.batch_size,
                        seed=train_seed,
                    )
                    n_train = count_transductive_nodes(
                        subset_dm.dataset_train, "train_mask"
                    )

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
                    n_total     = sum(p.numel() for p in model.parameters())
                    print(f"  params      : {n_trainable:,} / {n_total:,} trainable")
                    print(f"  labelled    : {n_train} / {n_train_full} training nodes "
                          f"({frac * 100:.0f}%)")

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
