"""Worker: run all transductive fine-tuning experiments for ONE pretrained W&B run.

Called by sweep_transductive.py (one subprocess per pretrained run).  For each
(mode × fraction × pooling × train_seed) combination defined in
sweep_config_transductive.yaml, a separate W&B run is created and test metrics
are logged.

Transductive vs inductive difference
--------------------------------------
"Fraction" here is a fraction of labelled training *nodes*, not training graphs.
The full graph is always used for message passing; only ``train_mask`` is
narrowed to a random subset of the original training nodes per (fraction, seed).
``seed_subsample`` is always true in the transductive setting: each train_seed
independently resamples the labelled node subset.

Usage (normally called by sweep_transductive.py, but can be run standalone)
-----
    python scripts/finetuning/worker_transductive.py \\
        --project  gpse_backbone_pretrain_sweep_transductive_graphmaev2_cocitation_cora \\
        --entity   <wandb-entity> \\
        --run-id   <pretrained-run-id> \\
        --device   cuda:0 \\
        --finetune-project  finetune_gpse_backbone_pretrain_sweep_transductive_graphmaev2_cocitation_cora_seedsub \\
        [--config  scripts/finetuning/sweep_config_transductive.yaml]
        [--skip-sanity-check]
"""

from __future__ import annotations

import argparse
import random
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import wandb

from scripts.finetuning.utils import (
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
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_sweep_config(path: str) -> dict:
    """Load YAML sweep config and apply defaults for missing keys."""
    defaults = dict(
        modes=["finetune-full", "finetune-probe", "random-init-full", "random-init-probe"],
        fractions=[0.01, 0.05, 0.15],
        poolings=["mean"],
        train_seeds=[0, 1, 2, 3],
        max_epochs=400,
        patience=10,
        seed=42,
        batch_size=None,
        # Transductive always uses seed_subsample=True — every train_seed
        # independently resamples the labelled training-node subset.
        seed_subsample=True,
        seed_subsample_project_suffix="_seedsub",
    )
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    merged = {**defaults, **cfg}
    # Enforce seed_subsample=True for transductive; log a warning if overridden.
    if not merged.get("seed_subsample", True):
        print(
            "  [WARNING] seed_subsample=false found in config but transductive "
            "fine-tuning always resamples nodes per train_seed.  Ignoring."
        )
    merged["seed_subsample"] = True
    return merged


def flatten_dict(d: dict, prefix: str = "", sep: str = ".") -> dict:
    out = {}
    for k, v in d.items():
        full_key = f"{prefix}{sep}{k}" if prefix else k
        if isinstance(v, dict):
            out.update(flatten_dict(v, full_key, sep))
        else:
            out[full_key] = v
    return out


def pretrained_config_dict(run: Any) -> dict:
    flat = flatten_dict(dict(run.config))
    return {f"pretrained_config_{k}": v for k, v in flat.items()}


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--project",  required=True, help="Pretrained W&B project.")
    p.add_argument("--entity",   required=True, help="W&B entity.")
    p.add_argument("--run-id",   required=True, dest="run_id")
    p.add_argument("--device",   default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--finetune-project", required=True, dest="finetune_project")
    p.add_argument("--config",
                   default=str(Path(__file__).parent / "sweep_config_transductive.yaml"))
    p.add_argument("--skip-sanity-check", action="store_true", dest="skip_sanity")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Main worker logic
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    cfg = load_sweep_config(args.config)

    print(f"\n{'═'*65}")
    print(f"  TRANSDUCTIVE WORKER  run={args.run_id}  device={device}")
    print(f"{'═'*65}")

    # ── 1. Load pretrained run + model ────────────────────────────────────────
    pretrain_run = fetch_run(args.project, args.entity, args.run_id)
    pretrain_overrides = get_hydra_overrides(pretrain_run)
    ckpt_path = get_checkpoint_path(pretrain_run)

    print("[Composing pretraining config + loading model...]")
    pretrain_hydra_cfg = compose_cfg(pretrain_overrides)
    pretrained_model = load_model(pretrain_hydra_cfg, ckpt_path, device)

    # ── 2. Optional sanity check ──────────────────────────────────────────────
    if not args.skip_sanity:
        print("[Sanity check: re-evaluating pretrained model on pretraining task...]")
        pretrain_dm = build_datamodule(pretrain_hydra_cfg)
        local_metrics = evaluate(pretrained_model, pretrain_dm, device, seed=cfg["seed"])
        summary = dict(pretrain_run.summary)
        ref = {k: v for k, v in sorted(summary.items()) if "test_best_rerun" in k}
        print(f"  local  : {local_metrics}")
        print(f"  W&B ref: {ref}")

    # ── 3. Downstream config + full datamodule ────────────────────────────────
    print("[Building downstream config + datamodule...]")
    downstream_overrides = extract_downstream_overrides(pretrain_overrides)
    downstream_cfg = compose_cfg(downstream_overrides)
    monitor_metric, monitor_mode = get_downstream_monitor(downstream_cfg)
    print(f"  monitor: {monitor_metric}  (mode={monitor_mode})")
    datamodule = build_datamodule(downstream_cfg)

    # For transductive datasets, report node counts (not graph counts).
    n_train_full = count_transductive_nodes(datamodule.dataset_train, "train_mask")
    n_val        = count_transductive_nodes(datamodule.dataset_val,   "val_mask")
    n_test       = count_transductive_nodes(datamodule.dataset_test,  "test_mask")
    print(f"  node splits: train={n_train_full}  val={n_val}  test={n_test}")

    pretrain_cfg_fields = pretrained_config_dict(pretrain_run)
    base_wandb_config = {
        **pretrain_cfg_fields,
        "pretrained_run_id":   pretrain_run.id,
        "pretrained_run_name": pretrain_run.name,
        "ft_max_epochs":   cfg["max_epochs"],
        "ft_patience":     cfg["patience"],
        "ft_seed":         cfg["seed"],
        "ft_seed_subsample": True,   # always True for transductive
        "n_train_full":    n_train_full,
        "n_val":           n_val,
        "n_test":          n_test,
        "learning_setting": "transductive",
    }

    # ── 4. Run all (mode × fraction × pooling × train_seed) experiments ──────
    # seed_subsample=True always: each train_seed draws a different random
    # subset of labelled training nodes (different data AND different init).
    for mode in cfg["modes"]:
        for fraction in cfg["fractions"]:
            for pooling in cfg["poolings"]:
                frac_pct = int(fraction * 100)

                for train_seed in cfg["train_seeds"]:
                    # Each train_seed → independent random subset of train nodes.
                    subset_dm = make_subset_transductive_datamodule(
                        datamodule, fraction,
                        batch_size=cfg["batch_size"],
                        seed=train_seed,
                    )
                    n_train = count_transductive_nodes(
                        subset_dm.dataset_train, "train_mask"
                    )

                    run_name = (
                        f"{pretrain_run.name}__{mode}__{frac_pct}pct"
                        f"__{pooling}__s{train_seed}"
                    )

                    print(f"\n{'─'*60}")
                    print(f"  {run_name}")
                    print(f"{'─'*60}")

                    try:
                        _set_seed(train_seed)

                        model = build_downstream_model(
                            pretrained_model, downstream_cfg, pooling
                        )
                        apply_finetuning_mode(model, mode)

                        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
                        n_total     = sum(p.numel() for p in model.parameters())
                        print(f"  params   : {n_trainable:,} / {n_total:,} trainable")
                        print(f"  labelled : {n_train} / {n_train_full} training nodes ({frac_pct}%)")

                        test_metrics, fit_metrics = run_finetune_experiment(
                            model=model,
                            datamodule=subset_dm,
                            device=device,
                            monitor_metric=monitor_metric,
                            monitor_mode=monitor_mode,
                            max_epochs=cfg["max_epochs"],
                            patience=cfg["patience"],
                            seed=train_seed,
                        )

                        print(f"  test     : {test_metrics}")
                        if fit_metrics:
                            print(f"  best ep  : {fit_metrics}")

                        # ── Log to W&B ────────────────────────────────────────
                        wandb.init(
                            project=args.finetune_project,
                            entity=args.entity,
                            name=run_name,
                            group=pretrain_run.id,
                            tags=[mode, f"{frac_pct}pct", pooling,
                                  f"seed{train_seed}", pretrain_run.name,
                                  "transductive"],
                            config={
                                **base_wandb_config,
                                "ft_mode":          mode,
                                "ft_fraction":      fraction,
                                "ft_pooling":       pooling,
                                "ft_train_seed":    train_seed,
                                "ft_subset_seed":   train_seed,   # always same as train_seed
                                "n_train":          n_train,
                                "n_trainable_params": n_trainable,
                                "n_total_params":     n_total,
                            },
                            reinit=True,
                        )
                        for k, v in test_metrics.items():
                            wandb.summary[k] = v
                        wandb.log(test_metrics)
                        for k, v in fit_metrics.items():
                            wandb.summary[k] = v
                        wandb.finish()

                    except Exception:
                        print(f"\n  [ERROR] {run_name} failed:")
                        traceback.print_exc()
                        try:
                            wandb.finish(exit_code=1)
                        except Exception:
                            pass

    print(f"\n  Worker done: {pretrain_run.name}  ({args.run_id})")


if __name__ == "__main__":
    main()
