"""Worker: run all fine-tuning experiments for ONE pretrained W&B run.

Called by sweep.py (one subprocess per pretrained run).  For each
(mode × fraction) combination defined in sweep_config.yaml, a separate W&B
run is created in the finetune project and the test metrics are logged.

Every downstream run also records the complete pretrained model config under
``pretrained_config_*`` keys so results are fully self-contained for analysis.

Usage (normally called by sweep.py, but can be run standalone)
-----
    python scripts/finetuning/worker.py \\
        --project  gin_pretrain_sweep_graphmaev2_BBB_Martins \\
        --entity   <wandb-entity> \\
        --run-id   <pretrained-run-id> \\
        --device   cuda:0 \\
        --finetune-project  finetune_gin_pretrain_sweep_graphmaev2_BBB_Martins \\
        [--config  scripts/finetuning/sweep_config.yaml]
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
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_sweep_config(path: str) -> dict:
    """Load YAML sweep config and apply defaults for missing keys."""
    defaults = dict(
        modes=["finetune-full", "finetune-probe", "random-init-full", "random-init-probe"],
        fractions=[0.05, 0.50, 1.00],
        poolings=["mean"],
        train_seeds=[0, 1, 2],
        max_epochs=100,
        patience=20,
        seed=42,
        batch_size=None,
        seed_subsample=False,
        seed_subsample_project_suffix="_seedsub",
    )
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    return {**defaults, **cfg}


def flatten_dict(d: dict, prefix: str = "", sep: str = ".") -> dict:
    """Recursively flatten a nested dict, joining keys with *sep*."""
    out = {}
    for k, v in d.items():
        full_key = f"{prefix}{sep}{k}" if prefix else k
        if isinstance(v, dict):
            out.update(flatten_dict(v, full_key, sep))
        else:
            out[full_key] = v
    return out


def pretrained_config_dict(run: Any) -> dict:
    """Return flattened pretrained run config prefixed with ``pretrained_config_``."""
    flat = flatten_dict(dict(run.config))
    return {f"pretrained_config_{k}": v for k, v in flat.items()}


def _set_seed(seed: int) -> None:
    """Seed all RNGs for reproducible model initialisation and training."""
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
    p.add_argument("--finetune-project", required=True, dest="finetune_project",
                   help="Target W&B project for downstream runs.")
    p.add_argument("--config",
                   default=str(Path(__file__).parent / "sweep_config.yaml"),
                   help="Path to sweep_config.yaml.")
    p.add_argument("--skip-sanity-check", action="store_true", dest="skip_sanity",
                   help="Skip pretraining metric verification.")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Main worker logic
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    cfg = load_sweep_config(args.config)

    print(f"\n{'═'*65}")
    print(f"  WORKER  run={args.run_id}  device={device}")
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

    n_train_full = len(datamodule.dataset_train)
    n_val  = len(datamodule.dataset_val)   if datamodule.dataset_val  is not None else 0
    n_test = len(datamodule.dataset_test)  if datamodule.dataset_test is not None else 0
    print(f"  splits: train={n_train_full}  val={n_val}  test={n_test}")

    # Pre-compute shared W&B config fields for every downstream run
    pretrain_cfg_fields = pretrained_config_dict(pretrain_run)
    base_wandb_config = {
        **pretrain_cfg_fields,
        "pretrained_run_id":   pretrain_run.id,
        "pretrained_run_name": pretrain_run.name,
        "ft_max_epochs": cfg["max_epochs"],
        "ft_patience":   cfg["patience"],
        "ft_seed":       cfg["seed"],
        "n_train_full":  n_train_full,
        "n_val":         n_val,
        "n_test":        n_test,
    }

    # ── 4. Run all (mode × fraction × pooling × train_seed) experiments ──────
    # seed_subsample controls whether each train_seed draws a DIFFERENT random
    # subset (True) or all seeds share the same fixed subset (False / legacy).
    seed_subsample: bool = cfg["seed_subsample"]
    fixed_subset_seed: int = cfg["seed"]

    # When seed_subsample is on, log to a separate W&B project so the two
    # experimental designs never mix in the same project.
    finetune_project = args.finetune_project
    if seed_subsample:
        suffix = cfg.get("seed_subsample_project_suffix", "_seedsub")
        finetune_project = args.finetune_project + suffix
        print(f"  [seed_subsample=True] → W&B project: {finetune_project}")

    for mode in cfg["modes"]:
        for fraction in cfg["fractions"]:
            for pooling in cfg["poolings"]:
                # When seed_subsample is off: build the subset once and reuse it
                # for all train_seeds (original behaviour — only model init varies).
                if not seed_subsample:
                    subset_dm = make_subset_datamodule(
                        datamodule, fraction,
                        batch_size=cfg["batch_size"],
                        seed=fixed_subset_seed,
                    )
                    n_train = len(subset_dm.dataset_train)

                frac_pct = int(fraction * 100)

                for train_seed in cfg["train_seeds"]:
                    # When seed_subsample is on: each train_seed draws its own
                    # independent random subsample of the training fraction.
                    if seed_subsample:
                        subset_seed = train_seed
                        subset_dm = make_subset_datamodule(
                            datamodule, fraction,
                            batch_size=cfg["batch_size"],
                            seed=subset_seed,
                        )
                        n_train = len(subset_dm.dataset_train)
                    else:
                        subset_seed = fixed_subset_seed

                    run_name = (
                        f"{pretrain_run.name}__{mode}__{frac_pct}pct"
                        f"__{pooling}__s{train_seed}"
                    )

                    print(f"\n{'─'*60}")
                    print(f"  {run_name}")
                    print(f"{'─'*60}")

                    try:
                        # Seed RNGs before model construction so that readout
                        # init and random-init weight randomisation are
                        # reproducible per train_seed.
                        _set_seed(train_seed)

                        model = build_downstream_model(
                            pretrained_model, downstream_cfg, pooling
                        )
                        apply_finetuning_mode(model, mode)

                        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
                        n_total     = sum(p.numel() for p in model.parameters())
                        print(f"  params   : {n_trainable:,} / {n_total:,} trainable")
                        print(f"  training : {n_train} samples ({frac_pct}%)")

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
                            project=finetune_project,
                            entity=args.entity,
                            name=run_name,
                            group=pretrain_run.id,
                            tags=[mode, f"{frac_pct}pct", pooling,
                                  f"seed{train_seed}", pretrain_run.name],
                            config={
                                **base_wandb_config,
                                "ft_mode":            mode,
                                "ft_fraction":        fraction,
                                "ft_pooling":         pooling,
                                "ft_train_seed":      train_seed,
                                "ft_subset_seed":     subset_seed,
                                "ft_seed_subsample":  seed_subsample,
                                "n_train":            n_train,
                                "n_trainable_params": n_trainable,
                                "n_total_params":     n_total,
                            },
                            reinit=True,
                        )
                        # Test: identical to original (summary + log step 0)
                        for k, v in test_metrics.items():
                            wandb.summary[k] = v
                        wandb.log(test_metrics)
                        # Train/val at best val epoch: summary only (no wandb.log)
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
