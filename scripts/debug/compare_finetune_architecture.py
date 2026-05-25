#!/usr/bin/env python3
"""Compare downstream models built from different pretraining wrappers.

Read-only diagnostic: does not modify training code.

Usage (from repo root, needs project deps: hydra, torch, torch_geometric):
    python scripts/debug/compare_finetune_architecture.py
    python scripts/debug/compare_finetune_architecture.py --csv-only
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import hydra
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig
from torch_geometric.data import Batch, Data

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import rootutils

rootutils.setup_root(_PROJECT_ROOT, indicator=".project-root", pythonpath=True)

from topobench.nn.wrappers.graph.gnn_wrapper import GNNWrapper

DATASET = "graph/IMDB-BINARY"
HIDDEN = 64
LAYERS = 2

METHOD_OVERRIDES: dict[str, list[str]] = {
    "graphcl": [
        "pretraining=graphcl",
        f"dataset={DATASET}",
        "model=graph/gin",
        f"++model.feature_encoder.out_channels={HIDDEN}",
        f"++model.backbone.num_layers={LAYERS}",
        "model.backbone_wrapper.aug1=mask_attr",
        "model.backbone_wrapper.aug2=drop_edge",
        "model.backbone_wrapper.residual_connections=true",
    ],
    "dgi": [
        "pretraining=dgi",
        f"dataset={DATASET}",
        "model=graph/gin",
        f"++model.feature_encoder.out_channels={HIDDEN}",
        f"++model.backbone.num_layers={LAYERS}",
        "model.backbone_wrapper.residual_connections=true",
    ],
    "bgrl": [
        "pretraining=bgrl",
        f"dataset={DATASET}",
        "model=graph/gin",
        f"++model.feature_encoder.out_channels={HIDDEN}",
        f"++model.backbone.num_layers={LAYERS}",
        "model.backbone_wrapper.residual_connections=true",
    ],
    "vgae": [
        "pretraining=vgae",
        f"dataset={DATASET}",
        "model=graph/gin",
        f"++model.feature_encoder.out_channels={HIDDEN}",
        f"++model.backbone.num_layers={LAYERS}",
        "model.backbone_wrapper.residual_connections=true",
    ],
    "graphmaev2": [
        "pretraining=graphmaev2",
        f"dataset={DATASET}",
        "model=graph/gin",
        f"++model.feature_encoder.out_channels={HIDDEN}",
        f"++model.backbone.num_layers={LAYERS}",
        "model.backbone_wrapper.residual_connections=true",
    ],
}

_DOWNSTREAM_KEEP = (
    "dataset=",
    "model=",
    "++model.feature_encoder.out_channels=",
    "++model.backbone.num_layers=",
    "++model.backbone.dropout=",
    "++model.feature_encoder.proj_dropout=",
    "++dataset.split_params.data_seed=",
    "++dataset.dataloader_params.batch_size=",
    "++optimizer.",
)


def compose_cfg(overrides: list[str]) -> DictConfig:
    from topobench.utils.config_resolvers import register_all_resolvers

    register_all_resolvers()
    config_dir = str(_PROJECT_ROOT / "configs")
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=config_dir, version_base="1.3", job_name="arch_check"):
        return compose(config_name="run", overrides=overrides)


def extract_downstream_overrides(pretrain_overrides: list[str]) -> list[str]:
    result = ["pretraining=none"]
    for arg in pretrain_overrides:
        if any(arg.startswith(k) for k in _DOWNSTREAM_KEEP):
            result.append(arg)
    return result


def build_downstream_wrapper_like_finetuning(pretrained_wrapper, downstream_cfg: DictConfig):
    """Mirror scripts/finetuning/utils.build_downstream_model wrapper construction."""
    pw = pretrained_wrapper

    if hasattr(pw, "encoder_ema") and pw.encoder_ema is not None:
        gnn_encoder = copy.deepcopy(pw.encoder_ema)
    elif hasattr(pw, "online_encoder"):
        gnn_encoder = copy.deepcopy(pw.online_encoder)
    else:
        gnn_encoder = copy.deepcopy(pw.backbone)

    out_channels = downstream_cfg.model.feature_encoder.out_channels
    residual = pw.residual_connections
    num_cell_dims = len(list(pw.dimensions))

    new_wrapper = GNNWrapper(
        backbone=gnn_encoder,
        out_channels=out_channels,
        num_cell_dimensions=num_cell_dims,
        residual_connections=residual,
    )
    if residual:
        for i in range(num_cell_dims):
            src = getattr(pw, f"ln_{i}", None)
            dst = getattr(new_wrapper, f"ln_{i}", None)
            if src is not None and dst is not None:
                dst.load_state_dict(src.state_dict())
    return new_wrapper, residual


def instantiate_pretrain_wrapper(cfg: DictConfig):
    """Instantiate feature_encoder + backbone_wrapper (no Lightning)."""
    feature_encoder = hydra.utils.instantiate(cfg.model.feature_encoder)
    backbone = hydra.utils.instantiate(cfg.model.backbone)
    # backbone_wrapper is _partial_: true — must pass backbone explicitly.
    wrapper_factory = hydra.utils.instantiate(cfg.model.backbone_wrapper)
    wrapper = wrapper_factory(backbone=backbone)
    readout = hydra.utils.instantiate(cfg.model.readout)
    return feature_encoder, wrapper, readout


def _make_batch(num_graphs: int = 4, num_nodes: int = 10, in_dim: int = 1) -> Batch:
    graphs = []
    for _ in range(num_graphs):
        x = torch.randn(num_nodes, in_dim)
        ei = torch.randint(0, num_nodes, (2, num_nodes * 3))
        graphs.append(Data(x=x, edge_index=ei, y=torch.tensor([0])))
    batch = Batch.from_data_list(graphs)
    batch.x_0 = batch.x
    batch.batch_0 = batch.batch
    return batch


def analyze_methods() -> None:
    print("=" * 72)
    print("Downstream architecture comparison (GIN 64-dim, 2-layer, IMDB-BINARY)")
    print("=" * 72)

    for method, overrides in METHOD_OVERRIDES.items():
        cfg = compose_cfg(overrides)
        _, pw, pre_readout = instantiate_pretrain_wrapper(cfg)
        ds_cfg = compose_cfg(extract_downstream_overrides(overrides))
        ds_wrapper, residual = build_downstream_wrapper_like_finetuning(pw, ds_cfg)

        dead_ln = 0
        if not residual:
            dead_ln = sum(
                p.numel() for name, p in ds_wrapper.named_parameters() if name.startswith("ln_")
            )

        print(f"\n[{method}]")
        print(f"  pretrain wrapper     : {type(pw).__name__}")
        print(f"  pretrain residual    : {getattr(pw, 'residual_connections', None)}  "
              f"(sweep may say true for GraphCL but wrapper forces false)")
        print(f"  pools in pre-wrapper : {hasattr(pw, 'pool_graph')}")
        print(f"  pretrain readout     : {type(pre_readout).__name__}")
        print(f"  downstream wrapper   : {type(ds_wrapper).__name__}")
        print(f"  downstream residual  : {residual}")
        print(f"  LayerNorm in forward : {residual}")
        print(f"  unused LayerNorm params after conversion: {dead_ln:,}")


def analyze_csv_confound(csv_dir: Path) -> None:
    try:
        import pandas as pd
    except ImportError:
        print("pandas not installed; skipping CSV analysis")
        return

    if not csv_dir.is_dir():
        print(f"No CSV dir: {csv_dir}")
        return

    print("\n" + "=" * 72)
    print("CSV: random-init-full @ 100% — one row per project after best-hyperparam pick")
    print("Selection does NOT include ft_pooling or pretrained hidden/layers/aug.")
    print("=" * 72)

    for path in sorted(csv_dir.glob("finetune_*_pretrain_sweep_*.csv")):
        if "_flagged" in path.name:
            continue
        df = pd.read_csv(path)
        sub = df[(df["ft_mode"] == "random-init-full") & (df["ft_fraction"].astype(float) == 1.0)]
        if sub.empty:
            continue
        parts = path.stem.replace("finetune_gin_pretrain_sweep_", "").split("_", 1)
        method, dataset = parts[0], parts[1] if len(parts) > 1 else "?"
        row = sub.iloc[0]
        test_cols = [c for c in sub.columns if c.startswith("test/") and c.endswith("_mean")]
        metric = row[test_cols[0]] if test_cols else float("nan")
        npar = row.get("n_total_params", "?")
        pool = row.get("ft_pooling", "?")
        hid = row.get("pretrained_config_model.feature_encoder.out_channels", "?")
        nl = row.get("pretrained_config_model.backbone.num_layers", "?")
        print(f"  {method:<10} {dataset:<22}  metric={metric:.4f}  "
              f"params={npar}  ft_pool={pool}  hidden={hid}  layers={nl}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv-dir",
        type=Path,
        default=_PROJECT_ROOT / "scripts/plotting/outputs/processed_projects",
    )
    parser.add_argument("--csv-only", action="store_true")
    args = parser.parse_args()

    if not args.csv_only:
        try:
            analyze_methods()
        except Exception as e:
            print(f"Hydra instantiation failed ({e}). Run in the project venv; CSV analysis still runs.")
    analyze_csv_confound(args.csv_dir)


if __name__ == "__main__":
    main()
