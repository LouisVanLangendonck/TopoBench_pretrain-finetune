"""Default W&B projects and parameter taxonomy for fine-tuning aggregation."""

from __future__ import annotations

from pathlib import Path

import yaml

_SCRIPT_DIR = Path(__file__).resolve().parent
_FINETUNE_CONFIG = _SCRIPT_DIR.parent / "finetuning" / "sweep_config.yaml"

# Hydra keys swept per method in scripts/pretraining/gin_pretrain_sweep.sh
METHOD_SWEPT_KEYS: dict[str, list[str]] = {
    "bgrl": [
        "model.backbone_wrapper.drop_edge_rate_2",
        "model.backbone_wrapper.drop_feature_rate_2",
        "model.backbone_wrapper.momentum",
        "model.readout.pooling_type",
    ],
    "dgi": [
        "model.backbone_wrapper.corruption_type",
        "model.readout.readout_type",
    ],
    "graphcl": [
        "model.backbone_wrapper.aug2",
        "model.backbone_wrapper.aug_ratio2",
        "model.readout.pooling_type",
    ],
    "graphmaev2": [
        "model.backbone_wrapper.mask_rate",
        "model.backbone_wrapper.momentum",
        "model.backbone_wrapper.residual_connections",
        "model.readout.pooling_type",
        "model.readout.decoder_type",
    ],
    "vgae": [
        "model.backbone_wrapper.edge_sample_ratio",
        "model.backbone_wrapper.variational",
        "model.readout.pooling_type",
    ],
}

# Shared GIN / optimizer sweep (same for every pretraining method)
SHARED_SWEPT_KEYS: list[str] = [
    "model.feature_encoder.out_channels",
    "model.backbone.num_layers",
    "optimizer.parameters.weight_decay",
    "optimizer.parameters.lr",
    "dataset.split_params.data_seed",
]

SHARED_COLUMN_NAMES: dict[str, str] = {
    "model.feature_encoder.out_channels": "gin_hidden",
    "model.backbone.num_layers": "gin_num_layers",
    "optimizer.parameters.weight_decay": "weight_decay",
    "optimizer.parameters.lr": "learning_rate",
    "dataset.split_params.data_seed": "pretrain_data_seed",
}

# Flattened W&B keys duplicated by canonical columns set in run_to_record()
PRETRAIN_CONFIG_TO_CANONICAL: dict[str, str] = {
    f"pretrained_config_{k}": v for k, v in SHARED_COLUMN_NAMES.items()
}

METHOD_PARAM_SHORT_NAMES: dict[str, str] = {
    "model.backbone_wrapper.drop_edge_rate_2": "drop_edge_rate_2",
    "model.backbone_wrapper.drop_feature_rate_2": "drop_feature_rate_2",
    "model.backbone_wrapper.momentum": "momentum",
    "model.readout.pooling_type": "readout_pooling",
    "model.backbone_wrapper.corruption_type": "corruption_type",
    "model.readout.readout_type": "readout_type",
    "model.backbone_wrapper.aug2": "aug2",
    "model.backbone_wrapper.aug_ratio2": "aug_ratio2",
    "model.backbone_wrapper.mask_rate": "mask_rate",
    "model.backbone_wrapper.residual_connections": "residual_connections",
    "model.readout.decoder_type": "decoder_type",
    "model.backbone_wrapper.edge_sample_ratio": "edge_sample_ratio",
    "model.backbone_wrapper.variational": "variational",
}

# Keys used to pick the best hyperparameter setting (pooling + method params compete)
SELECTION_GROUP_KEYS: list[str] = [
    "dataset",
    "pretraining_method",
    "gin_hidden",
    "gin_num_layers",
    "weight_decay",
    "learning_rate",
    "ft_mode",
    "ft_fraction",
]

# Never treat as hyperparameters (metadata / infra / outcomes)
NON_HYPERPARAM_PREFIXES = ("test/", "best_", "best_epoch/")
NON_HYPERPARAM_COLUMNS = frozenset({
    "wandb_project",
    "wandb_run_id",
    "wandb_run_name",
    "wandb_state",
    "wandb_group",
    "wandb_url",
    "pretrained_run_id",
    "pretrained_run_name",
    "pretrained_config_paths.output_dir",
    "pretrained_config_paths.log_dir",
    "pretrained_config_paths.work_dir",
    "ft_train_seed",
    "ft_subset_seed",
    "n_train",
    "n_val",
    "n_test",
    "n_train_full",
    "n_trainable_params",
    "n_total_params",
    "seed_count",
    "seed_ok",
    "selection_score",
    "monitor_metric",
    "monitor_mode",
    "monitor_test_column",
})


def load_finetune_sweep_defaults(config_path: str | Path | None = None) -> dict:
    """Load entity, pretrain projects, and train_seeds from finetuning sweep_config.yaml."""
    path = Path(config_path) if config_path else _FINETUNE_CONFIG
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    projects = cfg.get("projects") or []
    return {
        "entity": cfg.get(
            "entity",
            "louis-van-langendonck-universitat-polit-cnica-de-catalunya",
        ),
        "pretrain_projects": list(projects),
        "finetune_projects": [f"finetune_{p}" for p in projects],
        "train_seeds": list(cfg.get("train_seeds") or [0, 1, 2]),
    }


def parse_pretrain_project(project: str) -> tuple[str, str]:
    """Return (pretraining_method, dataset) from e.g. gin_pretrain_sweep_graphmaev2_Caco2_Wang."""
    prefix = "gin_pretrain_sweep_"
    if project.startswith("finetune_"):
        project = project[len("finetune_") :]
    if not project.startswith(prefix):
        return "unknown", project
    rest = project[len(prefix) :]
    parts = rest.split("_", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return rest, "unknown"
