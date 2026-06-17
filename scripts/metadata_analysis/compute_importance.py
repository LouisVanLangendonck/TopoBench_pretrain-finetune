"""Feature and structural importance estimation via gpse_backbone ablations.

For each dataset we run four supervised-training experiments with a fresh
``gpse_backbone`` model (no pretraining):

    baseline          – normal data, full transforms (PSEs on real graph)
    random_features   – node features replaced by i.i.d. Gaussian noise
                        BEFORE PSE computation; PSEs still reflect real topology
    shuffled_edges    – edges randomly rewired BEFORE PSE computation;
                        PSEs reflect the new random topology
    both              – both ablations applied simultaneously

Importance scores
-----------------
Let  best  = baseline performance
     worst = both_ablation performance (pure noise signal)

    PERCENTAGE(x) = (x − worst) / (best − worst)

    task_feature_importance    = 1 − PERCENTAGE(random_features_perf)
    task_structural_importance = 1 − PERCENTAGE(shuffled_edges_perf)

The formula is direction-agnostic: it works for both higher-is-better
(accuracy, ROC-AUC) and lower-is-better (MAE, RMSE) metrics because
``worst`` and ``best`` are defined semantically.

Injection point: BEFORE CombinedPSEs
--------------------------------------
Ablation transforms are injected **immediately before** the ``CombinedPSEs``
key in the transform pipeline (rather than prepended at the very start).
This ensures that any dataset-specific feature-generating transforms that run
first are respected:

- Molecular datasets (raw features → CombinedPSEs):
    [RandomizeNodeFeatures / ShuffleEdges] → CombinedPSEs
    → PSEs see real topology; semantic features are destroyed / rewired.

- IMDB-BINARY / IMDB-MULTI (node_degrees → one_hot_degree → CombinedPSEs):
    node_degrees → one_hot_degree → [ablation] → CombinedPSEs
    → Structural degree features are generated first, then the ablation is
      applied, so PSEs see the real (or shuffled) graph post-feature-generation.

- REDDIT-BINARY (equal_gaus_features → CombinedPSEs):
    equal_gaus_features → [ablation] → CombinedPSEs
    → Equal Gaussian features are assigned first, then optionally shuffled.

Structural-feature datasets (``structural_feature_datasets``)
--------------------------------------------------------------
If a dataset name is listed in ``structural_feature_datasets`` (config), all
ablation performances are set to ``null`` and importances to
``feature_importance=0, structural_importance=1`` without running any
experiments.  By default this list is empty — IMDB/REDDIT are now handled
via the injection-before-CombinedPSEs approach above.

New robust definition (``*_new`` scores)
----------------------------------------
Five ablations for **all** datasets (ignoring ``structural_feature_datasets``):

    baseline           – all edges, all features (real)
    no_edge_no_feat    – no edges, equal Gaussian features  (absolute zero)
    no_edge_all_feat   – no edges, real features
    all_edge_no_feat   – all edges, equal Gaussian features (structural ceiling)
    sub_edge_no_feat   – 75 % edges, equal Gaussian features

    feature_importance_new    = (P_no_edge_all_feat − P_no_edge_no_feat)
                                / (P_all_edge_all_feat − P_no_edge_no_feat)
    structural_importance_new = (P_all_edge_no_feat − P_sub_edge_no_feat)
                                / (P_all_edge_no_feat − P_no_edge_no_feat)

Both formulas flip numerator/denominator signs for lower-is-better metrics.

Edge ablations that neutralise features always pair with equal Gaussian
(``EqualGausFeatures`` or ``EqualGausNodeEdgeFeatures``).
"""

from __future__ import annotations

import math
import sys
import tempfile
import time
import json
from pathlib import Path
from typing import Any

# ── project root on sys.path ─────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch
from omegaconf import DictConfig, OmegaConf


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

ABLATION_TYPES = ["baseline", "random_features", "shuffled_edges", "both"]

# JSON key names for each ablation's performance value
PERF_KEYS = {
    "baseline":         "gpse_backbone_baseline_performance",
    "random_features":  "gpse_backbone_random_features_performance",
    "shuffled_edges":   "gpse_backbone_shuffled_edges_performance",
    "both":             "gpse_backbone_random_features_shuffled_edges_performance",
}

# ── New (robust) importance definition ───────────────────────────────────────
NEW_ABLATION_TYPES = [
    "baseline",
    "no_edge_no_feat",
    "no_edge_all_feat",
    "all_edge_no_feat",
    "sub_edge_no_feat",
]

NEW_PERF_KEYS = {
    "baseline":          "gpse_backbone_baseline_performance",
    "no_edge_no_feat":   "gpse_backbone_no_edge_no_feat_performance",
    "no_edge_all_feat":  "gpse_backbone_no_edge_all_feat_performance",
    "all_edge_no_feat":  "gpse_backbone_all_edge_no_feat_performance",
    "sub_edge_no_feat":  "gpse_backbone_sub_edge_no_feat_performance",
}

_BEST_HP_JSON_KEY = "GPSE_backbone_hyperparam_best_test"


def load_dataset_json(output_dir: Path, dataset_name: str) -> dict:
    """Load the full metadata JSON for a dataset, or ``{}`` if missing."""
    json_path = output_dir / f"{dataset_name}.json"
    if not json_path.exists():
        return {}
    try:
        with open(json_path) as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def load_best_hyperparams(output_dir: Path, dataset_name: str) -> dict | None:
    """Return the tuned-hyperparam block from a dataset JSON, if present."""
    hp = load_dataset_json(output_dir, dataset_name).get(_BEST_HP_JSON_KEY)
    return hp if isinstance(hp, dict) else None


def best_test_baseline_performance(output_dir: Path, dataset_name: str) -> float | None:
    """Return ``GPSE_backbone_hyperparam_best_test.test_score`` when available."""
    hp = load_best_hyperparams(output_dir, dataset_name)
    if hp is None:
        return None
    score = hp.get("test_score")
    if score is None or (isinstance(score, float) and math.isnan(score)):
        return None
    return float(score)


def hyperparam_overrides_from_json(hp: dict) -> list[str]:
    """Build Hydra overrides from a ``GPSE_backbone_hyperparam_best_test`` entry."""
    mapping = (
        ("best_hidden_channels", "++model.feature_encoder.out_channels"),
        ("best_num_layers", "++model.backbone.num_layers"),
        ("best_weight_decay", "++optimizer.parameters.weight_decay"),
        ("best_lr", "++optimizer.parameters.lr"),
    )
    overrides: list[str] = []
    for json_key, hydra_key in mapping:
        val = hp.get(json_key)
        if val is not None:
            overrides.append(f"{hydra_key}={val}")
    return overrides


def _first_sample_graph(raw_dataset) -> Any:
    """Return the first graph from a raw dataset."""
    if hasattr(raw_dataset, "__getitem__"):
        return raw_dataset[0]
    return next(iter(raw_dataset))


def _pipeline_uses_equal_gaus(cfg: DictConfig) -> bool:
    """True when the dataset pipeline already assigns ``EqualGausFeatures``."""
    transforms = cfg.get("transforms")
    if transforms is None:
        return False

    container = OmegaConf.to_container(transforms, resolve=True)
    if not isinstance(container, dict):
        return False

    for key, val in container.items():
        if "equal_gaus" in str(key).lower():
            return True
        if isinstance(val, dict) and val.get("transform_name") == "EqualGausFeatures":
            return True
    return False


def _dataset_has_semantic_features(cfg: DictConfig, sample: Any) -> bool:
    """True when the dataset carries informative node/edge features to neutralise."""
    if _pipeline_uses_equal_gaus(cfg):
        return False

    from omegaconf import ListConfig

    nf = cfg.dataset.parameters.get("num_features")
    if isinstance(nf, (list, ListConfig)):
        return True
    if sample is None:
        return False
    x = getattr(sample, "x", None)
    return x is not None and x.numel() > 0


def _resolve_equal_gaus_params(
    cfg: DictConfig, raw_dataset
) -> tuple[int, int | None, bool]:
    """Resolve equal-Gaussian dims and which transform to use.

    Returns
    -------
    (num_node_features, num_edge_features_or_None, use_node_edge_transform)

    Datasets **without** semantic features (DD, IMDB, …) keep the original
    ``EqualGausFeatures`` transform.  Datasets **with** real ``x`` / ``edge_attr``
    use ``EqualGausNodeEdgeFeatures`` so both are neutralised.
    """
    from omegaconf import ListConfig

    sample = _first_sample_graph(raw_dataset)
    nf = cfg.dataset.parameters.get("num_features")

    if _dataset_has_semantic_features(cfg, sample):
        if isinstance(nf, (list, ListConfig)):
            node_dim = int(nf[0])
            edge_dim = int(nf[1]) if len(nf) > 1 else None
        elif nf is not None:
            node_dim = int(nf)
            edge_dim = None
        else:
            x = sample.x
            node_dim = int(x.shape[1]) if x.dim() > 1 else 1

        if edge_dim is None:
            edge_attr = getattr(sample, "edge_attr", None)
            if edge_attr is not None and edge_attr.numel() > 0:
                edge_dim = (
                    int(edge_attr.shape[-1])
                    if edge_attr.dim() > 1
                    else 1
                )

        return node_dim, edge_dim, True

    if isinstance(nf, (list, ListConfig)):
        node_dim = int(nf[0])
    elif nf is not None:
        node_dim = int(nf)
    else:
        node_dim = 10

    return node_dim, None, False


# ─────────────────────────────────────────────────────────────────────────────
# Hydra config helpers
# ─────────────────────────────────────────────────────────────────────────────

def _compose_cfg(
    dataset: str,
    model: str,
    pretraining: str,
    data_seed: int,
    extra_overrides: list[str] | None = None,
) -> DictConfig:
    """Compose a Hydra DictConfig for the given dataset / model / pretraining."""
    import rootutils
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    rootutils.setup_root(_PROJECT_ROOT, indicator=".project-root", pythonpath=True)

    from topobench.utils.config_resolvers import register_all_resolvers
    register_all_resolvers()

    overrides = [
        f"dataset={dataset}",
        f"model={model}",
        f"pretraining={pretraining}",
        f"++dataset.split_params.data_seed={data_seed}",
        "logger=[]",
    ]
    if extra_overrides:
        overrides.extend(extra_overrides)

    config_dir = str(_PROJECT_ROOT / "configs")
    GlobalHydra.instance().clear()
    with initialize_config_dir(
        config_dir=config_dir, version_base="1.3", job_name="importance"
    ):
        cfg = compose(config_name="run", overrides=overrides)
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Transform config manipulation
# ─────────────────────────────────────────────────────────────────────────────

def _build_ablation_transforms_config(
    base_transforms_cfg: DictConfig | None,
    ablation: str,
    ablation_seed: int = 42,
) -> DictConfig | None:
    """Inject ablation pre-transforms immediately before ``CombinedPSEs``.

    The injection point is chosen as the position of the ``CombinedPSEs`` key
    in the transform pipeline.  This guarantees that any dataset-specific
    feature-generation transforms (e.g. ``node_degrees``,
    ``one_hot_node_degree_features``, ``equal_gaus_features``) run first,
    so the ablation acts on the features that the model would actually see
    during training.

    - For datasets where ``CombinedPSEs`` is the only transform (molecular),
      the ablation transforms are effectively prepended — same as before.
    - For IMDB/REDDIT-style datasets the ablation transforms are inserted
      *between* the structural-feature step and the PSE encoding step.
    - If no ``CombinedPSEs`` key is found the ablation transforms are
      prepended as a safe fallback.

    Parameters
    ----------
    base_transforms_cfg :
        The ``cfg.transforms`` DictConfig from the composed Hydra config, or
        ``None`` if no transforms are configured.
    ablation :
        One of ``"baseline"``, ``"random_features"``, ``"shuffled_edges"``,
        ``"both"``.
    ablation_seed :
        Seed passed to the ablation transforms for reproducibility.

    Returns
    -------
    DictConfig | None
        Modified transforms config with ablation transforms injected before
        ``CombinedPSEs``.
    """
    if ablation == "baseline":
        return base_transforms_cfg  # no modification

    ablation_entries: dict[str, dict] = {}
    if ablation in ("random_features", "both"):
        ablation_entries["RandomizeNodeFeatures_ablation"] = {
            "transform_name": "RandomizeNodeFeatures",
            "seed": ablation_seed,
        }
    if ablation in ("shuffled_edges", "both"):
        ablation_entries["ShuffleEdges_ablation"] = {
            "transform_name": "ShuffleEdges",
            "seed": ablation_seed,
        }

    if base_transforms_cfg is not None:
        base_dict: dict = OmegaConf.to_container(base_transforms_cfg, resolve=True)
    else:
        base_dict = {}

    # Find the CombinedPSEs key (exact match or key that contains "CombinedPSEs")
    combined_pses_key: str | None = None
    for key in base_dict:
        if key == "CombinedPSEs" or "CombinedPSEs" in str(key):
            combined_pses_key = key
            break

    if combined_pses_key is None:
        # No CombinedPSEs found — fall back to prepending at the start
        result_dict = {**ablation_entries, **base_dict}
    else:
        # Rebuild dict in order, inserting ablation transforms just before
        # the CombinedPSEs entry.
        result_dict = {}
        for key, val in base_dict.items():
            if key == combined_pses_key:
                result_dict.update(ablation_entries)
            result_dict[key] = val

    return OmegaConf.create(result_dict)


    return OmegaConf.create(result_dict)


def _equal_gaus_ablation_entry(
    num_features: int,
    num_edge_features: int | None,
    use_node_edge_equal_gaus: bool,
    equal_gaus_mean: float,
    equal_gaus_std: float,
    key_suffix: str = "",
) -> tuple[str, dict]:
    """Build a single equal-Gaussian ablation transform entry."""
    if use_node_edge_equal_gaus:
        key = f"EqualGausNodeEdgeFeatures_ablation{key_suffix}"
        entry: dict = {
            "transform_name": "EqualGausNodeEdgeFeatures",
            "mean": equal_gaus_mean,
            "std": equal_gaus_std,
            "num_features": num_features,
        }
        if num_edge_features is not None:
            entry["num_edge_features"] = num_edge_features
    else:
        key = f"EqualGausFeatures_ablation{key_suffix}"
        entry = {
            "transform_name": "EqualGausFeatures",
            "mean": equal_gaus_mean,
            "std": equal_gaus_std,
            "num_features": num_features,
        }
    return key, entry


def _build_new_ablation_transforms_config(
    base_transforms_cfg: DictConfig | None,
    ablation: str,
    ablation_seed: int = 42,
    sub_edges_frac: float = 0.75,
    num_features: int | None = None,
    num_edge_features: int | None = None,
    use_node_edge_equal_gaus: bool = False,
    equal_gaus_mean: float = 0.0,
    equal_gaus_std: float = 0.1,
) -> DictConfig | None:
    """Inject new-definition ablation transforms immediately before ``CombinedPSEs``.

    Ablation types
    --------------
    baseline           – no modification
    no_edge_no_feat    – no edges + equal Gaussian features (absolute zero)
    no_edge_all_feat   – no edges + real features
    all_edge_no_feat   – full graph + equal Gaussian features
    sub_edge_no_feat   – subsampled edges + equal Gaussian features
    """
    if ablation == "baseline":
        return base_transforms_cfg

    if num_features is None and ablation != "no_edge_all_feat":
        raise ValueError(
            f"num_features is required for the {ablation} ablation"
        )

    ablation_entries: dict[str, dict] = {}

    if ablation == "no_edge_all_feat":
        ablation_entries["RemoveEdges_ablation"] = {
            "transform_name": "RemoveEdges",
        }
    elif ablation == "no_edge_no_feat":
        ablation_entries["RemoveEdges_ablation"] = {
            "transform_name": "RemoveEdges",
        }
        gaus_key, gaus_entry = _equal_gaus_ablation_entry(
            num_features, num_edge_features, use_node_edge_equal_gaus,
            equal_gaus_mean, equal_gaus_std,
        )
        ablation_entries[gaus_key] = gaus_entry
    elif ablation == "all_edge_no_feat":
        gaus_key, gaus_entry = _equal_gaus_ablation_entry(
            num_features, num_edge_features, use_node_edge_equal_gaus,
            equal_gaus_mean, equal_gaus_std,
        )
        ablation_entries[gaus_key] = gaus_entry
    elif ablation == "sub_edge_no_feat":
        ablation_entries["SubsampleEdges_ablation"] = {
            "transform_name": "SubsampleEdges",
            "frac": sub_edges_frac,
            "seed": ablation_seed,
        }
        gaus_key, gaus_entry = _equal_gaus_ablation_entry(
            num_features, num_edge_features, use_node_edge_equal_gaus,
            equal_gaus_mean, equal_gaus_std,
        )
        ablation_entries[gaus_key] = gaus_entry
    else:
        raise ValueError(f"Unknown new ablation type: {ablation!r}")

    if base_transforms_cfg is not None:
        base_dict: dict = OmegaConf.to_container(base_transforms_cfg, resolve=True)
    else:
        base_dict = {}

    combined_pses_key: str | None = None
    for key in base_dict:
        if key == "CombinedPSEs" or "CombinedPSEs" in str(key):
            combined_pses_key = key
            break

    if combined_pses_key is None:
        result_dict = {**ablation_entries, **base_dict}
    else:
        result_dict = {}
        for key, val in base_dict.items():
            if key == combined_pses_key:
                result_dict.update(ablation_entries)
            result_dict[key] = val

    return OmegaConf.create(result_dict)


# ─────────────────────────────────────────────────────────────────────────────
# Datamodule construction
# ─────────────────────────────────────────────────────────────────────────────

def _build_ablation_datamodule(
    cfg: DictConfig,
    ablation: str,
    ablation_seed: int = 42,
    *,
    ablation_family: str = "legacy",
    sub_edges_frac: float = 0.75,
    equal_gaus_mean: float = 0.0,
    equal_gaus_std: float = 0.1,
) -> Any:
    """Build a TBDataloader with optional ablation pre-transforms.

    For ``ablation != "baseline"`` the ablation transform(s) are injected
    *before* the standard model/dataset pre-transforms (PSEs) so that:

    - ``random_features``: PSEs see the real graph; features are noise.
    - ``shuffled_edges``:  PSEs see the random graph; features are real.
    - ``both``:            PSEs see the random graph; features are noise.

    The PreProcessor caches each variant in its own directory (determined by
    a hash of the full transform list) so there is no cache collision.
    """
    import hydra
    from topobench.data.preprocessor import PreProcessor
    from topobench.dataloader import TBDataloader

    loader = hydra.utils.instantiate(cfg.dataset.loader)
    raw_dataset, dataset_dir = loader.load()

    if ablation_family == "new":
        num_features, num_edge_features, use_node_edge_equal_gaus = (
            _resolve_equal_gaus_params(cfg, raw_dataset)
        )
        transforms_cfg = _build_new_ablation_transforms_config(
            cfg.get("transforms"),
            ablation,
            ablation_seed,
            sub_edges_frac=sub_edges_frac,
            num_features=num_features,
            num_edge_features=num_edge_features,
            use_node_edge_equal_gaus=use_node_edge_equal_gaus,
            equal_gaus_mean=equal_gaus_mean,
            equal_gaus_std=equal_gaus_std,
        )
    else:
        transforms_cfg = _build_ablation_transforms_config(
            cfg.get("transforms"), ablation, ablation_seed
        )

    preprocessor = PreProcessor(raw_dataset, dataset_dir, transforms_cfg)
    train_ds, val_ds, test_ds = preprocessor.load_dataset_splits(
        cfg.dataset.split_params
    )

    return TBDataloader(
        dataset_train=train_ds,
        dataset_val=val_ds,
        dataset_test=test_ds,
        **cfg.dataset.get("dataloader_params", {}),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Model construction
# ─────────────────────────────────────────────────────────────────────────────

def _build_fresh_model(cfg: DictConfig) -> Any:
    """Instantiate a fresh gpse_backbone model (no pretrained weights)."""
    import hydra

    return hydra.utils.instantiate(
        cfg.model,
        evaluator=cfg.evaluator,
        optimizer=cfg.optimizer,
        loss=cfg.loss,
        learning_setting=cfg.dataset.split_params.get("learning_setting", None),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Monitor metric helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_monitor(cfg: DictConfig) -> tuple[str, str]:
    """Return (monitor_metric, monitor_mode) for the supervised downstream task."""
    from topobench.utils.config_resolvers import get_monitor_metric, get_monitor_mode

    params = cfg.dataset.parameters
    task = str(params.task)
    metric = str(params.monitor_metric)
    return get_monitor_metric(task, metric), get_monitor_mode(task, metric)


def _extract_test_scalar(
    test_metrics: dict[str, float], monitor_metric: str
) -> float | None:
    """Extract the single scalar performance value from test_metrics.

    Translates the val-phase monitor metric name to its test-phase equivalent
    (``"val/roc_auc"`` → ``"test/roc_auc"``).
    """
    test_key = monitor_metric.replace("val/", "test/")
    value = test_metrics.get(test_key)
    if value is not None:
        return float(value)

    # Fallback: find any test/* key whose suffix matches
    suffix = monitor_metric.split("/", 1)[-1]
    for k, v in test_metrics.items():
        if k.startswith("test/") and k.endswith(suffix):
            return float(v)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Single-ablation training run
# ─────────────────────────────────────────────────────────────────────────────

def _run_one_ablation(
    dataset: str,
    model: str,
    pretraining: str,
    data_seed: int,
    ablation: str,
    ablation_seed: int,
    train_cfg: dict,
    extra_cfg_overrides: list[str] | None = None,
    *,
    ablation_family: str = "legacy",
    sub_edges_frac: float = 0.75,
    equal_gaus_mean: float = 0.0,
    equal_gaus_std: float = 0.1,
) -> tuple[float | None, str, str]:
    """Train a fresh gpse_backbone and return (test_performance, monitor_metric, monitor_mode).

    Parameters
    ----------
    train_cfg :
        Must contain ``max_epochs``, ``patience``, ``seed``, ``device``.
    extra_cfg_overrides :
        Additional Hydra overrides forwarded to ``_compose_cfg`` (e.g.
        ``["model.backbone.num_layers=3"]``).
    """
    from lightning import Trainer
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

    from topobench.callbacks.best_epoch_metrics import BestEpochMetricsCallback

    device = torch.device(train_cfg.get("device", "cpu"))
    max_epochs = int(train_cfg.get("max_epochs", 100))
    patience = int(train_cfg.get("patience", 20))
    seed = int(train_cfg.get("seed", 42))

    cfg_overrides: list[str] = list(train_cfg.get("hyperparam_overrides") or [])
    if extra_cfg_overrides:
        cfg_overrides.extend(extra_cfg_overrides)

    cfg = _compose_cfg(
        dataset,
        model,
        pretraining,
        data_seed,
        cfg_overrides or None,
    )
    monitor_metric, monitor_mode = _get_monitor(cfg)

    print(f"    [{ablation}] Building datamodule …")
    datamodule = _build_ablation_datamodule(
        cfg,
        ablation,
        ablation_seed,
        ablation_family=ablation_family,
        sub_edges_frac=sub_edges_frac,
        equal_gaus_mean=equal_gaus_mean,
        equal_gaus_std=equal_gaus_std,
    )

    print(f"    [{ablation}] Building fresh model …")
    fresh_model = _build_fresh_model(cfg)

    accel = "cuda" if device.type == "cuda" else "cpu"
    devs = (
        [device.index if device.index is not None else 0]
        if accel == "cuda"
        else 1
    )

    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_cb = ModelCheckpoint(
            dirpath=tmpdir,
            monitor=monitor_metric,
            mode=monitor_mode,
            save_top_k=1,
        )
        fit_trainer = Trainer(
            accelerator=accel,
            devices=devs,
            max_epochs=max_epochs,
            logger=False,
            enable_progress_bar=True,
            num_sanity_val_steps=0,
            callbacks=[
                EarlyStopping(
                    monitor=monitor_metric, patience=patience, mode=monitor_mode
                ),
                ckpt_cb,
                BestEpochMetricsCallback(
                    monitor=monitor_metric, mode=monitor_mode
                ),
            ],
        )
        fresh_model.to(device)
        fit_trainer.fit(model=fresh_model, datamodule=datamodule)

        if ckpt_cb.best_model_path:
            ckpt = torch.load(
                ckpt_cb.best_model_path,
                map_location=device,
                weights_only=False,
            )
            fresh_model.load_state_dict(ckpt["state_dict"], strict=True)

    test_trainer = Trainer(
        accelerator=accel,
        devices=devs,
        logger=False,
        enable_progress_bar=False,
    )
    results = test_trainer.test(
        model=fresh_model, datamodule=datamodule, verbose=False
    )
    test_metrics = results[0] if results else {}
    performance = _extract_test_scalar(test_metrics, monitor_metric)
    return performance, monitor_metric, monitor_mode


# ─────────────────────────────────────────────────────────────────────────────
# Importance score computation
# ─────────────────────────────────────────────────────────────────────────────

def _compute_importance_scores(
    baseline: float | None,
    random_feat: float | None,
    shuffled_edge: float | None,
    both: float | None,
    monitor_mode: str,
) -> tuple[float | None, float | None]:
    """Compute feature and structural importance scores.

    Uses ``both`` as the "worst" baseline (no meaningful signal) and
    ``baseline`` as "best" (full information).

    PERCENTAGE(x) = (x − worst) / (best − worst)

    This formula is direction-agnostic: for higher-is-better metrics
    (best > worst) both numerator and denominator are positive; for
    lower-is-better metrics (best < worst) both are negative, so the ratio
    is still in [0, 1].

    Returns
    -------
    (feature_importance, structural_importance)
        Each is a float in [0, 1], or ``None`` if any required value is
        missing.
    """
    if any(v is None or (isinstance(v, float) and math.isnan(v))
           for v in [baseline, random_feat, shuffled_edge, both]):
        return None, None

    best = baseline
    worst = both

    if abs(best - worst) < 1e-8:
        # No signal at all — importances undefined
        return None, None

    pct_feat = (random_feat - worst) / (best - worst)
    pct_struct = (shuffled_edge - worst) / (best - worst)

    feat_imp = float(max(0.0, min(1.0, 1.0 - pct_feat)))
    struct_imp = float(max(0.0, min(1.0, 1.0 - pct_struct)))
    return feat_imp, struct_imp


def _compute_importance_scores_new(
    all_edge_all_feat: float | None,
    no_edge_no_feat: float | None,
    no_edge_all_feat: float | None,
    all_edge_no_feat: float | None,
    sub_edge_no_feat: float | None,
    *,
    higher_is_better: bool = True,
) -> tuple[float | None, float | None]:
    """Compute robust feature / structural importance (``_new`` definition).

    Absolute zero: ``no_edge_no_feat`` (no structure, no informative features).

    Feature importance — share of the zero→baseline gap recovered with real
    features but no edges::

        (P_no_edge_all_feat − P_no_edge_no_feat)
        / (P_all_edge_all_feat − P_no_edge_no_feat)          [higher-is-better]

    Structural importance — share of the zero→full-structure gap **lost** when
    edges are subsampled (neutral features throughout)::

        (P_all_edge_no_feat − P_sub_edge_no_feat)
        / (P_all_edge_no_feat − P_no_edge_no_feat)            [higher-is-better]

    For lower-is-better metrics both numerators and denominators are flipped so
  that a larger score always means more importance.
    """
    if any(
        v is None or (isinstance(v, float) and math.isnan(v))
        for v in [
            all_edge_all_feat,
            no_edge_no_feat,
            no_edge_all_feat,
            all_edge_no_feat,
            sub_edge_no_feat,
        ]
    ):
        return None, None

    zero = no_edge_no_feat

    if higher_is_better:
        feat_num = no_edge_all_feat - zero
        feat_denom = all_edge_all_feat - zero
        struct_num = all_edge_no_feat - sub_edge_no_feat
        struct_denom = all_edge_no_feat - zero
    else:
        feat_num = zero - no_edge_all_feat
        feat_denom = zero - all_edge_all_feat
        struct_num = sub_edge_no_feat - all_edge_no_feat
        struct_denom = zero - all_edge_no_feat

    if abs(feat_denom) < 1e-8:
        feat_imp = None
    else:
        feat_imp = float(max(0.0, min(1.0, feat_num / feat_denom)))

    if abs(struct_denom) < 1e-8:
        struct_imp = None
    else:
        struct_imp = float(max(0.0, min(1.0, struct_num / struct_denom)))

    return feat_imp, struct_imp


def _higher_is_better_from_json(existing: dict) -> bool | None:
    """Infer metric direction from a dataset JSON payload."""
    hp = existing.get(_BEST_HP_JSON_KEY)
    if isinstance(hp, dict) and "higher_better" in hp:
        return bool(hp["higher_better"])
    mode = existing.get("gpse_backbone_monitor_mode")
    if mode in ("max", "min"):
        return mode == "max"
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def compute_dataset_importance(
    dataset: str,
    model: str,
    pretraining: str,
    data_seed: int,
    train_cfg: dict,
    structural_feature_datasets: set[str],
    ablation_seed: int = 42,
) -> dict:
    """Run all ablation experiments for one dataset and return importance scores.

    Parameters
    ----------
    dataset :
        Hydra dataset override string, e.g. ``"graph/MUTAG"``.
    model :
        Hydra model override string, e.g. ``"graph/gpse_backbone"``.
    pretraining :
        Pretraining override (usually ``"none"`` for supervised training).
    data_seed :
        Random seed for the train/val/test split.
    train_cfg :
        Dict with keys ``max_epochs``, ``patience``, ``seed``, ``device``.
    structural_feature_datasets :
        Set of dataset *names* (the part after ``/``) for which node features
        are derived from structural properties.  Ablations are skipped for
        these datasets; feature_importance is set to 0 and
        structural_importance to 1.
    ablation_seed :
        Seed passed to ``RandomizeNodeFeatures`` / ``ShuffleEdges``.

    Returns
    -------
    dict
        Top-level keys ready to be merged into the dataset JSON:
        - ``gpse_backbone_baseline_performance``
        - ``gpse_backbone_random_features_performance``
        - ``gpse_backbone_shuffled_edges_performance``
        - ``gpse_backbone_random_features_shuffled_edges_performance``
        - ``gpse_backbone_monitor_metric``
        - ``gpse_backbone_monitor_mode``
        - ``task_feature_importance``
        - ``task_structural_importance``
    """
    dataset_name = dataset.split("/")[-1]
    result: dict = {}

    # ── Structural-feature datasets: skip ablations ───────────────────────────
    if dataset_name in structural_feature_datasets:
        print(
            f"  [{dataset_name}] structural-feature dataset — "
            "skipping ablations (feature_importance=0, structural_importance=1)"
        )
        for key in PERF_KEYS.values():
            result[key] = None
        result["gpse_backbone_monitor_metric"] = None
        result["gpse_backbone_monitor_mode"] = None
        result["task_feature_importance"] = 0.0
        result["task_structural_importance"] = 1.0
        return result

    # ── Run the 4 experiments ─────────────────────────────────────────────────
    performances: dict[str, float | None] = {}
    monitor_metric: str | None = None
    monitor_mode: str | None = None

    for ablation in ABLATION_TYPES:
        t0 = time.perf_counter()
        print(f"  [{dataset_name}] Running ablation: {ablation} …")
        try:
            perf, mm, mmmode = _run_one_ablation(
                dataset=dataset,
                model=model,
                pretraining=pretraining,
                data_seed=data_seed,
                ablation=ablation,
                ablation_seed=ablation_seed,
                train_cfg=train_cfg,
            )
            performances[ablation] = perf
            if monitor_metric is None:
                monitor_metric = mm
                monitor_mode = mmmode
            elapsed = time.perf_counter() - t0
            print(
                f"  [{dataset_name}] {ablation} → {perf}  "
                f"({elapsed:.1f}s, metric={mm})"
            )
        except Exception as exc:
            import traceback
            print(f"  [{dataset_name}] ERROR in ablation={ablation}: {exc}")
            traceback.print_exc()
            performances[ablation] = None

    # ── Store raw performance values ──────────────────────────────────────────
    for ablation, key in PERF_KEYS.items():
        result[key] = performances.get(ablation)

    result["gpse_backbone_monitor_metric"] = monitor_metric
    result["gpse_backbone_monitor_mode"] = monitor_mode

    # ── Compute importance scores ─────────────────────────────────────────────
    feat_imp, struct_imp = _compute_importance_scores(
        baseline=performances.get("baseline"),
        random_feat=performances.get("random_features"),
        shuffled_edge=performances.get("shuffled_edges"),
        both=performances.get("both"),
        monitor_mode=monitor_mode or "max",
    )
    result["task_feature_importance"] = feat_imp
    result["task_structural_importance"] = struct_imp

    return result


def compute_dataset_importance_new(
    dataset: str,
    model: str,
    pretraining: str,
    data_seed: int,
    train_cfg: dict,
    ablation_seed: int = 42,
    sub_edges_frac: float = 0.75,
    equal_gaus_mean: float = 0.0,
    equal_gaus_std: float = 0.1,
    output_dir: Path | str | None = None,
    use_best_test_baseline: bool = False,
) -> dict:
    """Run the new-definition ablation experiments for one dataset.

    Unlike ``compute_dataset_importance``, this runs for *all* datasets
    (including structural-feature datasets) with five ablations centred on
    ``no_edge_no_feat`` as the absolute zero (see module docstring).

    When *use_best_test_baseline* is ``True``, the baseline performance is
    taken from ``GPSE_backbone_hyperparam_best_test.test_score`` in the
    dataset JSON (no baseline training run).  Any ablation whose performance
    key is already present in the JSON is skipped; importance scores are
    always recomputed from the collected performances.

    Returns
    -------
    dict
        Top-level keys ready to be merged into the dataset JSON, including
        ``task_feature_importance_new`` and ``task_structural_importance_new``.
    """
    dataset_name = dataset.split("/")[-1]
    result: dict = {}
    performances: dict[str, float | None] = {}
    monitor_metric: str | None = None
    monitor_mode: str | None = None

    existing: dict = {}
    if output_dir is not None:
        existing = load_dataset_json(Path(output_dir), dataset_name)

    # ── Baseline: tuned sweep best test score (preferred) or cached value ───
    if use_best_test_baseline:
        tuned_baseline = best_test_baseline_performance(
            Path(output_dir), dataset_name
        )
        if tuned_baseline is not None:
            performances["baseline"] = tuned_baseline
            print(
                f"  [{dataset_name}] baseline (new) → {tuned_baseline}  "
                f"(from {_BEST_HP_JSON_KEY}.test_score, skipped training)"
            )
        hp = existing.get(_BEST_HP_JSON_KEY) or {}
        if hp.get("monitor_metric"):
            monitor_metric = f"val/{hp['monitor_metric']}"
            monitor_mode = "max" if hp.get("higher_better", True) else "min"

    if performances.get("baseline") is None:
        cached_baseline = existing.get(NEW_PERF_KEYS["baseline"])
        if cached_baseline is not None:
            performances["baseline"] = float(cached_baseline)
            print(
                f"  [{dataset_name}] baseline (new) → {cached_baseline}  "
                f"(cached, skipped training)"
            )

    ablations_to_run = [
        a for a in NEW_ABLATION_TYPES if a not in performances
    ]

    for ablation in ablations_to_run:
        perf_key = NEW_PERF_KEYS[ablation]
        cached = existing.get(perf_key)
        if cached is not None and not (
            isinstance(cached, float) and math.isnan(cached)
        ):
            performances[ablation] = float(cached)
            print(
                f"  [{dataset_name}] {ablation} (new) → {cached}  "
                f"(cached in JSON, skipped training)"
            )
            continue

        t0 = time.perf_counter()
        print(f"  [{dataset_name}] Running new ablation: {ablation} …")

        try:
            perf, mm, mmmode = _run_one_ablation(
                dataset=dataset,
                model=model,
                pretraining=pretraining,
                data_seed=data_seed,
                ablation=ablation,
                ablation_seed=ablation_seed,
                train_cfg=train_cfg,
                ablation_family="new",
                sub_edges_frac=sub_edges_frac,
                equal_gaus_mean=equal_gaus_mean,
                equal_gaus_std=equal_gaus_std,
            )
            performances[ablation] = perf
            if monitor_metric is None:
                monitor_metric = mm
                monitor_mode = mmmode
            elapsed = time.perf_counter() - t0
            print(
                f"  [{dataset_name}] {ablation} (new) → {perf}  "
                f"({elapsed:.1f}s, metric={mm})"
            )
        except Exception as exc:
            import traceback
            print(f"  [{dataset_name}] ERROR in new ablation={ablation}: {exc}")
            traceback.print_exc()
            performances[ablation] = None

    if monitor_metric is None:
        monitor_metric = existing.get("gpse_backbone_monitor_metric")
        monitor_mode = existing.get("gpse_backbone_monitor_mode")

    for ablation, key in NEW_PERF_KEYS.items():
        result[key] = performances.get(ablation)

    result["gpse_backbone_monitor_metric"] = monitor_metric
    result["gpse_backbone_monitor_mode"] = monitor_mode

    higher_is_better = _higher_is_better_from_json(existing)
    if higher_is_better is None:
        higher_is_better = monitor_mode != "min" if monitor_mode else True

    feat_imp, struct_imp = _compute_importance_scores_new(
        all_edge_all_feat=performances.get("baseline"),
        no_edge_no_feat=performances.get("no_edge_no_feat"),
        no_edge_all_feat=performances.get("no_edge_all_feat"),
        all_edge_no_feat=performances.get("all_edge_no_feat"),
        sub_edge_no_feat=performances.get("sub_edge_no_feat"),
        higher_is_better=higher_is_better,
    )
    result["task_feature_importance_new"] = feat_imp
    result["task_structural_importance_new"] = struct_imp

    print(
        f"  [{dataset_name}] importance_new recalculated → "
        f"feature={feat_imp}  structural={struct_imp}"
    )

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Task-depth estimation
# ─────────────────────────────────────────────────────────────────────────────

def compute_dataset_depth(
    dataset: str,
    model: str,
    pretraining: str,
    data_seed: int,
    train_cfg: dict,
    max_layers: int = 8,
    ablation_seed: int = 42,
) -> dict:
    """Estimate task depth by sweeping ``num_layers`` from 1 to *max_layers*.

    A baseline (real features + real structure) supervised run is performed for
    each layer count.  The layer count that achieves the best test-set
    performance (according to the dataset's own monitor metric and direction) is
    stored as ``task_depth``.

    Parameters
    ----------
    dataset :
        Hydra dataset override string, e.g. ``"graph/MUTAG"``.
    model :
        Hydra model override string, e.g. ``"graph/gpse_backbone"``.
    pretraining :
        Pretraining override (usually ``"none"`` for supervised training).
    data_seed :
        Random seed for the train/val/test split.
    train_cfg :
        Dict with keys ``max_epochs``, ``patience``, ``seed``, ``device``.
    max_layers :
        Upper bound of the layer sweep (inclusive).  Default 8.
    ablation_seed :
        Passed through to ``_run_one_ablation`` (unused for baseline but kept
        for API consistency).

    Returns
    -------
    dict
        Top-level keys ready to be merged into the dataset JSON:

        - ``task_depth``                    – int, best layer count (or ``null``)
        - ``task_depth_performances``       – dict mapping str(n_layers) → float|null
        - ``task_depth_monitor_metric``     – metric name used for comparison
        - ``task_depth_monitor_mode``       – ``"max"`` or ``"min"``
    """
    dataset_name = dataset.split("/")[-1]

    # Resolve monitor direction once from the base config
    base_cfg = _compose_cfg(dataset, model, pretraining, data_seed)
    monitor_metric, monitor_mode = _get_monitor(base_cfg)

    performances: dict[int, float | None] = {}

    for n_layers in range(1, max_layers + 1):
        t0 = time.perf_counter()
        print(f"  [{dataset_name}] depth sweep: num_layers={n_layers} …")
        try:
            perf, _, _ = _run_one_ablation(
                dataset=dataset,
                model=model,
                pretraining=pretraining,
                data_seed=data_seed,
                ablation="baseline",
                ablation_seed=ablation_seed,
                train_cfg=train_cfg,
                extra_cfg_overrides=[f"model.backbone.num_layers={n_layers}"],
            )
            performances[n_layers] = perf
            print(
                f"  [{dataset_name}] layers={n_layers} → {perf}  "
                f"({time.perf_counter() - t0:.1f}s)"
            )
        except Exception as exc:
            import traceback as _tb
            print(f"  [{dataset_name}] ERROR layers={n_layers}: {exc}")
            _tb.print_exc()
            performances[n_layers] = None

    # Pick the layer count with the best valid performance
    valid = {
        k: v for k, v in performances.items()
        if v is not None and not math.isnan(v)
    }
    if valid:
        best_layers: int | None = (
            min(valid, key=valid.__getitem__)
            if monitor_mode == "min"
            else max(valid, key=valid.__getitem__)
        )
    else:
        best_layers = None

    print(
        f"  [{dataset_name}] task_depth={best_layers}  "
        f"(metric={monitor_metric}, mode={monitor_mode})"
    )

    return {
        "task_depth": best_layers,
        "task_depth_performances": {str(k): v for k, v in performances.items()},
        "task_depth_monitor_metric": monitor_metric,
        "task_depth_monitor_mode": monitor_mode,
    }
