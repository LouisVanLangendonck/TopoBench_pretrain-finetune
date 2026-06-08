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

Structural-feature datasets (skip ablations)
--------------------------------------------
Datasets such as IMDB-BINARY / IMDB-MULTI / REDDIT-BINARY derive their
node features from structural properties (one-hot degree, equal Gaussian).
For these datasets the feature–structure split is meaningless, so all
ablation performances are set to ``null`` and importances to
``feature_importance=0, structural_importance=1``.
"""

from __future__ import annotations

import math
import sys
import tempfile
import time
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
    """Prepend ablation pre-transforms to the base transforms config.

    The resulting config has the ablation transform(s) listed FIRST so they
    are applied before the PSE encoding steps (LapPE / RWSE).

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
        Modified transforms config with ablation transforms prepended.
    """
    if ablation == "baseline":
        return base_transforms_cfg  # no modification

    ablation_entries: dict = {}
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
        base_dict = OmegaConf.to_container(base_transforms_cfg, resolve=True)
    else:
        base_dict = {}

    combined = OmegaConf.create({**ablation_entries, **base_dict})
    return combined


# ─────────────────────────────────────────────────────────────────────────────
# Datamodule construction
# ─────────────────────────────────────────────────────────────────────────────

def _build_ablation_datamodule(
    cfg: DictConfig,
    ablation: str,
    ablation_seed: int = 42,
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
) -> tuple[float | None, str, str]:
    """Train a fresh gpse_backbone and return (test_performance, monitor_metric, monitor_mode).

    Parameters
    ----------
    train_cfg :
        Must contain ``max_epochs``, ``patience``, ``seed``, ``device``.
    """
    from lightning import Trainer
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

    from topobench.callbacks.best_epoch_metrics import BestEpochMetricsCallback

    device = torch.device(train_cfg.get("device", "cpu"))
    max_epochs = int(train_cfg.get("max_epochs", 100))
    patience = int(train_cfg.get("patience", 20))
    seed = int(train_cfg.get("seed", 42))

    cfg = _compose_cfg(dataset, model, pretraining, data_seed)
    monitor_metric, monitor_mode = _get_monitor(cfg)

    print(f"    [{ablation}] Building datamodule …")
    datamodule = _build_ablation_datamodule(cfg, ablation, ablation_seed)

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
