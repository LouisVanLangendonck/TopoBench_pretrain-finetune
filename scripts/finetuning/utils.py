"""Utilities for loading pretrained models from W&B and fine-tuning them.

Workflow — pretraining evaluation
----------------------------------
1. fetch_run()              – pull a W&B run (by ID or the project's first run).
2. get_hydra_overrides()    – recover the original Hydra CLI args from run metadata.
3. get_checkpoint_path()    – locate the best .ckpt from the run's output_dir.
4. compose_cfg()            – rebuild the identical Hydra DictConfig.
5. build_datamodule()       – recreate the dataset + splits (same seed → same data).
6. load_model()             – instantiate the Lightning model and load checkpoint weights.
7. evaluate()               – run Trainer.test() and return the metrics dict.

Workflow — downstream fine-tuning (pretrained → supervised)
------------------------------------------------------------
8.  extract_downstream_overrides() – strip pretraining args, keep arch/data/optim.
9.  build_downstream_model()       – swap wrapper+readout for GNNWrapper+DownstreamReadOut.
10. apply_finetuning_mode()        – freeze / randomize as needed for the 4 modes.
11. make_subset_datamodule()       – few-shot subset of the training split.
12. run_finetune_experiment()      – fit + load best ckpt + test → return metrics.

Supported pretraining methods
------------------------------
GraphMAEv2  Uses ``encoder_ema`` (EMA teacher, trained on clean unmasked inputs).
BGRL        Uses ``online_encoder`` (directly trained; ``backbone`` is the frozen EMA target).
DGI         Uses ``backbone`` (sole encoder for positive and negative views).
GraphCL     Uses ``backbone`` (shared encoder for both augmented views).
VGAE        Uses ``backbone`` (GNN output h; latent projections fc_mu/fc_logvar dropped).
"""

from __future__ import annotations

import copy
import sys
import tempfile
from pathlib import Path
from typing import Any

# Ensure the project root is importable regardless of working directory
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import hydra
import rootutils
import torch
import torch.nn as nn
import wandb
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from omegaconf import DictConfig
from torch.utils.data import Subset

from topobench.callbacks.best_epoch_metrics import BestEpochMetricsCallback
from topobench.nn.readouts.base import AbstractZeroCellReadOut
from topobench.utils.config_resolvers import get_monitor_metric, get_monitor_mode

# ── Overrides that are machine/infra-specific and must not be replayed ────────
_INFRA_PREFIXES = (
    "trainer.devices",
    "logger.",
    "+logger.",
    "~logger.",
)


# ──────────────────────────────────────────────────────────────────────────────
# W&B helpers
# ──────────────────────────────────────────────────────────────────────────────

def fetch_run(project: str, entity: str, run_id: str | None = None) -> Any:
    """Return a W&B run object.

    Parameters
    ----------
    project, entity : str
        W&B project and entity names.
    run_id : str | None
        Specific run ID.  If *None*, returns the oldest run in the project.
    """
    api = wandb.Api()
    path = f"{entity}/{project}"
    if run_id:
        run = api.run(f"{path}/{run_id}")
    else:
        runs = list(api.runs(path, order="+created_at", per_page=1))
        if not runs:
            raise ValueError(f"No runs found in W&B project: {path}")
        run = runs[0]
    print(f"  run  : {run.name}  (id={run.id}, state={run.state})")
    return run


def get_hydra_overrides(run: Any) -> list[str]:
    """Extract the original Hydra CLI overrides from W&B run metadata.

    W&B captures ``sys.argv[1:]`` at launch time and stores it as
    ``run.metadata["args"]``.  We filter out machine-specific entries and
    apply ``++`` to all dotted value overrides so they work regardless of
    whether the key is pre-declared in the config struct (fixes Hydra's
    ``ConfigAttributeError: Key '...' is not in struct``).
    """
    raw: list[str] = (run.metadata or {}).get("args", [])
    if not raw:
        raise RuntimeError(
            "run.metadata['args'] is empty.  "
            "The run may have been launched without W&B metadata capture."
        )
    result = []
    for arg in raw:
        if any(arg.startswith(p) for p in _INFRA_PREFIXES):
            continue
        # Config-group overrides (e.g. "pretraining=bgrl") have no dot in the key
        # part — leave them as-is.  Value overrides may target keys not pre-declared
        # in the YAML struct, so we use ++ (upsert) to be safe.
        if not arg.startswith(("+", "~")):
            key = arg.split("=")[0]
            if "." in key:
                arg = "++" + arg
        result.append(arg)
    return result


def get_checkpoint_path(run: Any) -> Path:
    """Resolve the best-model checkpoint from the run's local output directory.

    The resolved ``paths.output_dir`` is logged as part of the Hydra config via
    ``log_hyperparameters()`` and is therefore available in ``run.config``.
    """
    paths_cfg: dict = (run.config or {}).get("paths") or {}
    output_dir = paths_cfg.get("output_dir")
    if not output_dir:
        raise ValueError(
            "'paths.output_dir' not found in W&B run config.  "
            "Ensure log_hyperparameters() was called during training."
        )
    ckpt_dir = Path(output_dir) / "checkpoints"
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

    ckpts = sorted(ckpt_dir.glob("*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"No .ckpt files in {ckpt_dir}")

    # Prefer the metric-best checkpoint over 'last.ckpt'
    best = [c for c in ckpts if "last" not in c.name] or ckpts
    print(f"  ckpt : {best[0]}")
    return best[0]


# ──────────────────────────────────────────────────────────────────────────────
# Config reconstruction
# ──────────────────────────────────────────────────────────────────────────────

def compose_cfg(overrides: list[str], config_dir: str | None = None) -> DictConfig:
    """Compose the full Hydra DictConfig from a list of CLI overrides.

    Mirrors what ``@hydra.main`` does at training time, so all custom resolvers
    (``get_pretraining_transform``, etc.) fire identically.
    """
    rootutils.setup_root(_PROJECT_ROOT, indicator=".project-root", pythonpath=True)

    from topobench.utils.config_resolvers import register_all_resolvers
    register_all_resolvers()

    if config_dir is None:
        config_dir = str(_PROJECT_ROOT / "configs")

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=config_dir, version_base="1.3", job_name="eval"):
        cfg = compose(config_name="run", overrides=overrides)
    return cfg


# ──────────────────────────────────────────────────────────────────────────────
# Data & model loading (pretraining evaluation)
# ──────────────────────────────────────────────────────────────────────────────

def build_datamodule(cfg: DictConfig) -> Any:
    """Reconstruct the Lightning datamodule with identical splits as training."""
    from topobench.data.preprocessor import PreProcessor
    from topobench.dataloader import TBDataloader

    loader = hydra.utils.instantiate(cfg.dataset.loader)
    dataset, dataset_dir = loader.load()

    preprocessor = PreProcessor(dataset, dataset_dir, cfg.get("transforms"))
    train, val, test = preprocessor.load_dataset_splits(cfg.dataset.split_params)

    return TBDataloader(
        dataset_train=train,
        dataset_val=val,
        dataset_test=test,
        **cfg.dataset.get("dataloader_params", {}),
    )


def load_model(
    cfg: DictConfig,
    ckpt_path: Path,
    device: torch.device,
) -> LightningModule:
    """Instantiate the model from *cfg* and load checkpoint weights."""
    model: LightningModule = hydra.utils.instantiate(
        cfg.model,
        evaluator=cfg.evaluator,
        optimizer=cfg.optimizer,
        loss=cfg.loss,
        learning_setting=cfg.dataset.split_params.get("learning_setting", None),
    )

    # Some wrappers (e.g. GraphMAEv2) lazily register EMA submodules on the
    # first forward pass via _init_ema().  Materialise them before load_state_dict.
    if hasattr(model.backbone, "_init_ema"):
        model.backbone._init_ema()

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.to(device).eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model: {type(model).__name__}  ({n_params:,} params)")
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation (pretraining)
# ──────────────────────────────────────────────────────────────────────────────

def evaluate(
    model: LightningModule,
    datamodule: Any,
    device: torch.device,
    seed: int = 42,
) -> dict[str, float]:
    """Run ``trainer.test()`` and return a flat metrics dict.

    RNG is seeded to suppress stochasticity from augmentation-based methods
    (e.g. DGI's feature-shuffle corruption applied during test forward pass).
    """
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if device.type == "cuda":
        trainer = Trainer(
            accelerator="cuda",
            devices=[device.index if device.index is not None else 0],
            logger=False,
            enable_progress_bar=True,
            num_sanity_val_steps=0,
        )
    else:
        trainer = Trainer(
            accelerator="cpu",
            devices=1,
            logger=False,
            enable_progress_bar=True,
            num_sanity_val_steps=0,
        )

    results = trainer.test(model=model, datamodule=datamodule)
    return results[0] if results else {}


# ──────────────────────────────────────────────────────────────────────────────
# Downstream readout
# ──────────────────────────────────────────────────────────────────────────────

class DownstreamReadOut(AbstractZeroCellReadOut):
    """Minimal graph-level readout: ReLU on node features → pool → linear → logits.

    Architecture
    ------------
    node embeddings  →  ReLU  →  scatter-pool (sum/mean/max)  →  Linear  →  logits

    The pool + Linear steps are handled by the base-class ``compute_logits``.
    This subclass only adds per-node ReLU before pooling.

    Parameters
    ----------
    hidden_dim : int
        Dimension of node embeddings from the GNN.
    out_channels : int
        Number of output logits (classes or regression targets).
    task_level : str
        ``"graph"`` (pool nodes → one vector per graph) or ``"node"``.
    pooling_type : str
        ``"mean"`` | ``"sum"`` | ``"max"``
    """

    def __init__(
        self,
        hidden_dim: int,
        out_channels: int,
        task_level: str,
        pooling_type: str = "mean",
        **kwargs,
    ):
        super().__init__(
            hidden_dim=hidden_dim,
            out_channels=out_channels,
            task_level=task_level,
            pooling_type=pooling_type,
            logits_linear_layer=True,
        )
        self.act = nn.ReLU()

    def forward(self, model_out: dict, batch) -> dict:
        model_out["x_0"] = self.act(model_out["x_0"])
        return model_out


# ──────────────────────────────────────────────────────────────────────────────
# Downstream fine-tuning — model construction
# ──────────────────────────────────────────────────────────────────────────────

# Overrides to carry from the pretraining run into the downstream config.
# Trainer / callback / pretraining-wrapper overrides are intentionally dropped.
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


def extract_downstream_overrides(pretrain_overrides: list[str]) -> list[str]:
    """Convert pretraining CLI overrides to a clean downstream (supervised) config.

    Keeps dataset identity, model architecture, and optimizer settings.
    Adds ``pretraining=none`` so the loss/evaluator/transforms revert to the
    original supervised task.  Trainer and callback overrides are dropped.
    """
    result = ["pretraining=none"]
    for arg in pretrain_overrides:
        if any(arg.startswith(k) for k in _DOWNSTREAM_KEEP):
            result.append(arg)
    return result


def build_downstream_model(
    pretrained_model: LightningModule,
    downstream_cfg: DictConfig,
    pooling_type: str = "mean",
) -> LightningModule:
    """Swap the pretraining wrapper + readout for supervised GNNWrapper + DownstreamReadOut.

    Encoder selection per pretraining method
    -----------------------------------------
    GraphMAEv2  ``encoder_ema``    — EMA teacher saw clean (unmasked) inputs; better downstream
                                     representations than the masked-input student.
    BGRL        ``online_encoder`` — directly-trained branch; ``backbone`` is the frozen EMA
                                     target and was never given gradient updates.
    DGI         ``backbone``       — single encoder processes both positive and negative views;
                                     full representation quality is in this module.
    GraphCL     ``backbone``       — shared encoder used for both augmented views.
    VGAE        ``backbone``       — GNN output ``h``; ``fc_mu``/``fc_logvar`` are latent
                                     projection layers that are pretraining-specific and dropped.

    Steps
    -----
    1. Deep-copy the appropriate GNN encoder and the feature encoder from the pretrained model.
    2. Wrap the GNN in a plain ``GNNWrapper``; copy LayerNorm weights if the pretrained
       run used residual connections.
    3. Attach a fresh ``DownstreamReadOut`` (ReLU + pool + linear head).
    4. Instantiate loss + evaluator from the supervised downstream config.
    5. Assemble a new ``TBModel``.

    Parameters
    ----------
    pretrained_model : LightningModule
        Model loaded by ``load_model()`` carrying pretrained weights.
    downstream_cfg : DictConfig
        Config composed with ``pretraining=none`` for the target supervised task.
    pooling_type : str
        Graph-level pooling for ``DownstreamReadOut``: ``"mean"`` | ``"sum"`` | ``"max"``.
    """
    from topobench.model import TBModel
    from topobench.nn.wrappers.graph.gnn_wrapper import GNNWrapper

    pw = pretrained_model.backbone  # e.g. GraphMAEv2GNNWrapper / BGRLGNNWrapper / …

    # ── 1. Extract pretrained GNN encoder ─────────────────────────────────────
    if hasattr(pw, "encoder_ema") and pw.encoder_ema is not None:
        # GraphMAEv2: EMA teacher (trained on clean inputs → better for downstream)
        gnn_encoder = copy.deepcopy(pw.encoder_ema)
    elif hasattr(pw, "online_encoder"):
        # BGRL: use the directly-trained online encoder.
        # pw.backbone is the frozen EMA target which was never given gradient updates
        # directly — the online_encoder is the one that received all the gradient signal.
        gnn_encoder = copy.deepcopy(pw.online_encoder)
    else:
        # DGI, GraphCL, VGAE (and any other method): use the main backbone GNN.
        # For VGAE this excludes the latent-space fc_mu / fc_logvar projection layers,
        # which are pretraining-specific and must not be transferred downstream.
        gnn_encoder = copy.deepcopy(pw.backbone)

    # The EMA target has requires_grad=False on all params (it is never directly
    # optimised during pretraining).  Unfreeze everything so the full model is
    # trainable by default; apply_finetuning_mode() will selectively re-freeze
    # for probe variants afterwards.
    for param in gnn_encoder.parameters():
        param.requires_grad = True

    feature_encoder = copy.deepcopy(pretrained_model.feature_encoder)

    out_channels = downstream_cfg.model.feature_encoder.out_channels
    residual = pw.residual_connections
    num_cell_dims = len(list(pw.dimensions))  # range → int

    # ── 2. Clean GNNWrapper ───────────────────────────────────────────────────
    new_wrapper = GNNWrapper(
        backbone=gnn_encoder,
        out_channels=out_channels,
        num_cell_dimensions=num_cell_dims,
        residual_connections=residual,
    )
    # Carry over LayerNorm weights if residual connections were active in pretraining
    if residual:
        for i in range(num_cell_dims):
            src = getattr(pw, f"ln_{i}", None)
            dst = getattr(new_wrapper, f"ln_{i}", None)
            if src is not None and dst is not None:
                dst.load_state_dict(src.state_dict())

    # ── 3. Downstream readout (fresh linear head) ─────────────────────────────
    ds_params = downstream_cfg.dataset.parameters
    readout = DownstreamReadOut(
        hidden_dim=out_channels,
        out_channels=int(ds_params.num_classes),
        task_level=str(ds_params.task_level),
        pooling_type=pooling_type,
    )

    # ── 4. Loss, evaluator, optimizer from supervised downstream config ────────
    loss = hydra.utils.instantiate(downstream_cfg.loss)
    evaluator = hydra.utils.instantiate(downstream_cfg.evaluator)
    # During normal training Hydra auto-instantiates nested _target_ configs.
    # Here we call TBModel() directly, so we must instantiate the optimizer
    # explicitly — otherwise self.optimizer is a raw DictConfig and
    # configure_optimizers() fails trying to call .configure_optimizer() on it.
    optimizer = hydra.utils.instantiate(downstream_cfg.optimizer)

    # ── 5. Assemble TBModel ───────────────────────────────────────────────────
    # Pass the GNNWrapper *instance* as backbone with backbone_wrapper=None so
    # TBModel stores it directly (self.backbone = new_wrapper).
    # compile=False: TBModel.setup reads self.hparams.compile; skip torch.compile.
    return TBModel(
        backbone=new_wrapper,
        backbone_wrapper=None,
        feature_encoder=feature_encoder,
        readout=readout,
        loss=loss,
        evaluator=evaluator,
        optimizer=optimizer,
        learning_setting=downstream_cfg.dataset.split_params.get("learning_setting", None),
        compile=False,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Fine-tuning modes
# ──────────────────────────────────────────────────────────────────────────────

FINETUNE_MODES = [
    "finetune-full",      # pretrained weights, all params trainable
    "finetune-probe",     # pretrained weights, only readout trainable
    "random-init-full",   # randomized weights, all params trainable
    "random-init-probe",  # randomized weights, only readout trainable
]


def randomize_all_weights(model: nn.Module) -> None:
    """Reset *every* parameter in *model* to its default random initialisation.

    Walks all submodules depth-first.  Calls ``reset_parameters()`` where
    available (PyG/PyTorch convention); otherwise falls back to Xavier-uniform
    for weight matrices and zeros for bias vectors.
    """
    for module in model.modules():
        if hasattr(module, "reset_parameters"):
            module.reset_parameters()
        else:
            for name, param in module.named_parameters(recurse=False):
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param.data)
                else:
                    nn.init.zeros_(param.data)


def apply_finetuning_mode(model: LightningModule, mode: str) -> None:
    """Set which parameters are trainable according to *mode*.

    Modes
    -----
    ``finetune-full``
        Pretrained weights, everything trainable.
    ``finetune-probe``
        Pretrained weights; feature_encoder + GNN backbone frozen;
        only the readout head is trained.
    ``random-init-full``
        All weights randomised first, then all trainable.
    ``random-init-probe``
        All weights randomised; feature_encoder + backbone frozen;
        only readout trained.

    Note: ``TBModel.configure_optimizers`` passes all params to the optimiser.
    Frozen params (``requires_grad=False``) never accumulate gradients, so
    ``optimizer.step()`` skips them — no special optimiser handling needed.
    """
    if "random-init" in mode:
        randomize_all_weights(model)

    if "probe" in mode:
        for param in model.feature_encoder.parameters():
            param.requires_grad = False
        for param in model.backbone.parameters():
            param.requires_grad = False
        for param in model.readout.parameters():
            param.requires_grad = True
    # else: full fine-tuning — all params already have requires_grad=True


# ──────────────────────────────────────────────────────────────────────────────
# Few-shot datamodule + finetuning runner
# ──────────────────────────────────────────────────────────────────────────────

def get_downstream_monitor(downstream_cfg: DictConfig) -> tuple[str, str]:
    """Return ``(monitor_metric, monitor_mode)`` for the downstream supervised task.

    Uses the same resolvers as the main training pipeline
    (``get_monitor_metric`` / ``get_monitor_mode``) so regression tasks (MAE, MSE)
    select checkpoints with ``mode="min"`` and classification metrics use ``"max"``.
    """
    params = downstream_cfg.dataset.parameters
    task = str(params.task)
    metric = str(params.monitor_metric)
    return get_monitor_metric(task, metric), get_monitor_mode(task, metric)


def make_subset_datamodule(
    full_datamodule,
    train_fraction: float,
    batch_size: int | None = None,
    seed: int = 0,
) -> Any:
    """Return a datamodule whose training split is a seeded random subset.

    Val and test splits are kept intact for fair cross-experiment comparison.
    Uses ``torch.utils.data.Subset`` — no data is copied; the original
    ``collate_fn`` and dataset type are preserved.

    Parameters
    ----------
    batch_size : int | None
        Override the batch size.  ``None`` inherits from *full_datamodule*.
    """
    from topobench.dataloader import TBDataloader

    effective_bs = batch_size if batch_size is not None else full_datamodule.batch_size

    dataset_train = full_datamodule.dataset_train
    if train_fraction < 1.0:
        n_total = len(dataset_train)
        n_keep = max(1, int(n_total * train_fraction))
        g = torch.Generator()
        g.manual_seed(seed)
        indices = torch.randperm(n_total, generator=g)[:n_keep].tolist()
        dataset_train = Subset(dataset_train, indices)

    return TBDataloader(
        dataset_train=dataset_train,
        dataset_val=full_datamodule.dataset_val,
        dataset_test=full_datamodule.dataset_test,
        batch_size=effective_bs,
        num_workers=full_datamodule.num_workers,
        pin_memory=full_datamodule.pin_memory,
    )


def run_finetune_experiment(
    model: LightningModule,
    datamodule: Any,
    device: torch.device,
    monitor_metric: str = "val/loss",
    monitor_mode: str = "min",
    max_epochs: int = 100,
    patience: int = 20,
    seed: int = 42,
) -> tuple[dict[str, float], dict[str, float]]:
    """Train *model*, load the best checkpoint, run test.

    Returns
    -------
    test_metrics : dict
        Test-set metrics only (``test/mae``, …) — unchanged from the original API.
    fit_metrics : dict
        Train/val metrics at the best-validation epoch (for logging only; does not
        affect checkpoint selection or the test pass).
    """
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    accel = "cuda" if device.type == "cuda" else "cpu"
    devs = [device.index if device.index is not None else 0] if accel == "cuda" else 1

    # Observational only — same monitor/mode as ModelCheckpoint but does not
    # influence which checkpoint is saved or which weights are loaded for test.
    best_epoch_cb = BestEpochMetricsCallback(monitor=monitor_metric, mode=monitor_mode)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_cb = ModelCheckpoint(
            dirpath=tmpdir, monitor=monitor_metric, mode=monitor_mode, save_top_k=1,
        )
        fit_trainer = Trainer(
            accelerator=accel,
            devices=devs,
            max_epochs=max_epochs,
            logger=False,
            enable_progress_bar=True,
            num_sanity_val_steps=0,
            callbacks=[
                EarlyStopping(monitor=monitor_metric, patience=patience, mode=monitor_mode),
                ckpt_cb,
                best_epoch_cb,
            ],
        )
        model.to(device)
        fit_trainer.fit(model=model, datamodule=datamodule)

        if ckpt_cb.best_model_path:
            ckpt = torch.load(
                ckpt_cb.best_model_path, map_location=device, weights_only=False
            )
            model.load_state_dict(ckpt["state_dict"], strict=True)

    fit_metrics: dict[str, float] = {}
    if best_epoch_cb.best_epoch_number is not None:
        fit_metrics["best_epoch"] = float(best_epoch_cb.best_epoch_number)
    if best_epoch_cb.best_monitored_value is not None:
        fit_metrics[f"best_{monitor_metric.replace('/', '_')}"] = float(
            best_epoch_cb.best_monitored_value
        )
    for key, value in best_epoch_cb.best_epoch_metrics.items():
        fit_metrics[f"best_epoch/{key}"] = float(value)

    test_trainer = Trainer(
        accelerator=accel, devices=devs, logger=False, enable_progress_bar=False,
    )
    results = test_trainer.test(model=model, datamodule=datamodule, verbose=False)
    test_metrics = results[0] if results else {}
    return test_metrics, fit_metrics
