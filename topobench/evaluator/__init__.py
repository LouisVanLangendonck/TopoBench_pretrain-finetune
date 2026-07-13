"""Evaluators for model evaluation."""

from torchmetrics.classification import (
    AUROC,
    Accuracy,
    ConfusionMatrix,
    F1Score,
    Precision,
    Recall,
)
from torchmetrics.regression import (
    MeanAbsoluteError,
    MeanSquaredError,
    R2Score,
)

from .metrics import ExampleRegressionMetric
from .metrics.multilabel_ap import MultilabelMeanAveragePrecision

# Define metrics
METRICS = {
    "accuracy": Accuracy,
    "precision": Precision,
    "recall": Recall,
    "auroc": AUROC,
    "f1": F1Score,
    "f1_macro": F1Score,
    "f1_weighted": F1Score,
    "confusion_matrix": ConfusionMatrix,
    "mae": MeanAbsoluteError,
    "mse": MeanSquaredError,
    "rmse": MeanSquaredError,  # We'll configure this with squared=False
    "r2": R2Score,
    "example": ExampleRegressionMetric,
    "ap": MultilabelMeanAveragePrecision,
}

from .base import AbstractEvaluator  # noqa: E402
from .bgrl_evaluator import BGRLEvaluator
from .dgi_evaluator import DGIEvaluator
from .dgmae_evaluator import DGMAEEvaluator
from .evaluator import TBEvaluator  # noqa: E402
from .grace_evaluator import GRACEEvaluator
from .graphcl_evaluator import GraphCLEvaluator
from .graphmaev2_evaluator import GraphMAEv2Evaluator
from .vgae_evaluator import VGAEEvaluator

__all__ = [
    "METRICS",
    "AbstractEvaluator",
    "TBEvaluator",
    "GraphMAEv2Evaluator",
    "DGMAEEvaluator",
    "GRACEEvaluator",
    "VGAEEvaluator",
    "DGIEvaluator",
    "GraphCLEvaluator",
    "BGRLEvaluator",
]
