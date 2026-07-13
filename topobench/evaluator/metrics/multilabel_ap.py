"""Multilabel average precision metric aligned with OGB evaluation."""

from typing import Any

import torch
from torchmetrics import Metric
from torchmetrics.classification import BinaryAveragePrecision


class MultilabelMeanAveragePrecision(Metric):
    r"""Unweighted mean average precision for multilabel tasks.

    Matches the OGB ``ogbg-molpcba`` evaluator: per-label AP is computed on
    labeled entries only (NaN targets are ignored), labels with only positives
    or only negatives are skipped, and the remaining label APs are averaged
    with equal weight.
    """

    is_differentiable = False
    higher_is_better = True
    full_state_update = False

    pos_count: torch.Tensor
    neg_count: torch.Tensor

    def __init__(self, num_labels: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if not isinstance(num_labels, int) or num_labels <= 0:
            raise ValueError(
                f"Expected positive integer num_labels but got {num_labels}"
            )
        self.num_labels = num_labels
        self.label_metrics = torch.nn.ModuleList(
            [BinaryAveragePrecision() for _ in range(num_labels)]
        )
        self.add_state(
            "pos_count",
            default=torch.zeros(num_labels, dtype=torch.long),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "neg_count",
            default=torch.zeros(num_labels, dtype=torch.long),
            dist_reduce_fx="sum",
        )

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Update state with predictions and targets."""
        if target.dim() == 3 and target.size(1) == 1:
            target = target.squeeze(1)
        if preds.dim() == 3 and preds.size(1) == 1:
            preds = preds.squeeze(1)

        target = target.float()
        for label_idx in range(self.num_labels):
            mask = ~torch.isnan(target[:, label_idx])
            if not mask.any():
                continue

            label_target = target[mask, label_idx].long()
            label_preds = preds[mask, label_idx]
            self.pos_count[label_idx] += (label_target == 1).sum()
            self.neg_count[label_idx] += (label_target == 0).sum()
            self.label_metrics[label_idx].update(label_preds, label_target)

    def compute(self) -> torch.Tensor:
        """Compute the unweighted mean AP across valid labels."""
        scores = []
        for label_idx in range(self.num_labels):
            if self.pos_count[label_idx] == 0 or self.neg_count[label_idx] == 0:
                continue
            score = self.label_metrics[label_idx].compute()
            if not torch.isnan(score):
                scores.append(score)

        if not scores:
            return torch.tensor(float("nan"), device=self.device)
        return torch.stack(scores).mean()
