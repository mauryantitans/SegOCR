"""Validation-time evaluator.

Computes segmentation metrics on the synthetic val split during training,
and on real benchmarks during the evaluation phase. Always emits per-class
IoU + the three mIoU variants (mean, foreground-only, binary collapsed).
"""
from __future__ import annotations

import torch
from torch import nn


class Evaluator:
    """Streaming confusion-matrix-based segmentation evaluator."""

    def __init__(self, num_classes: int, device: torch.device | str = "cuda") -> None:
        self.num_classes = num_classes
        self.device = device
        self.confusion_matrix = torch.zeros(
            (num_classes, num_classes), dtype=torch.int64, device=device
        )
        raise NotImplementedError("Evaluator.__init__ — Week 5")

    def reset(self) -> None:
        self.confusion_matrix.zero_()

    @torch.no_grad()
    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        """Add a batch of (pred_argmax, target) to the confusion matrix."""
        raise NotImplementedError("Evaluator.update — Week 5")

    def compute(self) -> dict[str, float]:
        """Return {miou, fg_miou, binary_miou, iou_<class>...}."""
        raise NotImplementedError("Evaluator.compute — Week 5")

    @torch.no_grad()
    def evaluate(self, model: nn.Module, loader) -> dict[str, float]:
        raise NotImplementedError("Evaluator.evaluate — Week 5")
