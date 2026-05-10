"""Validation-time evaluator.

Computes segmentation metrics on the synthetic val split during training,
and on real benchmarks during the evaluation phase. Always emits
per-class IoU + the three mIoU variants (mean, foreground-only, binary
collapsed).
"""
from __future__ import annotations

import logging

import torch
from torch import nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

EPS = 1e-6


class Evaluator:
    """Streaming confusion-matrix-based segmentation evaluator."""

    def __init__(
        self,
        num_classes: int,
        device: torch.device | str = "cuda",
    ) -> None:
        self.num_classes = int(num_classes)
        self.device = torch.device(device)
        self.confusion_matrix = torch.zeros(
            (self.num_classes, self.num_classes),
            dtype=torch.int64,
            device=self.device,
        )

    def reset(self) -> None:
        self.confusion_matrix.zero_()

    @torch.no_grad()
    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        """Add a batch to the confusion matrix.

        Args:
            predictions: (B, H, W) class-id argmax — long tensor.
            targets:     (B, H, W) class-id ground truth — long tensor.
        """
        pred = predictions.flatten().to(self.device)
        tgt = targets.flatten().to(self.device)
        valid = (tgt >= 0) & (tgt < self.num_classes)
        pred = pred[valid]
        tgt = tgt[valid]
        idx = tgt * self.num_classes + pred
        binc = torch.bincount(idx, minlength=self.num_classes ** 2)
        self.confusion_matrix += binc.view(self.num_classes, self.num_classes)

    def compute(self) -> dict[str, float]:
        """Return metrics dict: miou, fg_miou, binary_miou, per-class IoUs."""
        cm = self.confusion_matrix.float()
        diag = cm.diag()
        row_sum = cm.sum(dim=1)
        col_sum = cm.sum(dim=0)
        union = row_sum + col_sum - diag
        per_class_iou = (diag + EPS) / (union + EPS)
        # Mask classes that never appeared in either pred or target — they
        # would otherwise drag mIoU down to noise.
        present = (row_sum + col_sum) > 0

        metrics: dict[str, float] = {}
        if present.any():
            metrics["miou"] = float(per_class_iou[present].mean().item())
        else:
            metrics["miou"] = 0.0

        if present[1:].any():  # foreground-only
            metrics["fg_miou"] = float(per_class_iou[1:][present[1:]].mean().item())
        else:
            metrics["fg_miou"] = 0.0

        # Binary collapse: {0} vs {1..N}
        bg_pred = cm[:, 0].sum()
        fg_pred = cm[:, 1:].sum()
        bg_target = cm[0, :].sum()
        fg_target = cm[1:, :].sum()
        bg_intersect = cm[0, 0]
        fg_intersect = cm[1:, 1:].sum()
        bg_union = bg_pred + bg_target - bg_intersect
        fg_union = fg_pred + fg_target - fg_intersect
        bg_iou = (bg_intersect + EPS) / (bg_union + EPS)
        fg_iou = (fg_intersect + EPS) / (fg_union + EPS)
        metrics["binary_miou"] = float(((bg_iou + fg_iou) / 2.0).item())

        # Per-class IoU
        for c in range(self.num_classes):
            metrics[f"iou_class_{c:02d}"] = float(per_class_iou[c].item())

        return metrics

    @torch.no_grad()
    def evaluate(
        self,
        model: nn.Module,
        loader: DataLoader,
    ) -> dict[str, float]:
        model.eval()
        self.reset()
        for batch in loader:
            images = batch["image"].to(self.device, non_blocking=True)
            targets = batch["targets"]["semantic"].to(self.device, non_blocking=True)
            outputs = model(images)
            preds = outputs["semantic"].argmax(dim=1)
            self.update(preds, targets)
        model.train()
        return self.compute()
