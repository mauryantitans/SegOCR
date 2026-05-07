"""Combined loss for the multi-head segmentation model.

Implementation Guide §3.9. Total = focal_w·FocalCE + dice_w·Dice
                              + affinity_w·BCE + direction_w·SmoothL1.

Per-class inverse-frequency weights for the focal CE, with background
de-emphasized to ~0.1–0.3 and rare-class weight capped at 10.0
(Research Proposal §5.3).

Direction loss is masked to foreground pixels only — penalising direction
predictions on background would be meaningless.
"""
from __future__ import annotations

import torch
from torch import nn


class FocalLoss(nn.Module):
    """Multi-class focal cross-entropy with optional per-class alpha weights."""

    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor | None = None) -> None:
        super().__init__()
        self.gamma = gamma
        self.register_buffer("alpha", alpha if alpha is not None else torch.tensor([]))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("FocalLoss.forward — Week 5")


class DiceLoss(nn.Module):
    """Per-class soft Dice loss, averaged over classes."""

    def __init__(self, smooth: float = 1.0, ignore_index: int | None = None) -> None:
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("DiceLoss.forward — Week 5")


class SegOCRLoss(nn.Module):
    """Combined loss across the semantic, affinity, and direction heads."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.config = config
        # self.focal = FocalLoss(...)
        # self.dice = DiceLoss()
        # self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.0]))
        # self.smooth_l1 = nn.SmoothL1Loss(reduction="none")
        raise NotImplementedError("SegOCRLoss.__init__ — Week 5")

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Returns (total_loss, per_term_loss_dict) for logging."""
        raise NotImplementedError("SegOCRLoss.forward — Week 5")

    def _compute_class_weights(self, config: dict) -> torch.Tensor:
        """Inverse-frequency weights, background scaled to bg_weight,
        per-class weight clipped at max_class_weight."""
        raise NotImplementedError
