"""Combined loss for the multi-head segmentation model.

Implementation Guide §3.9 + Research Proposal §5.3.

Total loss for a sample:
    L = 0.5 * Focal_CE + 0.5 * Dice                      # semantic head
      + 0.3 * Affinity_BCE        if affinity head present
      + 0.2 * Direction_SmoothL1  if direction head present (foreground only)

The Focal CE uses per-class inverse-frequency weights with the
background class de-emphasized to ~0.2 and rare classes capped at 10×.

The direction loss is masked to foreground pixels only — penalising
direction predictions on background would be meaningless and would
dominate the loss given background is ~95% of pixels.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812 — F is the canonical torch.nn.functional alias
from torch import nn

EPS = 1e-6


class FocalLoss(nn.Module):
    """Multi-class focal cross-entropy.

    L = -alpha_y * (1 - p_y)^gamma * log(p_y)

    where p_y is the softmax probability of the target class.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: torch.Tensor | None = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = float(gamma)
        if alpha is not None:
            self.register_buffer("alpha", alpha.float())
        else:
            self.alpha = None  # type: ignore[assignment]
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: (B, C, H, W); targets: (B, H, W) long
        log_probs = F.log_softmax(logits, dim=1)
        # Per-pixel cross-entropy: gather the log-prob of the true class
        target_log_p = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        # (1 - p_y)^gamma
        focal = (1.0 - target_log_p.exp()) ** self.gamma
        loss = -focal * target_log_p

        if self.alpha is not None:
            alpha_per_pixel = self.alpha[targets]
            loss = loss * alpha_per_pixel

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class DiceLoss(nn.Module):
    """Per-class soft Dice loss, averaged over classes.

    For each class c:
        Dice_c = 2 * sum(p_c * y_c) / (sum(p_c) + sum(y_c))
        L_c = 1 - Dice_c
    Returned loss is the mean across classes (excluding background by default).
    """

    def __init__(
        self,
        smooth: float = 1.0,
        ignore_background: bool = True,
    ) -> None:
        super().__init__()
        self.smooth = float(smooth)
        self.ignore_background = ignore_background

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: (B, C, H, W); targets: (B, H, W) long
        # Per-class loop avoids materializing a full (B, C, H, W) one-hot tensor;
        # at C=63 + 512² that one-hot is ~63× the peak per-class tensor.
        probs = F.softmax(logits, dim=1)
        num_classes = probs.shape[1]
        start = 1 if self.ignore_background else 0

        dice_sum = logits.new_zeros(())
        count = 0
        for c in range(start, num_classes):
            prob_c = probs[:, c]
            target_c = (targets == c).float()
            intersection = (prob_c * target_c).sum()
            cardinality = prob_c.sum() + target_c.sum()
            dice_sum = dice_sum + (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
            count += 1
        return 1.0 - dice_sum / max(count, 1)


class SegOCRLoss(nn.Module):
    """Combined loss across the semantic, affinity, and direction heads.

    Construction:
        loss_fn = SegOCRLoss(config["model"]["loss"], num_classes=63)
        # Optionally pre-compute class weights from a dataset and set:
        loss_fn.set_class_weights(weights_tensor)
    """

    def __init__(self, loss_config: dict, num_classes: int) -> None:
        super().__init__()
        self.config = loss_config
        self.num_classes = int(num_classes)

        # Default uniform class weights; users can override via set_class_weights.
        default_alpha = torch.ones(self.num_classes, dtype=torch.float32)
        default_alpha[0] = float(loss_config.get("background_class_weight", 0.2))
        self.focal = FocalLoss(
            gamma=float(loss_config.get("focal_gamma", 2.0)),
            alpha=default_alpha,
        )
        self.dice = DiceLoss(ignore_background=True)
        self.bce = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([float(loss_config.get("affinity_pos_weight", 5.0))])
        )
        self.smooth_l1 = nn.SmoothL1Loss(reduction="none")

        self.focal_weight = float(loss_config.get("focal_weight", 0.5))
        self.dice_weight = float(loss_config.get("dice_weight", 0.5))
        self.affinity_weight = float(loss_config.get("affinity_weight", 0.3))
        self.direction_weight = float(loss_config.get("direction_weight", 0.2))

    def set_class_weights(self, weights: torch.Tensor) -> None:
        """Replace the focal-loss alpha vector with class-frequency weights."""
        if weights.shape != (self.num_classes,):
            raise ValueError(
                f"weights must be shape ({self.num_classes},); got {weights.shape}"
            )
        self.focal.alpha = weights.float().to(self.focal.alpha.device)  # type: ignore[union-attr]

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Returns (total_loss, per_term_loss_dict) for logging."""
        loss_terms: dict[str, torch.Tensor] = {}

        sem_logits = predictions["semantic"]
        sem_target = targets["semantic"]
        focal = self.focal(sem_logits, sem_target)
        dice = self.dice(sem_logits, sem_target)
        loss_terms["focal"] = focal
        loss_terms["dice"] = dice
        total = self.focal_weight * focal + self.dice_weight * dice

        if "affinity" in predictions and "affinity" in targets:
            aff_logits = predictions["affinity"].squeeze(1)  # (B, H, W)
            aff_target = (targets["affinity"] > 0).float()
            aff = self.bce(aff_logits, aff_target)
            loss_terms["affinity"] = aff
            total = total + self.affinity_weight * aff

        if "direction" in predictions and "direction" in targets:
            dir_pred = predictions["direction"]                # (B, 2, H, W)
            dir_target = targets["direction"]                  # (B, 2, H, W)
            fg = (sem_target > 0).unsqueeze(1).float()         # (B, 1, H, W)
            per_pixel = self.smooth_l1(dir_pred, dir_target).mean(dim=1, keepdim=True)
            denom = fg.sum().clamp(min=1.0)
            dir_loss = (per_pixel * fg).sum() / denom
            loss_terms["direction"] = dir_loss
            total = total + self.direction_weight * dir_loss

        loss_terms["total"] = total
        return total, loss_terms


# ── Helpers ─────────────────────────────────────────────────────────────────


def class_weights_from_distribution(
    class_pixel_counts: torch.Tensor,
    background_weight: float = 0.2,
    max_weight: float = 10.0,
) -> torch.Tensor:
    """Compute per-class focal-loss alpha from a per-class pixel-count vector.

    Uses inverse-frequency weighting normalized so the median class has
    weight 1.0; background is overridden to ``background_weight``;
    weights are clipped at ``max_weight``.
    """
    counts = class_pixel_counts.float().clamp(min=1.0)
    inv = 1.0 / counts
    # Median of foreground classes (skip background = index 0)
    median = inv[1:].median() if counts.numel() > 1 else inv.median()
    weights = (inv / median.clamp(min=EPS)).clamp(max=max_weight)
    weights[0] = background_weight
    return weights
