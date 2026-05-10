"""Loss function tests."""
from __future__ import annotations

import pytest
import torch

from segocr.models.losses import (
    DiceLoss,
    FocalLoss,
    SegOCRLoss,
    class_weights_from_distribution,
)


@pytest.fixture
def loss_config() -> dict:
    return {
        "focal_weight": 0.5,
        "focal_gamma": 2.0,
        "dice_weight": 0.5,
        "affinity_weight": 0.3,
        "direction_weight": 0.2,
        "background_class_weight": 0.2,
        "max_class_weight": 10.0,
        "affinity_pos_weight": 5.0,
    }


def test_focal_loss_perfect_prediction_is_near_zero() -> None:
    focal = FocalLoss(gamma=2.0)
    # 4-class problem, one-hot perfect predictions
    logits = torch.full((1, 4, 8, 8), -10.0)
    logits[:, 1, :, :] = 10.0  # confidently predict class 1
    target = torch.full((1, 8, 8), 1, dtype=torch.long)
    loss = focal(logits, target)
    assert loss.item() < 1e-3


def test_focal_loss_wrong_prediction_is_large() -> None:
    focal = FocalLoss(gamma=2.0)
    logits = torch.full((1, 4, 8, 8), -10.0)
    logits[:, 1, :, :] = 10.0
    target = torch.full((1, 8, 8), 3, dtype=torch.long)  # wrong class
    loss = focal(logits, target)
    assert loss.item() > 1.0


def test_dice_loss_perfect_prediction_is_near_zero() -> None:
    dice = DiceLoss(ignore_background=True)
    logits = torch.full((1, 4, 8, 8), -10.0)
    logits[:, 1, :, :] = 10.0
    target = torch.full((1, 8, 8), 1, dtype=torch.long)
    loss = dice(logits, target)
    assert loss.item() < 0.05


def test_segocr_loss_returns_scalar_and_components(loss_config: dict) -> None:
    loss_fn = SegOCRLoss(loss_config, num_classes=4)
    predictions = {
        "semantic": torch.randn(2, 4, 16, 16),
        "affinity": torch.randn(2, 1, 16, 16),
        "direction": torch.randn(2, 2, 16, 16),
    }
    targets = {
        "semantic": torch.randint(0, 4, (2, 16, 16)),
        "affinity": torch.randint(0, 3, (2, 16, 16)),
        "direction": torch.randn(2, 2, 16, 16),
    }
    total, terms = loss_fn(predictions, targets)
    assert total.dim() == 0
    assert "focal" in terms and "dice" in terms
    assert "affinity" in terms
    assert "direction" in terms
    assert "total" in terms
    assert torch.isfinite(total)


def test_segocr_loss_no_optional_heads(loss_config: dict) -> None:
    """When predictions lack affinity/direction, loss should still work."""
    loss_fn = SegOCRLoss(loss_config, num_classes=4)
    predictions = {"semantic": torch.randn(2, 4, 16, 16)}
    targets = {"semantic": torch.randint(0, 4, (2, 16, 16))}
    total, terms = loss_fn(predictions, targets)
    assert torch.isfinite(total)
    assert "affinity" not in terms
    assert "direction" not in terms


def test_direction_loss_masked_to_foreground(loss_config: dict) -> None:
    """Direction loss must ignore background pixels.

    Construct a scenario where direction predictions are wildly wrong on
    background but exact on foreground. The loss should be near zero.
    """
    loss_fn = SegOCRLoss(loss_config, num_classes=4)
    pred_dir = torch.zeros(1, 2, 8, 8)
    target_dir = torch.zeros(1, 2, 8, 8)
    # All pixels are background (semantic = 0) → direction loss ignored
    semantic_target = torch.zeros(1, 8, 8, dtype=torch.long)
    # Perturb direction predictions wildly
    pred_dir.fill_(100.0)

    predictions = {
        "semantic": torch.randn(1, 4, 8, 8),
        "direction": pred_dir,
    }
    targets = {"semantic": semantic_target, "direction": target_dir}
    _, terms = loss_fn(predictions, targets)
    # All-background target → direction loss should be 0 (no foreground pixels)
    assert terms["direction"].item() < 1e-3


def test_class_weights_from_distribution_caps_at_max() -> None:
    counts = torch.tensor([1_000_000.0, 1.0, 1.0, 1.0])  # background dominant
    weights = class_weights_from_distribution(counts, max_weight=10.0)
    assert weights[0] == pytest.approx(0.2, abs=1e-6)  # background overridden
    assert (weights[1:] <= 10.0).all()


def test_class_weights_background_explicit_value() -> None:
    counts = torch.tensor([100.0, 50.0, 50.0, 50.0])
    weights = class_weights_from_distribution(counts, background_weight=0.05)
    assert weights[0] == pytest.approx(0.05, abs=1e-6)
