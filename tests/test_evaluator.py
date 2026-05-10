"""Evaluator (segmentation metrics) tests."""
from __future__ import annotations

import torch

from segocr.training.evaluator import Evaluator


def test_evaluator_perfect_prediction_iou_one() -> None:
    evaluator = Evaluator(num_classes=4, device="cpu")
    pred = torch.tensor([[[0, 1, 2, 3], [0, 1, 2, 3]]])
    target = pred.clone()
    evaluator.update(pred, target)
    metrics = evaluator.compute()
    assert metrics["miou"] > 0.99


def test_evaluator_completely_wrong_iou_zero() -> None:
    evaluator = Evaluator(num_classes=4, device="cpu")
    pred = torch.tensor([[[1, 1, 1, 1], [1, 1, 1, 1]]])
    target = torch.tensor([[[2, 2, 2, 2], [2, 2, 2, 2]]])
    evaluator.update(pred, target)
    metrics = evaluator.compute()
    assert metrics["miou"] < 0.05


def test_binary_miou_collapse_correct() -> None:
    """Pred all class 1 vs target all class 2 → both foreground; binary
    collapse should give high IoU since both 'are foreground'."""
    evaluator = Evaluator(num_classes=4, device="cpu")
    pred = torch.tensor([[[1, 1, 1, 1]]])
    target = torch.tensor([[[2, 2, 2, 2]]])
    evaluator.update(pred, target)
    metrics = evaluator.compute()
    assert metrics["binary_miou"] > 0.4


def test_per_class_iou_keys_emitted() -> None:
    evaluator = Evaluator(num_classes=3, device="cpu")
    pred = torch.zeros(1, 4, 4, dtype=torch.long)
    target = torch.zeros(1, 4, 4, dtype=torch.long)
    evaluator.update(pred, target)
    metrics = evaluator.compute()
    assert "iou_class_00" in metrics
    assert "iou_class_01" in metrics
    assert "iou_class_02" in metrics


def test_evaluator_reset_clears_state() -> None:
    evaluator = Evaluator(num_classes=4, device="cpu")
    pred = torch.tensor([[[1, 1], [1, 1]]])
    target = torch.tensor([[[1, 1], [1, 1]]])
    evaluator.update(pred, target)
    assert evaluator.confusion_matrix.sum() > 0
    evaluator.reset()
    assert evaluator.confusion_matrix.sum() == 0


def test_evaluator_ignores_oor_targets() -> None:
    """Targets outside [0, num_classes) should be ignored, not crash."""
    evaluator = Evaluator(num_classes=4, device="cpu")
    pred = torch.tensor([[[0, 1, 2, 3]]])
    target = torch.tensor([[[0, 1, 99, 3]]])  # 99 is out of range
    evaluator.update(pred, target)
    # Should compute over the valid pixels only
    metrics = evaluator.compute()
    assert metrics["miou"] > 0.5
