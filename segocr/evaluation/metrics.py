"""Segmentation + recognition metrics.

Research Proposal §8.1–§8.2. Three mIoU variants are tracked separately
because they answer different questions:
    miou          — overall multi-class — paper headline
    fg_miou       — excludes background — true OCR signal
    binary_miou   — collapses 1..N → 1 — noise-removal byproduct (also a
                    sanity floor: should always exceed any per-class IoU)
"""
from __future__ import annotations

import numpy as np


def miou(confusion_matrix: np.ndarray) -> float:
    """Mean IoU across all classes including background."""
    raise NotImplementedError("miou — Week 6")


def fg_miou(confusion_matrix: np.ndarray) -> float:
    """Mean IoU across foreground (non-background) classes only."""
    raise NotImplementedError("fg_miou — Week 6")


def binary_miou(confusion_matrix: np.ndarray) -> float:
    """IoU after collapsing classes 1..N → 1. Reports the noise-removal
    use case as a free byproduct of the multi-class model."""
    raise NotImplementedError("binary_miou — Week 6")


def char_accuracy(predicted: str, target: str) -> float:
    """Fraction of correctly-recognized characters in the predicted string."""
    raise NotImplementedError("char_accuracy — Week 6")


def word_accuracy(predicted_words: list[str], target_words: list[str]) -> float:
    """Fraction of fully-correct words."""
    raise NotImplementedError("word_accuracy — Week 6")


def cer(predicted: str, target: str) -> float:
    """Character Error Rate: Levenshtein(pred, target) / len(target)."""
    raise NotImplementedError("cer — Week 6")


def ned(predicted: str, target: str) -> float:
    """Normalized Edit Distance: 1 - cer, clipped to [0, 1]."""
    raise NotImplementedError("ned — Week 6")
