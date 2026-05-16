"""Segmentation + recognition metrics.

Research Proposal §8.1–§8.2. Three mIoU variants are tracked separately
because they answer different questions:
    miou          — overall multi-class — paper headline
    fg_miou       — excludes background — true OCR signal
    binary_miou   — collapses 1..N → 1 — noise-removal byproduct (also a
                    sanity floor: should always exceed any per-class IoU)

The recognition metrics (char_accuracy, word_accuracy, cer, ned) operate
on decoded strings produced by ``segocr.postprocessing.reading_order``.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-9


def _per_class_iou(confusion_matrix: np.ndarray) -> np.ndarray:
    """Per-class IoU vector from a (C, C) confusion matrix (rows=target, cols=pred)."""
    cm = confusion_matrix.astype(np.float64)
    diag = np.diag(cm)
    row_sum = cm.sum(axis=1)
    col_sum = cm.sum(axis=0)
    union = row_sum + col_sum - diag
    return (diag + EPS) / (union + EPS)


def miou(confusion_matrix: np.ndarray) -> float:
    """Mean IoU across all classes that appeared in either pred or target.

    Classes that never appeared anywhere are excluded — otherwise they
    would drag the mean to EPS-noise.
    """
    iou = _per_class_iou(confusion_matrix)
    cm = confusion_matrix
    present = (cm.sum(axis=1) + cm.sum(axis=0)) > 0
    if not present.any():
        return 0.0
    return float(iou[present].mean())


def fg_miou(confusion_matrix: np.ndarray) -> float:
    """Mean IoU across foreground (non-background) classes only."""
    iou = _per_class_iou(confusion_matrix)
    cm = confusion_matrix
    present = (cm.sum(axis=1) + cm.sum(axis=0)) > 0
    fg_present = present[1:]
    if not fg_present.any():
        return 0.0
    return float(iou[1:][fg_present].mean())


def binary_miou(confusion_matrix: np.ndarray) -> float:
    """IoU after collapsing classes 1..N → 1.

    Reports the noise-removal use case as a byproduct of the multi-class
    model. Always ≥ any per-class IoU.
    """
    cm = confusion_matrix.astype(np.float64)
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
    return float((bg_iou + fg_iou) / 2.0)


# ── String metrics ──────────────────────────────────────────────────────────


def _levenshtein(a: str, b: str) -> int:
    """Classic edit distance. O(len(a) * len(b)) time, O(min) memory."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    # Ensure b is the shorter for the row buffer
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[-1]


def char_accuracy(predicted: str, target: str) -> float:
    """Fraction of correctly-recognized characters: 1 - CER, clipped to [0, 1]."""
    return max(0.0, 1.0 - cer(predicted, target))


def cer(predicted: str, target: str) -> float:
    """Character Error Rate: Levenshtein(pred, target) / len(target).

    Returns 0.0 when both strings are empty, and 1.0 if target is empty
    but prediction is not (every predicted char is an insertion).
    """
    if not target:
        return 0.0 if not predicted else 1.0
    return _levenshtein(predicted, target) / len(target)


def ned(predicted: str, target: str) -> float:
    """Normalized Edit Distance (a.k.a. 1 - CER), clipped to [0, 1]."""
    return max(0.0, 1.0 - cer(predicted, target))


def word_accuracy(predicted_words: list[str], target_words: list[str]) -> float:
    """Fraction of fully-correct word matches by position.

    Compares aligned positions; words beyond the shorter list count as
    mismatches. Use when you want a positional measure rather than a
    bag-of-words measure.
    """
    if not target_words and not predicted_words:
        return 1.0
    if not target_words:
        return 0.0
    n = max(len(predicted_words), len(target_words))
    correct = sum(
        1
        for i in range(min(len(predicted_words), len(target_words)))
        if predicted_words[i] == target_words[i]
    )
    return correct / n


def exact_match(predicted: str, target: str) -> float:
    """1.0 iff predicted == target, else 0.0. Useful for aggregating."""
    return 1.0 if predicted == target else 0.0
