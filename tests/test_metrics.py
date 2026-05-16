"""Tests for segocr.evaluation.metrics."""
from __future__ import annotations

import numpy as np

from segocr.evaluation.metrics import (
    binary_miou,
    cer,
    char_accuracy,
    exact_match,
    fg_miou,
    miou,
    ned,
    word_accuracy,
)

# ── Segmentation ────────────────────────────────────────────────────────────


def test_miou_perfect_predictions_is_one():
    # Diagonal CM (everything correct) → IoU=1 for every present class
    cm = np.diag([100, 50, 30]).astype(np.int64)
    assert abs(miou(cm) - 1.0) < 1e-6


def test_miou_ignores_classes_that_never_appear():
    # Class 2 never appears in pred or target — should not drag mean
    cm = np.zeros((3, 3), dtype=np.int64)
    cm[0, 0] = 100
    cm[1, 1] = 50
    assert abs(miou(cm) - 1.0) < 1e-6


def test_fg_miou_excludes_background():
    cm = np.zeros((3, 3), dtype=np.int64)
    cm[0, 0] = 100   # perfect bg
    cm[1, 1] = 50    # perfect fg class 1
    cm[2, 1] = 10    # class 2: 10 pixels labeled as class 1 (wrong)
    # fg_miou averages class 1 and class 2 IoUs only
    val = fg_miou(cm)
    # class 1 IoU: 50 / (50 + 10) = 0.833...
    # class 2 IoU: 0 / 10 = 0
    expected = (50 / 60 + 0) / 2
    assert abs(val - expected) < 1e-4


def test_binary_miou_collapses_foreground():
    cm = np.zeros((3, 3), dtype=np.int64)
    cm[0, 0] = 100  # bg perfect
    cm[1, 2] = 20   # class 1 → predicted as class 2 (still foreground)
    # binary IoU treats this as a foreground match, so should be 1.0
    val = binary_miou(cm)
    assert abs(val - 1.0) < 1e-4


# ── String metrics ──────────────────────────────────────────────────────────


def test_cer_identical_strings_is_zero():
    assert cer("hello", "hello") == 0.0


def test_cer_one_substitution():
    # 1 substitution / 5 chars = 0.2
    assert abs(cer("hallo", "hello") - 0.2) < 1e-9


def test_cer_empty_target_full_pred_is_one():
    assert cer("abc", "") == 1.0


def test_cer_both_empty_is_zero():
    assert cer("", "") == 0.0


def test_char_accuracy_matches_inverse_cer():
    assert abs(char_accuracy("hallo", "hello") - 0.8) < 1e-9


def test_ned_clipped_to_zero():
    # Long noise vs short target: CER > 1, ned should clip to 0
    assert ned("aaaaaaaaaaaaa", "x") == 0.0


def test_word_accuracy_aligned_positions():
    pred = ["the", "quick", "brown", "fox"]
    tgt = ["the", "quack", "brown", "fox"]
    # 3/4 correct
    assert abs(word_accuracy(pred, tgt) - 0.75) < 1e-9


def test_word_accuracy_length_mismatch_counted_as_error():
    pred = ["the", "fox"]
    tgt = ["the", "quick", "fox"]
    # 1 correct out of max(2, 3) = 3 → 1/3
    assert abs(word_accuracy(pred, tgt) - (1 / 3)) < 1e-9


def test_exact_match():
    assert exact_match("abc", "abc") == 1.0
    assert exact_match("abc", "abd") == 0.0
