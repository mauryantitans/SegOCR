"""Saliency / placement-scorer tests."""
from __future__ import annotations

import numpy as np

from segocr.generator.saliency import compute_placement_score, find_best_position


def test_score_returns_correct_shape() -> None:
    bg = (np.random.rand(64, 96, 3) * 255).astype(np.uint8)
    score = compute_placement_score(bg)
    assert score.shape == (64, 96)
    assert score.dtype == np.float32


def test_score_is_in_valid_range() -> None:
    bg = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
    score = compute_placement_score(bg)
    assert score.min() >= 0.0
    assert score.max() <= 1.0


def test_flat_region_scores_higher_than_busy_region() -> None:
    """A solid-color background should score higher in the center than
    a heavily-textured background does anywhere."""
    flat = np.full((128, 128, 3), 128, dtype=np.uint8)
    flat_score = compute_placement_score(flat)

    # Heavily textured: random per-pixel noise.
    busy = (np.random.rand(128, 128, 3) * 255).astype(np.uint8)
    busy_score = compute_placement_score(busy)

    # Center is far from edges → minimal border penalty.
    flat_center = flat_score[64, 64]
    busy_max = busy_score.max()
    assert flat_center > busy_max, (
        f"Flat region center ({flat_center:.3f}) should outscore "
        f"the best of busy ({busy_max:.3f})"
    )


def test_border_pixels_penalized() -> None:
    flat = np.full((128, 128, 3), 128, dtype=np.uint8)
    score = compute_placement_score(flat)
    center = float(score[64, 64])
    corner = float(score[2, 2])
    assert corner < center, "Border pixels should be penalized vs center"


def test_find_best_position_returns_valid_offset() -> None:
    score = np.zeros((100, 100), dtype=np.float32)
    score[50, 50] = 1.0
    y, x = find_best_position(score, region_shape=(20, 20), randomness=0.0)
    assert 0 <= y <= 80
    assert 0 <= x <= 80


def test_find_best_position_zero_randomness_is_argmax() -> None:
    """With randomness=0, the picked offset's window should contain the
    score peak."""
    score = np.zeros((100, 100), dtype=np.float32)
    score[40:60, 40:60] = 1.0  # 20×20 hot region centered at (50, 50)
    y, x = find_best_position(score, region_shape=(20, 20), randomness=0.0)
    # Optimal top-left is (40, 40) for a window that covers the hot region
    assert 30 <= y <= 50
    assert 30 <= x <= 50


def test_find_best_position_handles_oversized_region() -> None:
    """If region >= score map, return (0, 0) gracefully."""
    score = np.ones((50, 50), dtype=np.float32)
    y, x = find_best_position(score, region_shape=(100, 100))
    assert (y, x) == (0, 0)


def test_find_best_position_top_k_yields_high_score_picks() -> None:
    """With randomness > 0, picks should still be in the top quantile of
    the score map, not arbitrary."""
    score = np.zeros((100, 100), dtype=np.float32)
    score[40:60, 40:60] = 1.0
    np.random.seed(0)
    picks = [find_best_position(score, region_shape=(20, 20), randomness=0.3)
             for _ in range(20)]
    # Every pick's top-left should land within the broad neighborhood
    # of the hot region (within a 30-pixel slack)
    for y, x in picks:
        assert 10 <= y <= 70, f"Pick y={y} too far from hot region"
        assert 10 <= x <= 70, f"Pick x={x} too far from hot region"
