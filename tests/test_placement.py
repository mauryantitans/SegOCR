"""PlacementMaskTracker tests."""
from __future__ import annotations

import numpy as np

from segocr.generator.placement import PlacementMaskTracker


def test_first_placement_is_collision_free() -> None:
    tracker = PlacementMaskTracker(image_size=(64, 64), min_separation_px=0)
    candidate = np.ones((10, 10), dtype=np.uint8)
    assert tracker.attempt_placement(candidate, (10, 10))


def test_overlap_after_commit_detected() -> None:
    tracker = PlacementMaskTracker(image_size=(64, 64), min_separation_px=0)
    candidate = np.ones((10, 10), dtype=np.uint8)
    tracker.commit(candidate, (10, 10))
    # Overlapping placement
    assert not tracker.attempt_placement(candidate, (15, 15))


def test_non_overlap_succeeds_after_commit() -> None:
    tracker = PlacementMaskTracker(image_size=(64, 64), min_separation_px=0)
    candidate = np.ones((10, 10), dtype=np.uint8)
    tracker.commit(candidate, (0, 0))
    assert tracker.attempt_placement(candidate, (40, 40))


def test_min_separation_pushes_collision_boundary_outward() -> None:
    """With min_separation_px=3, two adjacent placements should collide
    even when their masks don't share a pixel."""
    tracker_loose = PlacementMaskTracker(image_size=(64, 64), min_separation_px=0)
    tracker_strict = PlacementMaskTracker(image_size=(64, 64), min_separation_px=3)

    candidate = np.ones((10, 10), dtype=np.uint8)
    for tracker in (tracker_loose, tracker_strict):
        tracker.commit(candidate, (0, 0))

    # Place a second candidate exactly adjacent (no shared pixel)
    assert tracker_loose.attempt_placement(candidate, (0, 10))     # touches edge → no overlap
    assert not tracker_strict.attempt_placement(candidate, (0, 10))  # within separation buffer


def test_reset_clears_mask() -> None:
    tracker = PlacementMaskTracker(image_size=(64, 64))
    candidate = np.ones((10, 10), dtype=np.uint8)
    tracker.commit(candidate, (0, 0))
    assert tracker.collision_mask.sum() > 0
    tracker.reset()
    assert tracker.collision_mask.sum() == 0


def test_extract_polygon_returns_only_boundary_pixels() -> None:
    """A 5×5 solid square has 16 boundary pixels and 9 interior pixels."""
    tracker = PlacementMaskTracker(image_size=(20, 20))
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[5:10, 5:10] = 1  # 5×5 solid square
    polygon = tracker.extract_polygon(mask)
    # Boundary of a 5×5 square = 5*5 - 3*3 = 16 pixels
    assert len(polygon) == 16


def test_offcanvas_placement_returns_false() -> None:
    tracker = PlacementMaskTracker(image_size=(64, 64))
    candidate = np.ones((10, 10), dtype=np.uint8)
    # Entirely outside the canvas
    assert not tracker.attempt_placement(candidate, (-100, -100))


def test_partial_offcanvas_treats_visible_part_only() -> None:
    """A candidate clipped at the canvas edge should still be testable."""
    tracker = PlacementMaskTracker(image_size=(64, 64), min_separation_px=0)
    candidate = np.ones((20, 20), dtype=np.uint8)
    # Place mostly outside, but with a visible corner.
    assert tracker.attempt_placement(candidate, (-15, -15))
