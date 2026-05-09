"""Target-builder tests — verifies derived training targets."""
from __future__ import annotations

import numpy as np

from segocr.generator.targets import (
    build_affinity_mask,
    build_direction_field,
    build_instance_mask,
)


def _make_metadata(chars_with_bboxes: list[tuple[str, int, tuple[int, int, int, int]]]):
    """Helper: build (char, class_id, bbox) tuples → metadata dicts."""
    out = []
    for char, class_id, (x0, y0, x1, y1) in chars_with_bboxes:
        out.append(
            {
                "char": char,
                "class_id": class_id,
                "bbox": (x0, y0, x1, y1),
                "centroid": ((x0 + x1) / 2, (y0 + y1) / 2),
                "area": (x1 - x0) * (y1 - y0),
            }
        )
    return out


def test_instance_mask_separates_same_class_chars() -> None:
    """Two 'A's at different positions should get distinct instance IDs."""
    semantic = np.zeros((20, 40), dtype=np.uint8)
    semantic[5:15, 2:12] = 1   # A at left
    semantic[5:15, 22:32] = 1  # A at right
    metadata = _make_metadata(
        [("A", 1, (2, 5, 12, 15)), ("A", 1, (22, 5, 32, 15))]
    )
    instance = build_instance_mask(semantic, metadata)
    assert instance[10, 7] == 1   # first A
    assert instance[10, 27] == 2  # second A
    assert instance[0, 0] == 0    # background


def test_instance_mask_background_is_zero() -> None:
    semantic = np.zeros((20, 20), dtype=np.uint8)
    metadata = _make_metadata([("A", 1, (5, 5, 10, 10))])
    instance = build_instance_mask(semantic, metadata)
    assert (instance == 0).all()  # nothing in semantic mask → no instances


def test_affinity_mask_groups_close_chars_into_word() -> None:
    """Three closely-spaced chars with no gap should share a word ID."""
    semantic = np.zeros((20, 60), dtype=np.uint8)
    for i in range(3):
        x = i * 10 + 2
        semantic[5:15, x : x + 8] = i + 1
    metadata = _make_metadata(
        [
            ("A", 1, (2, 5, 10, 15)),
            ("B", 2, (12, 5, 20, 15)),
            ("C", 3, (22, 5, 30, 15)),
        ]
    )
    affinity = build_affinity_mask(semantic, metadata)
    # All foreground pixels in the three chars should share a single word ID
    word_ids_used = set(np.unique(affinity).tolist()) - {0}
    assert len(word_ids_used) == 1, f"Expected 1 word, got {word_ids_used}"


def test_affinity_mask_separates_words_by_gap() -> None:
    """A wide gap between two chars should put them in different words."""
    semantic = np.zeros((20, 80), dtype=np.uint8)
    semantic[5:15, 2:10] = 1
    semantic[5:15, 60:68] = 1
    metadata = _make_metadata(
        [("A", 1, (2, 5, 10, 15)), ("B", 1, (60, 5, 68, 15))]
    )
    affinity = build_affinity_mask(semantic, metadata)
    word_ids_used = set(np.unique(affinity).tolist()) - {0}
    assert len(word_ids_used) == 2, f"Expected 2 words, got {word_ids_used}"


def test_direction_field_zero_at_centroid() -> None:
    semantic = np.zeros((20, 20), dtype=np.uint8)
    semantic[5:15, 5:15] = 1
    metadata = _make_metadata([("A", 1, (5, 5, 15, 15))])
    instance = build_instance_mask(semantic, metadata)
    direction = build_direction_field(instance, metadata)

    # Centroid is at (10, 10); direction there should be ~zero
    cx, cy = 10, 10
    dx, dy = direction[cy, cx]
    assert abs(dx) < 0.05
    assert abs(dy) < 0.05


def test_direction_field_points_toward_centroid() -> None:
    """A pixel left-of-centroid should have positive dx (pointing right)."""
    semantic = np.zeros((20, 20), dtype=np.uint8)
    semantic[5:15, 5:15] = 1
    metadata = _make_metadata([("A", 1, (5, 5, 15, 15))])
    instance = build_instance_mask(semantic, metadata)
    direction = build_direction_field(instance, metadata)
    # Pixel (5, 10) — left edge — centroid is at (10, 10) → dx should be > 0
    dx, _dy = direction[10, 5]
    assert dx > 0, f"Expected positive dx pointing right toward centroid; got {dx}"


def test_direction_field_zero_on_background() -> None:
    semantic = np.zeros((20, 20), dtype=np.uint8)
    semantic[5:15, 5:15] = 1
    metadata = _make_metadata([("A", 1, (5, 5, 15, 15))])
    instance = build_instance_mask(semantic, metadata)
    direction = build_direction_field(instance, metadata)
    # Far-corner background pixel should have zero direction
    assert (direction[0, 0] == 0).all()


def test_direction_field_unit_normalized_per_instance() -> None:
    """Maximum magnitude in any instance should be ≈ 1.0."""
    semantic = np.zeros((20, 20), dtype=np.uint8)
    semantic[5:15, 5:15] = 1
    metadata = _make_metadata([("A", 1, (5, 5, 15, 15))])
    instance = build_instance_mask(semantic, metadata)
    direction = build_direction_field(instance, metadata)
    instance_pixels = instance == 1
    magnitudes = np.linalg.norm(direction[instance_pixels], axis=1)
    assert magnitudes.max() <= 1.001
    assert magnitudes.max() >= 0.99  # at least one pixel near max range


def test_empty_metadata_returns_zero_targets() -> None:
    semantic = np.zeros((10, 10), dtype=np.uint8)
    instance = build_instance_mask(semantic, [])
    affinity = build_affinity_mask(semantic, [])
    direction = build_direction_field(instance, [])
    assert (instance == 0).all()
    assert (affinity == 0).all()
    assert (direction == 0).all()
