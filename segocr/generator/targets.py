"""Build derived training targets from semantic mask + char metadata.

Three derived outputs are needed beyond the semantic mask (the primary
oracle):

    instance_mask   (H, W)   uint16 — unique ID per character instance.
                                       Separates "AA" into two distinct
                                       characters even though they share
                                       a class.

    affinity_mask   (H, W)   uint16 — word-group ID per pixel. Used as
                                       the affinity-head target. Pairs of
                                       adjacent characters within the
                                       same word are linked by a region
                                       between them.

    direction_field (H, W, 2) float32 — unit vector from each foreground
                                         pixel to the centroid of its
                                         character instance. Used as the
                                         direction-head target. Critical
                                         for disambiguating rotation-
                                         symmetric characters (M↔W, 6↔9).

All three are derivable from (semantic_mask, char_metadata), so the
generator can save just those two and leave derivation to the dataset
loader. We compute them up front anyway so smoke-tests can verify them.
"""
from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

WORD_GAP_RATIO = 1.5  # gap > 1.5× median char width → word boundary


def build_instance_mask(
    semantic_mask: np.ndarray,
    char_metadata: list[dict[str, Any]],
) -> np.ndarray:
    """Assign a unique instance ID to each character.

    Each char's pixels = pixels in its bbox with semantic_mask == its class_id.
    Later chars overwrite earlier on overlap (matches z-order in renderer).

    Returns (H, W) uint16 with 0 = background and 1..N = instances in the
    order they appear in ``char_metadata``.
    """
    h, w = semantic_mask.shape
    instance_mask = np.zeros((h, w), dtype=np.uint16)

    for instance_id, m in enumerate(char_metadata, start=1):
        class_id = m["class_id"]
        x0, y0, x1, y1 = m["bbox"]
        x0 = max(0, x0)
        y0 = max(0, y0)
        x1 = min(w, x1)
        y1 = min(h, y1)
        if x1 <= x0 or y1 <= y0:
            continue
        region = semantic_mask[y0:y1, x0:x1]
        char_pixels = region == class_id
        instance_mask[y0:y1, x0:x1] = np.where(
            char_pixels, instance_id, instance_mask[y0:y1, x0:x1]
        )
    return instance_mask


def build_affinity_mask(
    semantic_mask: np.ndarray,
    char_metadata: list[dict[str, Any]],
    word_gap_ratio: float = WORD_GAP_RATIO,
) -> np.ndarray:
    """Build the affinity (word-grouping) mask.

    Word boundaries are inferred from inter-character gaps in metadata:
    consecutive characters whose horizontal gap exceeds ``word_gap_ratio``
    × the median gap are considered to be in different words. Within a
    word, foreground pixels of the constituent characters share the
    same word ID, and a connecting "link" region is drawn between
    adjacent character centroids so the affinity-head loss has signal
    on the inter-character pixels.

    For paragraph-mode metadata where each entry has a ``line_index``,
    word grouping is constrained to within a single line.

    Returns (H, W) uint16 with 0 = background / no-link and 1..M = word IDs.
    """
    h, w = semantic_mask.shape
    affinity_mask = np.zeros((h, w), dtype=np.uint16)
    if not char_metadata:
        return affinity_mask

    word_ids = _assign_word_ids(char_metadata, word_gap_ratio)

    # Stamp word IDs onto each char's pixels (using semantic mask as the cookie-cutter).
    for m, word_id in zip(char_metadata, word_ids, strict=True):
        class_id = m["class_id"]
        x0, y0, x1, y1 = m["bbox"]
        x0 = max(0, x0)
        y0 = max(0, y0)
        x1 = min(w, x1)
        y1 = min(h, y1)
        if x1 <= x0 or y1 <= y0:
            continue
        region = semantic_mask[y0:y1, x0:x1]
        affinity_mask[y0:y1, x0:x1] = np.where(
            region == class_id, word_id, affinity_mask[y0:y1, x0:x1]
        )

    # Draw inter-character links between adjacent chars in the same word.
    for i in range(len(char_metadata) - 1):
        if word_ids[i] != word_ids[i + 1] or word_ids[i] == 0:
            continue
        cx0, cy0 = char_metadata[i]["centroid"]
        cx1, cy1 = char_metadata[i + 1]["centroid"]
        link_thickness = max(2, int(_estimate_char_height(char_metadata[i]) * 0.25))
        cv2.line(
            affinity_mask,
            (int(cx0), int(cy0)),
            (int(cx1), int(cy1)),
            color=int(word_ids[i]),
            thickness=link_thickness,
        )
    return affinity_mask


def build_direction_field(
    instance_mask: np.ndarray,
    char_metadata: list[dict[str, Any]],
) -> np.ndarray:
    """Build the direction field — unit vector from each foreground pixel
    to its instance's centroid, normalized so that the maximum magnitude
    in any instance is 1.

    Returns (H, W, 2) float32. Channel 0 = dx, channel 1 = dy.
    Background pixels are (0, 0).
    """
    h, w = instance_mask.shape
    direction = np.zeros((h, w, 2), dtype=np.float32)
    if not char_metadata:
        return direction

    centroids: dict[int, tuple[float, float]] = {}
    for instance_id, m in enumerate(char_metadata, start=1):
        centroids[instance_id] = m["centroid"]

    y_coords, x_coords = np.meshgrid(
        np.arange(h, dtype=np.float32),
        np.arange(w, dtype=np.float32),
        indexing="ij",
    )

    for instance_id, (cx, cy) in centroids.items():
        instance_pixels = instance_mask == instance_id
        if not instance_pixels.any():
            continue
        dx = cx - x_coords
        dy = cy - y_coords
        magnitude = np.sqrt(dx * dx + dy * dy)
        max_mag = float(magnitude[instance_pixels].max())
        if max_mag < 1e-6:
            continue
        direction[instance_pixels, 0] = dx[instance_pixels] / max_mag
        direction[instance_pixels, 1] = dy[instance_pixels] / max_mag
    return direction


# ── Helpers ─────────────────────────────────────────────────────────────────


def _assign_word_ids(
    char_metadata: list[dict[str, Any]],
    word_gap_ratio: float,
) -> list[int]:
    """Assign a word ID (1..M) to each character. 0 if char is alone."""
    if not char_metadata:
        return []

    # Group by line first if line_index is present (paragraph mode)
    by_line: dict[int, list[int]] = {}
    for i, m in enumerate(char_metadata):
        by_line.setdefault(m.get("line_index", 0), []).append(i)

    word_ids = [0] * len(char_metadata)
    next_word_id = 1
    for _line_idx, indices in by_line.items():
        if not indices:
            continue
        # Use median bbox width as the spacing reference
        widths = [
            char_metadata[i]["bbox"][2] - char_metadata[i]["bbox"][0]
            for i in indices
        ]
        median_w = float(np.median(widths)) if widths else 1.0
        gap_threshold = max(2.0, median_w * word_gap_ratio)

        word_ids[indices[0]] = next_word_id
        for k in range(1, len(indices)):
            prev = char_metadata[indices[k - 1]]
            curr = char_metadata[indices[k]]
            prev_x_end = prev["bbox"][2]
            curr_x_start = curr["bbox"][0]
            gap = curr_x_start - prev_x_end
            if gap > gap_threshold:
                next_word_id += 1
            word_ids[indices[k]] = next_word_id
        next_word_id += 1
    return word_ids


def _estimate_char_height(m: dict[str, Any]) -> float:
    x0, y0, x1, y1 = m["bbox"]
    return max(1.0, float(y1 - y0))
