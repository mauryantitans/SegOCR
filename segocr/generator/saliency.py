"""Edge-density-based saliency for "realistic" text placement.

Implements the Low-tier of the smart-placement upgrade. The classic
SynthText-paper approach minus depth/plane fitting: a per-pixel score
where high values mean "good place to put text" — low local edge
density, low local color variance, well away from the image border.

This module is cheap (no learned models, pure cv2/numpy) and runs in
~5–10 ms on a 512×512 background. It's used by the engine when the
configured ``layout.placement.realistic_fraction`` triggers; otherwise
text is placed at a uniform-random position as before.

Caveats:
    - The score doesn't know about semantic categories (it can't tell a
      flat wall from a flat patch of sky). The Mid-tier upgrade —
      COCO-segmentation-based or SAM-based placement — would fix that.
    - The score is computed in image-space, not world-space, so it
      doesn't enforce text alignment with surface normals. That's also
      Mid-tier work.
"""
from __future__ import annotations

import cv2
import numpy as np

LOCAL_WINDOW = 15            # pixels — local-edge / local-variance window
EDGE_PENALTY_FRACTION = 1 / 16  # of min(H, W) — pixels near the image border are penalized
EDGE_PENALTY_FACTOR = 0.2    # multiplier applied within the penalty band


def compute_placement_score(background: np.ndarray) -> np.ndarray:
    """Compute a per-pixel placement-suitability score.

    Args:
        background: (H, W, 3) uint8 RGB.

    Returns:
        (H, W) float32 score in roughly [0, 1]. Higher = better text spot.
    """
    if background.ndim != 3 or background.shape[2] != 3:
        raise ValueError(f"Expected (H, W, 3) RGB; got {background.shape}")

    gray = cv2.cvtColor(background, cv2.COLOR_RGB2GRAY)

    # Local edge density: average over LOCAL_WINDOW of canny output.
    edges = cv2.Canny(gray, 50, 150).astype(np.float32) / 255.0
    edge_density = cv2.boxFilter(edges, -1, (LOCAL_WINDOW, LOCAL_WINDOW))

    # Local color std-dev (E[X^2] - E[X]^2).
    g = gray.astype(np.float32)
    mean = cv2.boxFilter(g, -1, (LOCAL_WINDOW, LOCAL_WINDOW))
    sq_mean = cv2.boxFilter(g * g, -1, (LOCAL_WINDOW, LOCAL_WINDOW))
    variance = np.maximum(0.0, sq_mean - mean * mean)
    std_dev = np.sqrt(variance)
    max_std = float(std_dev.max())
    std_norm = std_dev / max(1.0, max_std)

    score = (1.0 - edge_density) * (1.0 - std_norm)

    # Penalize pixels near the image border — text touching the edge
    # looks unnatural in real photos.
    h, w = score.shape
    pad = max(8, int(min(h, w) * EDGE_PENALTY_FRACTION))
    if pad > 0:
        penalty = np.ones_like(score)
        penalty[:pad, :] *= EDGE_PENALTY_FACTOR
        penalty[-pad:, :] *= EDGE_PENALTY_FACTOR
        penalty[:, :pad] *= EDGE_PENALTY_FACTOR
        penalty[:, -pad:] *= EDGE_PENALTY_FACTOR
        score = score * penalty

    return score.astype(np.float32)


def find_best_position(
    score_map: np.ndarray,
    region_shape: tuple[int, int],
    randomness: float = 0.3,
) -> tuple[int, int]:
    """Return (y_offset, x_offset) for placing a region of ``region_shape``.

    Picks the position whose ``region_shape``-window mean score is highest;
    if ``randomness > 0``, picks uniformly at random among the top-k% of
    valid positions so the same background doesn't always produce the
    same placement.

    Args:
        score_map:   (H, W) float32 from ``compute_placement_score``.
        region_shape: (target_h, target_w) of the strip we want to place.
        randomness:  in [0, 1]. 0 = deterministic argmax. Default 0.3.

    Returns:
        (y_offset, x_offset). Both are in the integer pixel range
        [0, H - target_h] × [0, W - target_w].
    """
    target_h, target_w = region_shape
    map_h, map_w = score_map.shape

    if target_h <= 0 or target_w <= 0:
        return 0, 0
    if target_h >= map_h or target_w >= map_w:
        return 0, 0

    # Window-mean: each pixel = mean score of a target_h × target_w window
    # whose top-left corner is at that pixel.
    integral = cv2.integral(score_map.astype(np.float64))
    h_out = map_h - target_h + 1
    w_out = map_w - target_w + 1
    window_sum = (
        integral[target_h : target_h + h_out, target_w : target_w + w_out]
        - integral[:h_out, target_w : target_w + w_out]
        - integral[target_h : target_h + h_out, :w_out]
        + integral[:h_out, :w_out]
    )
    window_mean = window_sum / (target_h * target_w)

    if randomness > 0:
        flat = window_mean.flatten()
        # Top-k%: keep the best `randomness * 50%` of candidates.
        threshold = float(np.quantile(flat, max(0.0, 1.0 - randomness * 0.5)))
        candidates = np.argwhere(window_mean >= threshold)
        if len(candidates) > 0:
            choice = candidates[np.random.randint(len(candidates))]
            return int(choice[0]), int(choice[1])

    flat_idx = int(window_mean.argmax())
    y, x = np.unravel_index(flat_idx, window_mean.shape)
    return int(y), int(x)
