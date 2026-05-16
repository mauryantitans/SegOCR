"""Raw-prediction cleanup. Implementation Guide §3.11 / Research Proposal §7.1.

(1) Confidence-threshold low-confidence pixels to background.
(2) Per-class morphological opening (2×2) to remove salt noise.
(3) Per-class morphological closing (2×2) to fill small holes.
(4) Per-class connected-component area filter (drop components < 20 px).
"""
from __future__ import annotations

import cv2
import numpy as np


def cleanup_prediction(
    pred_map: np.ndarray,
    confidence_map: np.ndarray,
    threshold: float = 0.5,
    min_component_area: int = 20,
) -> np.ndarray:
    """Clean a raw argmax prediction map.

    Args:
        pred_map: (H, W) class-id argmax of model output.
        confidence_map: (H, W) max-softmax probability per pixel.
        threshold: confidence below which pixel → background.
        min_component_area: drop CCs smaller than this.

    Returns:
        (H, W) cleaned class-id map, same dtype as ``pred_map``.
    """
    if pred_map.shape != confidence_map.shape:
        raise ValueError(
            f"pred_map {pred_map.shape} and confidence_map "
            f"{confidence_map.shape} must have the same shape"
        )

    # Step 1 — confidence gate
    gated = np.where(confidence_map >= threshold, pred_map, 0).astype(pred_map.dtype)
    if gated.max() == 0:
        return gated

    # Steps 2-4 — per foreground class: morph open + close + area filter
    kernel = np.ones((2, 2), dtype=np.uint8)
    num_classes = int(gated.max()) + 1
    result = np.zeros_like(gated)
    for c in range(1, num_classes):
        mask = (gated == c).astype(np.uint8)
        if mask.sum() == 0:
            continue
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for i in range(1, n_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_component_area:
                result[labels == i] = c
    return result
