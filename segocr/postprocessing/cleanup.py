"""Raw-prediction cleanup. Implementation Guide §3.11 / Research Proposal §7.1.

(1) Confidence-threshold low-confidence pixels to background.
(2) Morphological opening (2×2) to remove salt noise.
(3) Morphological closing (2×2) to fill small holes.
(4) Connected-component area filter (drop components < 20 px).
"""
from __future__ import annotations

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
        (H, W) cleaned class-id map.
    """
    raise NotImplementedError("cleanup_prediction — Week 6")
