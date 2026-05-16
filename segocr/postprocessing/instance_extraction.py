"""Instance extraction. Implementation Guide §3.11 / Research Proposal §7.2.

Per-class connected components on the cleaned class map. The direction
head is accepted for API completeness but not yet used for splitting
touching same-class characters — the affinity map gives us most of that
signal at reading-order time, and the splitter would be a separate
contribution worth careful evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class CharacterInstance:
    """Single detected character."""

    class_id: int
    bbox: tuple[int, int, int, int]            # (x, y, w, h)
    centroid: tuple[float, float]
    area: int
    pixels: np.ndarray | None = field(default=None, repr=False)
    confidence: float = 0.0


def extract_instances(
    clean_map: np.ndarray,
    direction_map: np.ndarray | None = None,  # noqa: ARG001 — reserved for splitter
    min_size: int = 8,
    max_size: int = 256,
) -> list[CharacterInstance]:
    """Extract individual character instances from a cleaned class map.

    Args:
        clean_map: (H, W) class-id map post-cleanup.
        direction_map: optional (2, H, W) unit-vector field. Reserved
            for a future same-class-splitter; currently ignored.
        min_size, max_size: bbox-side bounds for filtering bogus
            components. ``min_size`` rejects sub-character speckle that
            survived cleanup; ``max_size`` rejects blob-merges.

    Returns:
        List of ``CharacterInstance``, one per accepted connected component.
        Order is undefined; reading_order.recover_text reorders them.
    """
    if clean_map.size == 0 or int(clean_map.max()) == 0:
        return []

    num_classes = int(clean_map.max()) + 1
    instances: list[CharacterInstance] = []
    for c in range(1, num_classes):
        mask = (clean_map == c).astype(np.uint8)
        if mask.sum() == 0:
            continue
        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        for i in range(1, n_labels):
            x = int(stats[i, cv2.CC_STAT_LEFT])
            y = int(stats[i, cv2.CC_STAT_TOP])
            w = int(stats[i, cv2.CC_STAT_WIDTH])
            h = int(stats[i, cv2.CC_STAT_HEIGHT])
            area = int(stats[i, cv2.CC_STAT_AREA])
            longest_side = max(w, h)
            if longest_side < min_size or longest_side > max_size:
                continue
            instances.append(
                CharacterInstance(
                    class_id=c,
                    bbox=(x, y, w, h),
                    centroid=(float(centroids[i][0]), float(centroids[i][1])),
                    area=area,
                )
            )
    return instances
