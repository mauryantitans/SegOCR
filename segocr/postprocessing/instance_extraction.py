"""Instance extraction. Implementation Guide §3.11 / Research Proposal §7.2.

If the direction head is available: cluster pixels by direction-vector
convergence (this naturally separates adjacent same-class characters,
and disambiguates rotated symmetric chars per the brainstorming concern
about M↔W, 6↔9, O↔0).

Without direction: per-class connected components; watershed-split blobs
that are unreasonably large.
"""
from __future__ import annotations

from dataclasses import dataclass, field

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
    direction_map: np.ndarray | None = None,
    min_size: int = 8,
    max_size: int = 256,
) -> list[CharacterInstance]:
    """Extract individual character instances from a cleaned class map.

    Args:
        clean_map: (H, W) class-id map post-cleanup.
        direction_map: optional (2, H, W) unit-vector field from direction head.
        min_size, max_size: bbox-side bounds for filtering bogus components.
    """
    raise NotImplementedError("extract_instances — Week 6")
