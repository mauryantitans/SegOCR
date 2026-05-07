"""Collision-mask placement tracker.

Per CharSeg Dataset Brainstorming #4: maintain a B/W collision mask that
accumulates all successfully-placed text pixels for the current image.
For each new placement attempt, build a temp B/W mask, AND-test against
the collision mask, retry up to N times with rotation/curve perturbations,
otherwise abandon the placement. Reset between images.

This sits between the renderer and the layout engine: layout proposes a
position+transform, the placement tracker validates non-collision, and on
success the collision mask is updated.
"""
from __future__ import annotations

import numpy as np


class PlacementMaskTracker:
    """Per-image collision-avoidance bookkeeping."""

    def __init__(
        self,
        image_size: tuple[int, int],
        max_retries: int = 5,
        min_separation_px: int = 2,
    ) -> None:
        h, w = image_size
        self.collision_mask: np.ndarray = np.zeros((h, w), dtype=np.uint8)
        self.max_retries = max_retries
        self.min_separation_px = min_separation_px

    def reset(self) -> None:
        """Clear the collision mask. Call once per image."""
        self.collision_mask.fill(0)

    def attempt_placement(
        self,
        candidate_mask: np.ndarray,
        offset: tuple[int, int],
    ) -> bool:
        """Test whether ``candidate_mask`` placed at ``offset`` collides.

        Returns:
            True if the placement is collision-free (does not overlap
            existing pixels in self.collision_mask, accounting for
            min_separation_px). Caller should then call commit().
        """
        raise NotImplementedError("PlacementMaskTracker.attempt_placement — Week 2")

    def commit(self, candidate_mask: np.ndarray, offset: tuple[int, int]) -> None:
        """Merge candidate into the running collision mask."""
        raise NotImplementedError("PlacementMaskTracker.commit — Week 2")

    def extract_polygon(self, mask: np.ndarray) -> np.ndarray:
        """Extract the outer-pixel polygon from a B/W mask.

        Per brainstorming #1: outer-most pixels are those with at least
        one 4-neighbour zero pixel.
        """
        raise NotImplementedError("PlacementMaskTracker.extract_polygon — Week 2")
