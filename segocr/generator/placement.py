"""Collision-mask placement tracker.

Per CharSeg Dataset Brainstorming #4: maintain a B/W collision mask that
accumulates all successfully-placed text pixels for the current image.
For each new placement attempt, build a temp B/W mask, AND-test against
the collision mask, retry up to N times with rotation/curve perturbations,
otherwise abandon the placement. Reset between images.

Used by the engine to support multiple text instances per image (sign
with several scattered words, document with multiple paragraphs, etc.).
A single-instance image can skip the tracker entirely.
"""
from __future__ import annotations

import cv2
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
        self.image_size = (h, w)
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
        """Test whether ``candidate_mask`` placed at ``offset = (y, x)``
        collides with anything already committed.

        ``min_separation_px`` is enforced by morphologically dilating the
        candidate before the AND-test.

        Returns True iff the placement is collision-free. The caller
        should then call commit() with the same arguments.
        """
        canvas_candidate = self._stamp_on_canvas(candidate_mask, offset)
        if canvas_candidate is None:
            return False
        if self.min_separation_px > 0:
            kernel_size = 2 * self.min_separation_px + 1
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            test_canvas = cv2.dilate(canvas_candidate, kernel, iterations=1)
        else:
            test_canvas = canvas_candidate
        overlap = np.logical_and(test_canvas > 0, self.collision_mask > 0)
        return not overlap.any()

    def commit(
        self,
        candidate_mask: np.ndarray,
        offset: tuple[int, int],
    ) -> None:
        """Merge candidate into the running collision mask."""
        canvas_candidate = self._stamp_on_canvas(candidate_mask, offset)
        if canvas_candidate is not None:
            self.collision_mask = np.logical_or(
                self.collision_mask > 0, canvas_candidate > 0
            ).astype(np.uint8)

    def extract_polygon(self, mask: np.ndarray) -> np.ndarray:
        """Extract the outer-boundary pixel coordinates from a B/W mask.

        Per dataset brainstorming #1: outer-most pixels are those that have
        at least one 4-neighbour zero pixel. Returns an (N, 2) array of
        (x, y) coordinates in arbitrary order. Holes are not extracted.
        """
        bin_mask = (mask > 0).astype(np.uint8)
        kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.uint8)
        # Where a foreground pixel has a zero in any 4-neighbour direction,
        # it's a boundary pixel. Equivalently: it differs from its 4-erosion.
        eroded = cv2.erode(bin_mask, kernel, iterations=1)
        boundary = bin_mask & ~eroded
        ys, xs = np.where(boundary > 0)
        return np.stack([xs, ys], axis=1)

    # ── Internal ────────────────────────────────────────────────────────────

    def _stamp_on_canvas(
        self,
        candidate_mask: np.ndarray,
        offset: tuple[int, int],
    ) -> np.ndarray | None:
        """Place candidate at offset on a canvas-sized array, clipping to
        image bounds. Returns None if the candidate would land entirely
        outside the canvas."""
        canvas_h, canvas_w = self.image_size
        cand_h, cand_w = candidate_mask.shape[:2]
        y_off, x_off = offset

        y0 = max(0, y_off)
        x0 = max(0, x_off)
        y1 = min(canvas_h, y_off + cand_h)
        x1 = min(canvas_w, x_off + cand_w)
        if y1 <= y0 or x1 <= x0:
            return None

        canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
        cand_y0 = y0 - y_off
        cand_x0 = x0 - x_off
        cand_y1 = cand_y0 + (y1 - y0)
        cand_x1 = cand_x0 + (x1 - x0)
        canvas[y0:y1, x0:x1] = (candidate_mask[cand_y0:cand_y1, cand_x0:cand_x1] > 0).astype(
            np.uint8
        )
        return canvas
