"""Layout engine — arranges rendered characters in 6 spatial configurations.

Implementation Guide §3.4 + Mode 6 (paragraph) added per brainstorming #11.
The exact same affine/perspective/elastic transform must be applied to
both the RGB image (INTER_LINEAR) and the class mask (INTER_NEAREST).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from segocr.generator.placement import PlacementMaskTracker


class LayoutEngine:
    """Arranges rendered characters using one of 6 layout modes."""

    def __init__(self, config: dict) -> None:
        """
        Args:
            config: the ``layout`` section of the YAML config.
        """
        self.config = config
        self.mode_probs = config["modes"]
        raise NotImplementedError("LayoutEngine.__init__ — Week 2/3")

    def apply_layout(
        self,
        char_images: list[np.ndarray],
        char_masks: list[np.ndarray],
        char_metadata: list[dict],
        image_size: tuple[int, int],
        placement: "PlacementMaskTracker | None" = None,
    ) -> tuple[np.ndarray, np.ndarray, list[dict]]:
        """Apply a randomly-sampled layout mode.

        Returns:
            composed_text:    (H, W, 4) RGBA arranged-text image.
            composed_mask:    (H, W)    class-id mask matching composed_text.
            updated_metadata: per-character bbox/centroid/angle after layout.
        """
        raise NotImplementedError("LayoutEngine.apply_layout — Week 2/3")

    def _sample_mode(self) -> str:
        """Sample a layout mode by configured probability."""
        raise NotImplementedError

    # ── Mode implementations ─────────────────────────────────────────────────
    def _horizontal(self, *args, **kwargs):
        """Mode 1: left-to-right placement with variable spacing."""
        raise NotImplementedError("LayoutEngine._horizontal — Week 2")

    def _rotated(self, *args, **kwargs):
        """Mode 2: render horizontal then rotate the entire block."""
        raise NotImplementedError("LayoutEngine._rotated — Week 2")

    def _curved(self, *args, **kwargs):
        """Mode 3: place chars along a parametric curve (Bézier / sinusoidal /
        circular). Each char individually tangent-rotated. Hardest mode."""
        raise NotImplementedError("LayoutEngine._curved — Week 3")

    def _perspective(self, *args, **kwargs):
        """Mode 4: render flat then 4-point perspective warp."""
        raise NotImplementedError("LayoutEngine._perspective — Week 3")

    def _deformed(self, *args, **kwargs):
        """Mode 5: thin-plate-spline / grid-displacement elastic warp."""
        raise NotImplementedError("LayoutEngine._deformed — Week 3")

    def _paragraph(self, *args, **kwargs):
        """Mode 6: multi-line paragraph layout (added per brainstorming #11).

        Document-OCR-style data: configurable line spacing, word spacing,
        left/centered/justified alignment, multiple lines per image.
        """
        raise NotImplementedError("LayoutEngine._paragraph — Week 3")
