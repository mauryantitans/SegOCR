"""Character renderer — the oracle.

Implementation Guide §3.3. Renders individual characters via Pillow and
extracts the alpha channel as the pixel-perfect segmentation mask. Class
labels are assigned per character; overlapping pixels resolve via z-order.

Critical: render at 2× resolution and downsample with LANCZOS for the image
and NEAREST for the mask. Bilinear/bicubic on masks creates invalid
fractional class IDs (Implementation Guide §6 gotcha #1).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from PIL.ImageFont import FreeTypeFont

    from segocr.generator.font_manager import FontManager


class CharacterRenderer:
    """Renders characters and extracts oracle segmentation masks."""

    def __init__(self, config: dict, font_manager: "FontManager") -> None:
        self.config = config
        self.font_manager = font_manager
        # Character → class-id map. 0 = background, 1..N = characters.
        self.char_to_class: dict[str, int] = self._build_class_map()
        raise NotImplementedError("CharacterRenderer.__init__ — Week 2")

    def render_character(
        self,
        char: str,
        font: "FreeTypeFont",
        size: int,
        color: tuple[int, int, int],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Render a single character on a transparent canvas.

        Returns:
            char_image: (H, W, 4) RGBA uint8 array.
            char_mask:  (H, W)    binary uint8 array — the alpha-channel
                                  thresholded at 128 (the oracle).
        """
        raise NotImplementedError("CharacterRenderer.render_character — Week 2")

    def render_text(
        self,
        text: str,
        font: "FreeTypeFont",
        size: int,
        color: tuple[int, int, int],
        char_spacing: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray, list[dict]]:
        """Render a full text string as a horizontal strip.

        Returns:
            text_image: (H, W, 4) RGBA composite of all characters.
            class_mask: (H, W)    each pixel = character class ID (0 = bg).
            char_metadata: per-character dicts with class_id, bbox,
                           centroid, width, height.
        """
        raise NotImplementedError("CharacterRenderer.render_text — Week 2")

    def get_class_id(self, char: str) -> int:
        """Map a character to its class ID. 0 = background, 1..62 alphanumeric."""
        return self.char_to_class.get(char, 0)

    def _build_class_map(self) -> dict[str, int]:
        """Build the canonical char → class-id map for the active tier."""
        raise NotImplementedError("CharacterRenderer._build_class_map — Week 2")
