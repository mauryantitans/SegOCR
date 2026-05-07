"""Text-onto-background compositor.

Implementation Guide §3.6. Compositing modifies only the RGB image — the
class mask is invariant under all compositing operations (it represents
ground-truth character locations regardless of visual style).
"""
from __future__ import annotations

import numpy as np


class Compositor:
    """Composites rendered text onto backgrounds."""

    def __init__(self, config: dict) -> None:
        self.config = config
        self.mode_probs = config["modes"]
        self.color_strategy_probs = config["color_strategy"]
        raise NotImplementedError("Compositor.__init__ — Week 2")

    def composite(
        self,
        text_rgba: np.ndarray,
        text_mask: np.ndarray,
        background: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Composite text onto background.

        Returns:
            (final_rgb, final_class_mask). Mask is unchanged — passed
            through for caller convenience.
        """
        raise NotImplementedError("Compositor.composite — Week 2")

    def _select_text_color(
        self,
        background: np.ndarray,
        text_region_bbox: tuple[int, int, int, int],
    ) -> tuple[int, int, int]:
        """Select text color per the configured strategy:
        - contrast_aware: enforce >=4.5:1 contrast vs region mean
        - random: uniform RGB
        - low_contrast: deliberately close to region mean
        """
        raise NotImplementedError("Compositor._select_text_color — Week 2")

    def _apply_shadow(
        self,
        text_rgba: np.ndarray,
        offset: tuple[int, int] = (3, 3),
        blur: float = 2.0,
        opacity: float = 0.5,
    ) -> np.ndarray:
        raise NotImplementedError("Compositor._apply_shadow — Week 3")

    def _apply_emboss(
        self, text_mask: np.ndarray, light_angle: float = 45.0, depth: float = 2.0
    ) -> np.ndarray:
        """Simulate engraved/raised text using mask as height map."""
        raise NotImplementedError("Compositor._apply_emboss — Week 3")

    def _apply_textured_fill(
        self, text_rgba: np.ndarray, texture: np.ndarray
    ) -> np.ndarray:
        """Fill text with a texture pattern (alpha cookie-cutter)."""
        raise NotImplementedError("Compositor._apply_textured_fill — Week 3")
