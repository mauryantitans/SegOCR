"""4-tier background generator.

Implementation Guide §3.5. Tier distribution is the single biggest knob
for real-world generalization. Tier 3 is the bottleneck — pre-load a
buffer of images to amortize disk I/O.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


class BackgroundGenerator:
    """Generates backgrounds at 4 complexity tiers."""

    def __init__(self, config: dict) -> None:
        """
        Args:
            config: the ``background`` section of the YAML config.
        """
        self.config = config
        self.tier_probs = config["tier_distribution"]
        self.natural_image_paths: list[Path] = []
        self.preload_buffer: list[np.ndarray] = []
        self.preload_buffer_size = config.get("preload_buffer_size", 200)
        raise NotImplementedError("BackgroundGenerator.__init__ — Week 2")

    def generate(self, size: tuple[int, int]) -> np.ndarray:
        """Generate a background at the configured tier distribution.

        Returns:
            (H, W, 3) uint8 RGB array.
        """
        raise NotImplementedError("BackgroundGenerator.generate — Week 2")

    def _tier1_solid(self, size: tuple[int, int]) -> np.ndarray:
        """Solid color, linear gradient, or radial gradient."""
        raise NotImplementedError("BackgroundGenerator._tier1_solid — Week 2")

    def _tier2_procedural(self, size: tuple[int, int]) -> np.ndarray:
        """Perlin noise at 1–4 octaves, optional random colormap."""
        raise NotImplementedError("BackgroundGenerator._tier2_procedural — Week 2")

    def _tier3_natural(self, size: tuple[int, int]) -> np.ndarray:
        """Random crop from COCO/DTD/Places365.

        Use the preload buffer to avoid per-sample disk I/O.
        """
        raise NotImplementedError("BackgroundGenerator._tier3_natural — Week 3")

    def _tier4_adversarial(self, size: tuple[int, int]) -> np.ndarray:
        """Backgrounds designed to confuse the model:
        (A) document images with other text,
        (B) synthetic text-like patterns rendered faintly,
        (C) tier-3 tinted to match a foreground color,
        (D) dense COCO clutter.
        """
        raise NotImplementedError("BackgroundGenerator._tier4_adversarial — Week 3/4")

    def _refresh_preload_buffer(self) -> None:
        """Repopulate the preload buffer with random images from disk."""
        raise NotImplementedError("BackgroundGenerator._refresh_preload_buffer — Week 3")
