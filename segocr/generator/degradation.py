"""Degradation pipeline — realistic capture-condition augmentations.

Implementation Guide §3.7. Applied to RGB images only — the class mask is
never modified by degradations. Albumentations covers ~80% of these out
of the box; shadow overlay and occlusion patches are custom.

JPEG compression is the single most impactful degradation for real-world
performance — without it, the model exploits high-frequency details that
JPEG destroys. Do not lower its probability.
"""
from __future__ import annotations

import numpy as np


class DegradationPipeline:
    """Applies degradations to composite images."""

    def __init__(self, config: dict) -> None:
        self.config = config
        self.pipeline = self._build_pipeline(config)
        raise NotImplementedError("DegradationPipeline.__init__ — Week 3")

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply random degradations to the RGB image.

        Mask is never touched — caller passes it through.
        """
        raise NotImplementedError("DegradationPipeline.apply — Week 3")

    def apply_local_blur(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        bbox: tuple[int, int, int, int],
        kernel: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """SynthText-style: gaussian-blur a bbox region of image AND mask
        with the same kernel, then re-extract the polygon shape from the
        blurred mask (per dataset brainstorming #10).
        """
        raise NotImplementedError("DegradationPipeline.apply_local_blur — Week 3")

    def _build_pipeline(self, config: dict):
        """Build the albumentations Compose stack."""
        raise NotImplementedError

    def _apply_random_shadow(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _apply_random_occlusion(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError
