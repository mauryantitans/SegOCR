"""Qualitative visualization utilities — overlays, heatmaps, error-case grids."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def overlay_predictions(
    image: np.ndarray,
    pred_mask: np.ndarray,
    target_mask: np.ndarray | None = None,
    alpha: float = 0.5,
) -> np.ndarray:
    """Overlay class colors on the input image; if ``target_mask`` given,
    diff regions are highlighted in red. Returns an (H, W, 3) RGB array."""
    raise NotImplementedError("overlay_predictions — Week 4 (validator)")


def save_qualitative_grid(
    samples: list[dict],
    output_path: str | Path,
    grid_size: tuple[int, int] = (4, 4),
) -> None:
    """Save a grid of (image | pred | target) for visual inspection."""
    raise NotImplementedError("save_qualitative_grid — Week 6")
