"""Fourier Domain Adaptation (FDA) — Research Proposal §6.2 Option B.

Swap low-frequency Fourier components between synthetic and real images
to transfer color/style distribution. ~20 lines, no training required —
this is the "easy ticket" version of style transfer. Run before CycleGAN
for the cheap baseline.
"""
from __future__ import annotations

import numpy as np


def fourier_domain_adaptation(
    src_image: np.ndarray,
    target_image: np.ndarray,
    beta: float = 0.01,
) -> np.ndarray:
    """Transfer the low-frequency style of ``target_image`` onto ``src_image``.

    Args:
        src_image:    (H, W, 3) uint8 — synthetic image to be stylized.
        target_image: (H, W, 3) uint8 — real image whose style we want.
        beta:         fraction of the spectrum centered on DC to swap.
                      Typical 0.005–0.05.

    Returns:
        (H, W, 3) uint8 — synthetic content with target style.
    """
    raise NotImplementedError("fourier_domain_adaptation — Week 8")
