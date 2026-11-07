"""Fourier Domain Adaptation (FDA) — Research Proposal §6.2 Option B.

Swap low-frequency Fourier components between synthetic and real images
to transfer color/style distribution. ~20 lines, no training required —
this is the "easy ticket" version of style transfer. Run before CycleGAN
for the cheap baseline.

Yang & Soatto, "FDA: Fourier Domain Adaptation for Semantic Segmentation",
CVPR 2020. Only the amplitude of a small centered low-frequency window is
replaced; the source phase (which carries stroke geometry) is kept, so the
synthetic class mask stays valid.
"""
from __future__ import annotations

import cv2
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
                      Resized to the source size if the shapes differ.
        beta:         fraction of the spectrum centered on DC to swap.
                      Typical 0.005–0.05.

    Returns:
        (H, W, 3) uint8 — synthetic content with target style.
    """
    h, w = src_image.shape[:2]
    if target_image.shape[:2] != (h, w):
        target_image = cv2.resize(target_image, (w, h), interpolation=cv2.INTER_AREA)

    src_fft = np.fft.fftshift(np.fft.fft2(src_image.astype(np.float32), axes=(0, 1)), axes=(0, 1))
    tgt_fft = np.fft.fftshift(np.fft.fft2(target_image.astype(np.float32), axes=(0, 1)), axes=(0, 1))

    amplitude = np.abs(src_fft)
    phase = np.angle(src_fft)

    b = int(np.floor(min(h, w) * beta))
    if b > 0:
        cy, cx = h // 2, w // 2
        window = (slice(cy - b, cy + b + 1), slice(cx - b, cx + b + 1))
        amplitude[window] = np.abs(tgt_fft)[window]

    mixed = np.fft.ifftshift(amplitude * np.exp(1j * phase), axes=(0, 1))
    out = np.real(np.fft.ifft2(mixed, axes=(0, 1)))
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)
