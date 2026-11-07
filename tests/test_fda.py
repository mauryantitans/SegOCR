"""Fourier Domain Adaptation tests."""
from __future__ import annotations

import cv2
import numpy as np

from segocr.adaptation.fda import fourier_domain_adaptation


def _text_image(h: int = 128, w: int = 128) -> np.ndarray:
    img = np.full((h, w, 3), 200, np.uint8)
    cv2.putText(img, "EXIT 42", (5, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (20, 20, 20), 3)
    return img


def _real_like(h: int = 128, w: int = 128) -> np.ndarray:
    # Warm, dark, smoothly lit "photo" — very different global style.
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    shade = 60 + 50 * (x / w) + 20 * (y / h)
    return np.stack([shade * 1.4, shade, shade * 0.6], axis=-1).clip(0, 255).astype(np.uint8)


def test_beta_zero_is_identity() -> None:
    src = _text_image()
    out = fourier_domain_adaptation(src, _real_like(), beta=0.0)
    assert np.array_equal(out, src)


def test_preserves_shape_and_dtype_with_mismatched_target() -> None:
    src = _text_image(96, 160)
    out = fourier_domain_adaptation(src, _real_like(200, 300), beta=0.02)
    assert out.shape == src.shape
    assert out.dtype == np.uint8


def test_moves_global_color_toward_target() -> None:
    src, tgt = _text_image(), _real_like()
    out = fourier_domain_adaptation(src, tgt, beta=0.02)
    src_gap = np.abs(src.mean(axis=(0, 1)) - tgt.mean(axis=(0, 1)))
    out_gap = np.abs(out.astype(np.float32).mean(axis=(0, 1)) - tgt.mean(axis=(0, 1)))
    assert np.all(out_gap < 0.2 * src_gap)


def test_preserves_text_strokes() -> None:
    """High-frequency stroke structure (what the masks describe) survives."""
    src = _text_image()
    out = fourier_domain_adaptation(src, _real_like(), beta=0.02)
    lap_src = cv2.Laplacian(cv2.cvtColor(src, cv2.COLOR_RGB2GRAY), cv2.CV_32F)
    lap_out = cv2.Laplacian(cv2.cvtColor(out, cv2.COLOR_RGB2GRAY), cv2.CV_32F)
    corr = np.corrcoef(lap_src.ravel(), lap_out.ravel())[0, 1]
    assert corr > 0.9
