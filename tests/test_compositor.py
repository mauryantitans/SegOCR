"""Compositor tests — verifies the mask-invariance contract."""
from __future__ import annotations

import numpy as np
import pytest

from segocr.generator.compositor import (
    Compositor,
    _alpha_composite,
    _contrast_aware_color,
    _contrast_ratio,
    _luminance,
)


@pytest.fixture
def compositor_config() -> dict:
    return {
        "modes": {
            "standard": 0.4,
            "semi_transparent": 0.1,
            "textured_fill": 0.1,
            "outline": 0.1,
            "shadow": 0.15,
            "emboss": 0.15,
        },
        "color_strategy": {
            "contrast_aware": 0.4,
            "random": 0.3,
            "low_contrast": 0.3,
        },
    }


@pytest.fixture
def compositor(compositor_config):
    return Compositor(compositor_config)


@pytest.fixture
def synthetic_text_inputs():
    """A 64×64 RGBA strip with a 20×20 white square 'character'."""
    rgba = np.zeros((64, 64, 4), dtype=np.uint8)
    rgba[20:40, 20:40] = (255, 255, 255, 255)
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[20:40, 20:40] = 1
    bg = np.full((64, 64, 3), 128, dtype=np.uint8)
    return rgba, mask, bg


def test_composite_preserves_mask(
    compositor: Compositor, synthetic_text_inputs
) -> None:
    rgba, mask, bg = synthetic_text_inputs
    original = mask.copy()
    for _ in range(20):
        _, out_mask = compositor.composite(rgba.copy(), mask.copy(), bg.copy())
        assert np.array_equal(out_mask, original), "compositor mutated the class mask"


def test_composite_returns_correct_shape(
    compositor: Compositor, synthetic_text_inputs
) -> None:
    rgba, mask, bg = synthetic_text_inputs
    out_rgb, out_mask = compositor.composite(rgba, mask, bg)
    assert out_rgb.shape == bg.shape
    assert out_mask.shape == mask.shape


def test_composite_size_mismatch_raises(
    compositor: Compositor, synthetic_text_inputs
) -> None:
    rgba, mask, bg = synthetic_text_inputs
    with pytest.raises(ValueError):
        compositor.composite(rgba, mask, bg[:32])


def test_alpha_composite_full_alpha_replaces_bg() -> None:
    fg = np.zeros((4, 4, 4), dtype=np.uint8)
    fg[..., :3] = 255
    fg[..., 3] = 255
    bg = np.zeros((4, 4, 3), dtype=np.uint8)
    out = _alpha_composite(fg, bg)
    assert (out == 255).all()


def test_alpha_composite_zero_alpha_preserves_bg() -> None:
    fg = np.zeros((4, 4, 4), dtype=np.uint8)
    fg[..., :3] = 255
    fg[..., 3] = 0
    bg = np.full((4, 4, 3), 17, dtype=np.uint8)
    out = _alpha_composite(fg, bg)
    assert (out == 17).all()


def test_luminance_extremes() -> None:
    assert _luminance((0, 0, 0)) == pytest.approx(0.0, abs=1e-6)
    assert _luminance((255, 255, 255)) == pytest.approx(1.0, abs=1e-3)


def test_contrast_ratio_white_on_black_is_max() -> None:
    ratio = _contrast_ratio((255, 255, 255), (0, 0, 0))
    assert ratio == pytest.approx(21.0, abs=0.01)


def test_contrast_aware_picks_high_contrast_color() -> None:
    bg = np.full((64, 64, 3), 0, dtype=np.uint8)  # black background
    color = _contrast_aware_color(bg, (0, 0, 64, 64))
    # Whatever was picked, it should have high contrast against black.
    assert _contrast_ratio(color, (0, 0, 0)) >= 4.5


def test_modes_all_produce_valid_output(
    compositor: Compositor, synthetic_text_inputs
) -> None:
    rgba, mask, bg = synthetic_text_inputs
    for mode in ("standard", "semi_transparent", "textured_fill", "outline", "shadow", "emboss"):
        # Force the mode by monkey-patching the sampler
        compositor._sample_mode = lambda m=mode: m  # type: ignore[method-assign]
        out_rgb, out_mask = compositor.composite(rgba.copy(), mask.copy(), bg.copy())
        assert out_rgb.shape == bg.shape
        assert out_rgb.dtype == np.uint8
        assert np.array_equal(out_mask, mask)


def test_emboss_brightness_modulation_localized(
    compositor: Compositor, synthetic_text_inputs
) -> None:
    """Emboss must change pixels at character boundaries but leave pixels
    far from the mask largely untouched."""
    rgba, mask, bg = synthetic_text_inputs
    compositor._sample_mode = lambda: "emboss"  # type: ignore[method-assign]
    out_rgb, _ = compositor.composite(rgba, mask, bg)
    # Far-corner pixel should be very close to original background.
    assert np.abs(int(out_rgb[0, 0, 0]) - int(bg[0, 0, 0])) <= 5
