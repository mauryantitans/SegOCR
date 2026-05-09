"""BackgroundGenerator tests."""
from __future__ import annotations

import numpy as np
import pytest

from segocr.generator.background import BackgroundGenerator


@pytest.fixture
def bg_config_no_natural() -> dict:
    return {
        "tier_distribution": {
            "tier1_solid": 0.25,
            "tier2_procedural": 0.25,
            "tier3_natural": 0.25,
            "tier4_adversarial": 0.25,
        },
        "natural_image_dirs": [],
        "preload_buffer_size": 8,
    }


@pytest.fixture
def bg_generator(bg_config_no_natural):
    return BackgroundGenerator(bg_config_no_natural)


def test_generate_returns_correct_shape_and_dtype(bg_generator: BackgroundGenerator) -> None:
    bg = bg_generator.generate((128, 256))
    assert bg.shape == (128, 256, 3)
    assert bg.dtype == np.uint8


def test_each_tier_method_produces_valid_output(bg_generator: BackgroundGenerator) -> None:
    size = (64, 96)
    for fn in (
        bg_generator._tier1_solid,
        bg_generator._tier2_procedural,
        bg_generator._tier3_natural,        # falls back to procedural
        bg_generator._tier4_adversarial,    # falls back paths exercised
    ):
        bg = fn(size)
        assert bg.shape == (64, 96, 3)
        assert bg.dtype == np.uint8


def test_tier1_solid_subtype_is_flat(
    bg_generator: BackgroundGenerator, monkeypatch
) -> None:
    """The 'solid' subtype must produce zero spatial variance."""
    monkeypatch.setattr(
        "segocr.generator.background.random.choice",
        lambda choices: "solid" if "solid" in choices else choices[0],
    )
    bg = bg_generator._tier1_solid((64, 64))
    # All pixels identical → spatial std is exactly zero per channel
    assert bg.std(axis=(0, 1)).sum() == 0


def test_tier1_gradient_subtype_has_variance(
    bg_generator: BackgroundGenerator, monkeypatch
) -> None:
    """Linear-gradient subtype should have non-trivial spatial variance."""
    monkeypatch.setattr(
        "segocr.generator.background.random.choice",
        lambda choices: "linear_gradient"
        if "linear_gradient" in choices
        else choices[0],
    )
    bg = bg_generator._tier1_solid((64, 64))
    assert bg.std() > 5


def test_tier3_falls_back_when_no_images(bg_generator: BackgroundGenerator) -> None:
    """No natural images on disk → tier 3 must still return something valid."""
    assert bg_generator.natural_image_paths == []
    bg = bg_generator._tier3_natural((64, 64))
    assert bg.shape == (64, 64, 3)
    assert bg.dtype == np.uint8


def test_tier_distribution_respects_weights() -> None:
    """A 100% tier1 config should always go through tier1."""
    config = {
        "tier_distribution": {
            "tier1_solid": 1.0,
            "tier2_procedural": 0.0,
            "tier3_natural": 0.0,
            "tier4_adversarial": 0.0,
        },
        "natural_image_dirs": [],
        "preload_buffer_size": 8,
    }
    bg = BackgroundGenerator(config)
    for _ in range(5):
        result = bg.generate((32, 32))
        assert result.shape == (32, 32, 3)


def test_failed_tier_falls_back_to_tier1(monkeypatch, bg_generator: BackgroundGenerator) -> None:
    """If a tier method raises, generate() must catch it and fall back."""
    def boom(*_args, **_kwargs):
        raise RuntimeError("simulated tier failure")
    monkeypatch.setattr(bg_generator, "_tier2_procedural", boom)
    monkeypatch.setattr(bg_generator, "_sample_tier", lambda: "tier2_procedural")
    bg = bg_generator.generate((48, 48))
    assert bg.shape == (48, 48, 3)
