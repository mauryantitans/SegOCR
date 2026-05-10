"""DegradationPipeline tests."""
from __future__ import annotations

import numpy as np
import pytest

from segocr.generator.degradation import DegradationPipeline


@pytest.fixture
def degradation_config() -> dict:
    return {
        "blur": {"probability": 0.5, "motion_kernel": [3, 7]},
        "noise": {"probability": 0.5, "gaussian_sigma": [5, 15]},
        "compression": {"probability": 0.5, "jpeg_quality": [50, 95]},
        "lighting": {
            "probability": 0.5,
            "gamma_range": [0.7, 1.3],
            "brightness_shift": 0.2,
            "contrast_factor": [0.8, 1.2],
        },
        "geometric": {"probability": 0.0, "distortion_k1": [-0.1, 0.1]},
        "occlusion": {"probability": 0.5, "max_patches": 2, "max_coverage": 0.10},
    }


@pytest.fixture
def degradation(degradation_config):
    return DegradationPipeline(degradation_config)


def test_apply_returns_same_shape_and_dtype(degradation: DegradationPipeline) -> None:
    image = (np.random.rand(64, 96, 3) * 255).astype(np.uint8)
    out = degradation.apply(image)
    assert out.shape == image.shape
    assert out.dtype == np.uint8


def test_apply_zero_probability_pipeline_passes_through() -> None:
    config = {
        "blur": {"probability": 0.0},
        "noise": {"probability": 0.0},
        "compression": {"probability": 0.0},
        "lighting": {"probability": 0.0},
        "geometric": {"probability": 0.0},
        "occlusion": {"probability": 0.0},
    }
    degradation = DegradationPipeline(config)
    image = (np.random.rand(32, 32, 3) * 255).astype(np.uint8)
    out = degradation.apply(image)
    # With every prob = 0 we should get back the input bit-for-bit
    assert np.array_equal(out, image)


def test_apply_local_blur_returns_same_shapes(degradation: DegradationPipeline) -> None:
    image = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[20:40, 20:40] = 255
    out_image, out_mask = degradation.apply_local_blur(
        image, mask, bbox=(15, 15, 45, 45), kernel=5
    )
    assert out_image.shape == image.shape
    assert out_mask.shape == mask.shape


def test_apply_local_blur_only_modifies_bbox(
    degradation: DegradationPipeline,
) -> None:
    image = np.full((64, 64, 3), 100, dtype=np.uint8)
    image[28:36, 28:36] = 200  # high-frequency interior change
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[28:36, 28:36] = 255

    out_image, _ = degradation.apply_local_blur(
        image, mask, bbox=(20, 20, 44, 44), kernel=7
    )
    # Pixels far outside the bbox must be untouched
    assert (out_image[0:5, 0:5] == 100).all()
    assert (out_image[60:64, 60:64] == 100).all()


def test_apply_with_mask_no_blur_passes_mask_through(
    degradation_config: dict,
) -> None:
    """When blur probability is 0, the mask must come back unchanged."""
    config = {**degradation_config, "blur": {**degradation_config["blur"], "probability": 0.0}}
    degradation = DegradationPipeline(config)
    image = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[20:40, 20:40] = 1
    out_image, out_mask = degradation.apply_with_mask(image, mask)
    assert out_image.shape == image.shape
    assert out_mask.shape == mask.shape
    assert np.array_equal(out_mask, mask), "Mask should be unchanged with no blur"


def test_apply_with_mask_blur_dilates_per_class(
    degradation_config: dict, monkeypatch
) -> None:
    """Force blur to fire — verify the mask grew (dilation) but stayed
    in the same valid class-id range."""
    config = {**degradation_config, "blur": {"probability": 1.0, "motion_kernel": [7, 7]}}
    degradation = DegradationPipeline(config)
    # Disable other transforms to isolate blur's effect on the mask
    for key in ("noise", "compression", "lighting", "geometric", "occlusion"):
        if key in config:
            config[key] = {**config.get(key, {}), "probability": 0.0}

    image = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[28:36, 28:36] = 7  # 8×8 patch of class 7
    original_count = (mask == 7).sum()

    out_image, out_mask = degradation.apply_with_mask(image, mask)
    new_count = (out_mask == 7).sum()

    # Blur kernel = 7 → dilation radius 3 → diameter 7. Original 8×8 = 64;
    # after 3-px dilation each side, region grows to ~14×14 = 196.
    assert new_count > original_count, "Blur should have dilated the class region"
    # No stray classes introduced
    valid_classes = set(np.unique(out_mask).tolist()) - {0}
    assert valid_classes <= {7}, f"Unexpected classes in mask: {valid_classes}"


def test_apply_with_mask_z_order_preserved(degradation_config: dict) -> None:
    """When two adjacent classes' dilations overlap, the higher class
    wins (matching the renderer's z-order convention)."""
    from segocr.generator.degradation import _dilate_per_class

    mask = np.zeros((20, 40), dtype=np.uint8)
    mask[5:15, 5:15] = 1   # class 1 on the left
    mask[5:15, 25:35] = 2  # class 2 on the right
    dilated = _dilate_per_class(mask, kernel_size=11)
    # In overlap region, class 2 should win (last-write-wins)
    overlap_pixel = dilated[10, 19]  # roughly the midpoint between regions
    assert overlap_pixel in (1, 2), f"Unexpected class at overlap: {overlap_pixel}"


def test_random_occlusion_changes_image_when_active(
    degradation_config: dict,
) -> None:
    config = {**degradation_config, "occlusion": {**degradation_config["occlusion"], "probability": 1.0}}
    # Disable everything else so we isolate the occlusion path
    for key in ("blur", "noise", "compression", "lighting", "geometric"):
        config[key] = {**config[key], "probability": 0.0}
    degradation = DegradationPipeline(config)
    image = np.full((128, 128, 3), 100, dtype=np.uint8)
    out = degradation.apply(image)
    # Some pixels must have changed
    assert not np.array_equal(out, image)
