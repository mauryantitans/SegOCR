"""Multi-head UNet model tests."""
from __future__ import annotations

import pytest
import torch

from segocr.models.unet import SegOCRUNet, build_model


@pytest.fixture
def model_config() -> dict:
    return {
        "architecture": "unet",
        "encoder": "resnet18",   # smaller than resnet50 for fast tests
        "encoder_weights": None,  # skip ImageNet download in tests
        "num_classes": 8,
        "head_features": 16,
        "decoder_channels": [128, 64, 32, 16, 16],
        "heads": {
            "semantic": True,
            "affinity": True,
            "direction": True,
        },
    }


def test_forward_returns_three_heads(model_config: dict) -> None:
    model = SegOCRUNet(model_config)
    x = torch.randn(1, 3, 64, 64)
    out = model(x)
    assert "semantic" in out
    assert "affinity" in out
    assert "direction" in out


def test_forward_output_shapes(model_config: dict) -> None:
    model = SegOCRUNet(model_config)
    x = torch.randn(2, 3, 64, 64)
    out = model(x)
    assert out["semantic"].shape == (2, 8, 64, 64)
    assert out["affinity"].shape == (2, 1, 64, 64)
    assert out["direction"].shape == (2, 2, 64, 64)


def test_optional_heads_can_be_disabled(model_config: dict) -> None:
    config = {**model_config, "heads": {"semantic": True, "affinity": False, "direction": False}}
    model = SegOCRUNet(config)
    x = torch.randn(1, 3, 64, 64)
    out = model(x)
    assert "semantic" in out
    assert "affinity" not in out
    assert "direction" not in out


def test_forward_supports_backprop(model_config: dict) -> None:
    """Verify the model is fully differentiable end to end."""
    model = SegOCRUNet(model_config)
    x = torch.randn(1, 3, 64, 64, requires_grad=True)
    out = model(x)
    loss = out["semantic"].sum() + out["affinity"].sum() + out["direction"].sum()
    loss.backward()
    # Some encoder param should have a gradient
    grads = [p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()]
    assert any(grads)


def test_build_model_dispatches_to_unet(model_config: dict) -> None:
    model = build_model(model_config)
    assert isinstance(model, SegOCRUNet)


def test_build_model_unknown_arch_raises() -> None:
    with pytest.raises(ValueError, match="Unknown architecture"):
        build_model({"architecture": "nonexistent", "num_classes": 8})
