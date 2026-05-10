"""UNet + ResNet-50 prototype model.

Implementation Guide §3.9 + Research Proposal §5.1: start with UNet for
rapid prototyping, switch to SegFormer for the final model.

Built on ``segmentation_models_pytorch`` so it doesn't require the
fragile mmsegmentation/mmcv-full stack. The encoder uses ImageNet
pretrained weights by default; swap to ``encoder_weights=None`` for a
from-scratch baseline.

Multi-head design: shared encoder + decoder, then 1×1-conv heads for
semantic / affinity / direction. Direction head outputs are unbounded
(L1-regressed against unit-normalized targets); semantic and affinity
heads output raw logits.
"""
from __future__ import annotations

import segmentation_models_pytorch as smp
import torch
from torch import nn

from segocr.models.heads import AffinityHead, DirectionHead, SemanticHead


class SegOCRUNet(nn.Module):
    """UNet baseline with multi-head output."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.config = config
        self.num_classes = int(config["num_classes"])
        encoder_name = str(config.get("encoder", "resnet50"))
        encoder_weights = config.get("encoder_weights", "imagenet")

        decoder_channels = tuple(
            config.get("decoder_channels", (256, 128, 64, 32, 32))
        )

        # smp.Unet returns a (B, classes, H, W) tensor after its built-in
        # segmentation_head. We use that final head as a feature projection
        # (set to a moderate width) and add our own multi-head modules on top.
        self._head_features = int(config.get("head_features", 32))

        self._unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=self._head_features,
            decoder_channels=decoder_channels,
        )

        head_cfg = config.get("heads", {}) or {}
        self.semantic_head = SemanticHead(self._head_features, self.num_classes)
        self.affinity_head: AffinityHead | None = (
            AffinityHead(self._head_features) if head_cfg.get("affinity") else None
        )
        self.direction_head: DirectionHead | None = (
            DirectionHead(self._head_features) if head_cfg.get("direction") else None
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self._unet(x)  # (B, head_features, H, W)
        out: dict[str, torch.Tensor] = {"semantic": self.semantic_head(features)}
        if self.affinity_head is not None:
            out["affinity"] = self.affinity_head(features)
        if self.direction_head is not None:
            out["direction"] = self.direction_head(features)
        return out


def build_model(config: dict) -> nn.Module:
    """Factory: instantiate the architecture named in ``config["architecture"]``.

    For now only UNet is wired up; SegFormer is gated on the
    mmsegmentation install.
    """
    arch = str(config.get("architecture", "unet")).lower()
    if arch == "unet":
        return SegOCRUNet(config)
    if arch == "segformer":
        from segocr.models.segformer import SegOCRModel

        return SegOCRModel(config)
    raise ValueError(f"Unknown architecture: {arch!r}")
