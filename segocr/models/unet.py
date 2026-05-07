"""UNet + ResNet-50 prototype model (Week 5 baseline).

Implementation Guide §3.9 + Research Proposal §5.1: start with UNet for
rapid prototyping, switch to SegFormer for final results. UNet ships via
``segmentation_models_pytorch`` so it works without mmsegmentation.
"""
from __future__ import annotations

import torch
from torch import nn


class SegOCRUNet(nn.Module):
    """UNet baseline with the same multi-head output as SegOCRModel."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.config = config
        self.num_classes = config["num_classes"]
        # self.backbone = smp.Unet(encoder_name="resnet50", encoder_weights="imagenet",
        #                          in_channels=3, classes=decoder_channels)
        # self.semantic_head, self.affinity_head, self.direction_head
        raise NotImplementedError("SegOCRUNet.__init__ — Week 5")

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        raise NotImplementedError("SegOCRUNet.forward — Week 5")
