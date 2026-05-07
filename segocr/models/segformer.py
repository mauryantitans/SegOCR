"""SegFormer-based multi-head model — the primary architecture.

Implementation Guide §3.9. Wraps an MMSegmentation MiT-B2 backbone +
decoder, attaches the semantic / affinity / direction heads from
``segocr.models.heads``.

NOTE: mmsegmentation/mmcv-full are not installed by default in the Phase 1
environment (Windows builds are fragile). Install them when SegFormer is
needed (Week 6+) following requirements/train.txt instructions.
"""
from __future__ import annotations

import torch
from torch import nn


class SegOCRModel(nn.Module):
    """Multi-head SegFormer for character-level OCR-as-segmentation.

    forward() returns a dict with keys ``semantic``, optionally
    ``affinity``, optionally ``direction`` — head presence is determined
    by config["heads"].
    """

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.config = config
        self.num_classes = config["num_classes"]
        # self.backbone = ...  # MiT-B2 encoder + lightweight decoder
        # self.semantic_head, self.affinity_head, self.direction_head
        raise NotImplementedError("SegOCRModel.__init__ — Week 6")

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        raise NotImplementedError("SegOCRModel.forward — Week 6")
