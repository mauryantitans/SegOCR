"""Output heads for the multi-task segmentation model.

Implementation Guide §3.9. Three heads on a shared decoder output:
    Semantic   — H×W×(N+1) class logits
    Affinity   — H×W×1     inter-character word linkage
    Direction  — H×W×2     unit vector to character centroid

The Direction Head is critical for handling rotation-symmetric character
ambiguity (M↔W, 6↔9, O↔0): pixels' direction-to-centroid vectors encode
upright orientation, letting the loss disambiguate a rotated 'M' from a
naturally-oriented 'W'.
"""
from __future__ import annotations

import torch
from torch import nn


class SemanticHead(nn.Module):
    """1×1 conv → (N+1) class logits."""

    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class AffinityHead(nn.Module):
    """1×1 conv → single-channel logit; sigmoid-decoded for word grouping."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DirectionHead(nn.Module):
    """1×1 conv → 2-channel (dx, dy) regression to the character centroid.

    Output is regressed in pixel-space units; loss applies only to
    foreground (non-background) pixels.
    """

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 2, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)
