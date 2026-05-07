"""CycleGAN-based synthetic→real style transfer.

Research Proposal §6.2 Option A. Train a CycleGAN between the synthetic
domain and real text images (ICDAR/COCO-Text), then apply style transfer
to ~50% of synthetic images during training. The class mask is unchanged.
~1–2 days on a single GPU.
"""
from __future__ import annotations

import torch
from torch import nn


class CycleGANAdapter:
    """Wraps a trained CycleGAN G_synth→real generator for inference-time
    style transfer applied during training data loading."""

    def __init__(self, generator_ckpt: str, device: torch.device | str = "cuda") -> None:
        self.generator_ckpt = generator_ckpt
        self.device = device
        self.generator: nn.Module | None = None
        raise NotImplementedError("CycleGANAdapter.__init__ — Week 9")

    @torch.no_grad()
    def stylize(self, image: torch.Tensor) -> torch.Tensor:
        """Run G_synth→real on a (B, 3, H, W) batch."""
        raise NotImplementedError("CycleGANAdapter.stylize — Week 9")
