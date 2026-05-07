"""Domain-Adversarial Neural Network adaptation.

Research Proposal §6.4. A domain discriminator is attached to the encoder
output through a Gradient Reversal Layer (GRL); the encoder is trained to
fool it, forcing domain-invariant features. Mix synthetic and real images
50/50 per batch. Lambda ramps 0→1 over ``lambda_rampup_iters``.

Expected gain: +3–7%, complementary to CycleGAN/self-training.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.autograd import Function


class GradientReversalFn(Function):
    """Identity forward, scaled-negate backward."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        return -ctx.lambda_ * grad_output, None


def grad_reverse(x: torch.Tensor, lambda_: float = 1.0) -> torch.Tensor:
    return GradientReversalFn.apply(x, lambda_)


class DomainDiscriminator(nn.Module):
    """N-layer MLP head over global-pooled encoder features → 2 classes
    (synthetic / real)."""

    def __init__(self, in_channels: int, hidden_dim: int = 512, num_layers: int = 3) -> None:
        super().__init__()
        raise NotImplementedError("DomainDiscriminator.__init__ — Week 10")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("DomainDiscriminator.forward — Week 10")


class DANNTrainer:
    """Wires a DomainDiscriminator + GRL into the main training loop."""

    def __init__(self, config: dict) -> None:
        self.config = config
        self.lambda_max: float = config["lambda_max"]
        self.lambda_rampup_iters: int = config["lambda_rampup_iters"]
        raise NotImplementedError("DANNTrainer.__init__ — Week 10")

    def current_lambda(self, iteration: int) -> float:
        """Linear ramp 0 → lambda_max over rampup_iters."""
        return min(1.0, iteration / self.lambda_rampup_iters) * self.lambda_max
