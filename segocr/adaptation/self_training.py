"""Self-training / pseudo-labeling on unlabeled real data.

Research Proposal §6.3. Highest-impact single adaptation technique:
expect +5–15% cumulative improvement over 2–3 iterations.

Loop:
    1. Run inference on N unlabeled real images.
    2. Filter: pixel softmax > 0.9, instance >70% confident pixels,
              image >50% confident coverage.
    3. Use filtered predictions as pseudo-labels.
    4. Fine-tune on 70% synthetic + 30% pseudo-labeled real.
    5. Repeat 2–3 iterations.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


class SelfTrainer:
    """Drives the self-training loop end-to-end."""

    def __init__(self, config: dict) -> None:
        self.config = config
        self.confidence_threshold: float = config["confidence_threshold"]
        self.instance_coverage_threshold: float = config["instance_coverage_threshold"]
        self.image_coverage_threshold: float = config["image_coverage_threshold"]
        raise NotImplementedError("SelfTrainer.__init__ — Week 9")

    @torch.no_grad()
    def generate_pseudo_labels(
        self,
        model: nn.Module,
        unlabeled_dir: str | Path,
        output_dir: str | Path,
    ) -> int:
        """Run inference on unlabeled real images, write filtered pseudo-labels
        to ``output_dir``. Returns count of accepted images."""
        raise NotImplementedError("SelfTrainer.generate_pseudo_labels — Week 9")

    def run(self, iterations: int = 3) -> None:
        """Run the full self-training loop."""
        raise NotImplementedError("SelfTrainer.run — Week 9")
