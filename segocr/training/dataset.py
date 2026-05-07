"""PyTorch Dataset for the synthetic SegOCR dataset.

Implementation Guide §3.10. Loads (image, semantic_mask, instance_mask,
affinity_mask, direction_field) tuples produced by GeneratorEngine and
applies training-time augmentation (color jitter, random flip, etc.).

For 500K+ images, switch to WebDataset (sequential tar reading) or FFCV
to bypass file-system metadata overhead. PNG-per-sample works fine up to
~200K samples.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset


class SegOCRDataset(Dataset):
    """Dataset for loading generated synthetic data."""

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        transforms=None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.split = split
        self.transforms = transforms
        self.image_paths: list[Path] = []
        self.mask_paths: list[Path] = []
        raise NotImplementedError("SegOCRDataset.__init__ — Week 5")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | dict]:
        """
        Returns:
            {
              "image":     (3, H, W)  float32 normalized to ImageNet stats,
              "targets": {
                  "semantic":  (H, W)    long,
                  "affinity":  (H, W)    float,    # optional
                  "direction": (2, H, W) float,    # optional
              },
            }
        """
        raise NotImplementedError("SegOCRDataset.__getitem__ — Week 5")
