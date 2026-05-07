"""Training loop.

Implementation Guide §3.10. AdamW + polynomial LR decay + 1500-iter warmup.
Mixed precision via torch.cuda.amp. Logs per-class IoU at every eval step
(not just mean — Implementation Guide §6 gotcha #7) so we can spot rare-
character regression early.

Mitigations for the across-epoch instability seen in the pilot:
  - EMA of weights via torch.optim.swa_utils.AveragedModel (default on)
  - Top-3 checkpoint averaging at end of training (default on)
"""
from __future__ import annotations

from pathlib import Path


def train(config_path: str | Path) -> None:
    """Train SegOCR end-to-end.

    Steps:
        1. Load config.
        2. Create train/val DataLoaders.
        3. Build model + EMA wrapper.
        4. Build optimizer (AdamW) + scheduler (PolyLR) + criterion.
        5. wandb.init().
        6. Iteration loop with periodic eval, save, EMA update.
        7. End-of-training: average top-N checkpoints by val mIoU.
    """
    raise NotImplementedError("train — Week 5")


def evaluate(model, val_loader, device) -> dict[str, float]:
    """Run evaluation. Returns metrics dict with at minimum
    'miou', 'fg_miou', 'binary_miou' + per-class 'iou_<char>'."""
    raise NotImplementedError("evaluate — Week 5")


def save_checkpoint(model, optimizer, iteration: int, path: str | Path) -> None:
    raise NotImplementedError("save_checkpoint — Week 5")


def average_checkpoints(checkpoint_paths: list[Path], output_path: Path) -> None:
    """Average state_dicts of the top-N checkpoints. Used at end of training
    to mitigate epoch-to-epoch quality jitter."""
    raise NotImplementedError("average_checkpoints — Week 5")
