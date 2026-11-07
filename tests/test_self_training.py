"""Self-training (pseudo-labeling) tests."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from torch import nn

from segocr.adaptation.self_training import SelfTrainer
from segocr.generator.engine import GeneratorEngine
from segocr.models.losses import IGNORE_INDEX, SegOCRLoss
from segocr.training.dataset import SegOCRDataset, collate_fn
from tests.test_train_resume import _build_train_config

NUM_CLASSES = 8


def _config(size: int = 32, **st_overrides) -> dict:
    st = {
        "confidence_threshold": 0.9,
        "instance_coverage_threshold": 0.7,
        "image_coverage_threshold": 0.5,
        "real_ratio": 0.3,
        **st_overrides,
    }
    return {"generator": {"image_size": [size, size]}, "adaptation": {"self_training": st}}


def _probs(pred: np.ndarray, conf: np.ndarray) -> np.ndarray:
    """(C, H, W) probs whose argmax is ``pred`` with max-prob ``conf``."""
    rest = (1.0 - conf) / (NUM_CLASSES - 1)
    probs = np.repeat(rest[None], NUM_CLASSES, axis=0)
    np.put_along_axis(probs, pred[None], conf[None], axis=0)
    return probs


def test_filter_keeps_confident_instance_and_ignores_uncertain_one() -> None:
    pred = np.zeros((32, 32), dtype=np.int64)
    conf = np.full((32, 32), 0.99)
    pred[2:10, 2:10] = 3                 # fully confident char → kept
    pred[20:28, 20:28] = 5               # half-confident char → dropped whole
    conf[20:24, 20:28] = 0.6
    conf[0, 30:] = 0.5                   # uncertain background pixels

    semantic, instance, chars = SelfTrainer(_config()).filter_prediction(_probs(pred, conf))

    assert (semantic[2:10, 2:10] == 3).all()
    assert (semantic[20:28, 20:28] == IGNORE_INDEX).all()
    assert (semantic[0, 30:] == IGNORE_INDEX).all()
    assert semantic[15, 15] == 0
    assert [c["class_id"] for c in chars] == [3]
    assert chars[0]["bbox"] == [2, 2, 10, 10]
    assert set(np.unique(instance)) == {0, 1}


def test_filter_rejects_low_coverage_image_and_ignores_padding() -> None:
    pred = np.zeros((32, 32), dtype=np.int64)
    conf = np.full((32, 32), 0.99)
    conf[:20] = 0.5
    trainer = SelfTrainer(_config())
    assert trainer.filter_prediction(_probs(pred, conf)) is None

    # Same image, but the uncertain rows are letterbox padding → accepted.
    valid = np.ones((32, 32), dtype=bool)
    valid[:20] = False
    semantic, _, _ = trainer.filter_prediction(_probs(pred, conf), valid)
    assert (semantic[:20] == IGNORE_INDEX).all()
    assert (semantic[20:] == 0).all()


class _FixedModel(nn.Module):
    """Predicts class 3 with high confidence in the top-left quadrant."""

    def __init__(self) -> None:
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        b, _, h, w = x.shape
        logits = torch.zeros(b, NUM_CLASSES, h, w)
        logits[:, 0] = 10.0
        logits[:, 0, 4 : h // 2, 4 : w // 2] = 0.0
        logits[:, 3, 4 : h // 2, 4 : w // 2] = 10.0
        return {"semantic": logits + self.dummy}


def test_pseudo_labels_round_trip_through_dataset_and_loss(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    cv2.imwrite(str(real / "a.jpg"), np.full((40, 64, 3), 120, np.uint8))  # non-square
    cv2.imwrite(str(real / "b.png"), np.full((32, 32, 3), 60, np.uint8))

    out = tmp_path / "pseudo"
    n = SelfTrainer(_config()).generate_pseudo_labels(_FixedModel(), real, out)
    assert n == 2

    ds = SegOCRDataset(out, split="train", train_aug=False, val_fraction=0.0)
    batch = collate_fn([ds[i] for i in range(len(ds))])
    sem, aff, direction = (batch["targets"][k] for k in ("semantic", "affinity", "direction"))
    assert batch["image"].shape == (2, 3, 32, 32)
    assert (sem == 3).any() and (aff == -1).all()
    assert direction[:, :, sem[0] == 3].abs().sum() > 0
    # Letterboxed 40x64 → 20x32 image + 12 rows of padding, all ignored.
    meta = [m["name"] for m in batch["metadata"]]
    padded = sem[meta.index("000000")]
    assert (padded[20:] == IGNORE_INDEX).all()

    loss_fn = SegOCRLoss({}, num_classes=NUM_CLASSES)
    preds = {
        "semantic": torch.randn(2, NUM_CLASSES, 32, 32),
        "affinity": torch.randn(2, 1, 32, 32),
        "direction": torch.randn(2, 2, 32, 32),
    }
    total, _ = loss_fn(preds, batch["targets"])
    assert torch.isfinite(total)


def test_run_end_to_end_on_cpu(engine_config_path: Path, tmp_path: Path) -> None:
    """One self-training round with a tiny UNet: pseudo-label → fine-tune."""
    cfg_path = _build_train_config(engine_config_path, tmp_path, total_iters=2)
    cfg = yaml.safe_load(cfg_path.read_text())
    GeneratorEngine(cfg_path).generate_dataset(
        num_images=4, output_dir=cfg["generator"]["output_dir"], num_workers=0
    )
    from segocr.training.train import train

    init = train(cfg_path)
    assert init is not None

    real = tmp_path / "real"
    real.mkdir()
    for i in range(3):
        cv2.imwrite(str(real / f"{i}.png"), np.random.default_rng(i).integers(0, 255, (50, 70, 3), dtype=np.uint8))
    # A barely-trained model is never 90% confident; open the gates for the smoke test.
    cfg["adaptation"] = {"self_training": {
        "confidence_threshold": 0.0,
        "instance_coverage_threshold": 0.0,
        "image_coverage_threshold": 0.0,
        "real_ratio": 0.5,
        "finetune_iters": 2,
        "unlabeled_real_dir": str(real),
    }}
    final = SelfTrainer(cfg).run(init, tmp_path / "st", iterations=1)

    assert final.exists() and final.parent == tmp_path / "st" / "round_1" / "weights"
    assert len(list((tmp_path / "st" / "round_1" / "pseudo" / "images").glob("*.png"))) == 3
