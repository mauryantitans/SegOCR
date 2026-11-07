"""PyTorch Dataset for the synthetic SegOCR dataset.

Implementation Guide §3.10. Loads the on-disk format produced by
``GeneratorEngine.generate_dataset``:

    output_dir/
    ├── images/      XXXXXX.png
    ├── semantic/    XXXXXX.png   (uint8 class IDs)
    ├── instance/    XXXXXX.png   (uint16 instance IDs)
    └── metadata/    XXXXXX.json  (per-character metadata)

Affinity and direction targets are recomputed on the fly from
(semantic, instance, metadata) — saves ~1MB of disk per sample at the
cost of ~1ms CPU per __getitem__.

Augmentation pipeline (training only): optional Fourier Domain Adaptation
toward a pool of real images, random horizontal flip, color jitter,
random grayscale. Mask is flipped consistently with the image;
no other geometric augmentations (the generator already produces
diverse layouts).
"""
from __future__ import annotations

import json
import logging
import zlib
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from segocr.adaptation.fda import fourier_domain_adaptation
from segocr.generator.targets import build_affinity_mask, build_direction_field

logger = logging.getLogger(__name__)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
REAL_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp")


class SegOCRDataset(Dataset):
    """Loads generated synthetic data from disk.

    Args:
        data_dir:       Directory written by ``generate_dataset``.
        split:          "train" | "val". Uses a deterministic 95/5 split
                        based on filename hash so train/val are disjoint
                        across runs.
        train_aug:      If True, apply training-time augmentation.
        return_targets: Which targets to compute. ``"all"`` builds
                        semantic + affinity + direction; subset names
                        save CPU when only some heads are trained.
        fda_target_dir: Optional directory of (unlabeled) real images. When
                        set, training samples are FDA-stylized toward a
                        random real image with ``fda_probability``.
        fda_probability: Per-sample probability of applying FDA.
        fda_beta:       FDA low-frequency window size (fraction of image).
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        train_aug: bool = True,
        return_targets: tuple[str, ...] = ("semantic", "affinity", "direction"),
        val_fraction: float = 0.05,
        fda_target_dir: str | Path | None = None,
        fda_probability: float = 0.5,
        fda_beta: float = 0.01,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.split = split
        self.train_aug = train_aug and split == "train"
        self.return_targets = tuple(return_targets)

        self.image_paths = sorted((self.data_dir / "images").glob("*.png"))
        if not self.image_paths:
            raise FileNotFoundError(
                f"No images found under {self.data_dir / 'images'}. "
                "Did you run scripts/generate_dataset.py?"
            )

        # Deterministic split: hash filename → bucket
        self.image_paths = [
            p for p in self.image_paths if self._in_split(p.stem, val_fraction)
        ]
        logger.info("SegOCRDataset[%s]: %d samples", split, len(self.image_paths))

        self.fda_probability = float(fda_probability)
        self.fda_beta = float(fda_beta)
        self.fda_target_paths: list[Path] = []
        if fda_target_dir is not None and self.train_aug:
            self.fda_target_paths = sorted(
                p for p in Path(fda_target_dir).rglob("*")
                if p.suffix.lower() in REAL_IMAGE_SUFFIXES
            )
            if not self.fda_target_paths:
                raise FileNotFoundError(f"No FDA target images found under {fda_target_dir}")
            logger.info("FDA enabled: %d real target images", len(self.fda_target_paths))

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | dict]:
        img_path = self.image_paths[idx]
        name = img_path.stem

        image = self._load_image(img_path)
        semantic = self._load_mask(self.data_dir / "semantic" / f"{name}.png", np.uint8)
        instance = self._load_mask(
            self.data_dir / "instance" / f"{name}.png", np.uint16
        )
        with open(self.data_dir / "metadata" / f"{name}.json", encoding="utf-8") as f:
            metadata = json.load(f)
        char_metadata = metadata.get("characters", [])

        if self.train_aug and self.fda_target_paths:
            rng = np.random.default_rng()
            if rng.random() < self.fda_probability:
                target = self._load_image(
                    self.fda_target_paths[rng.integers(len(self.fda_target_paths))]
                )
                image = fourier_domain_adaptation(image, target, beta=self.fda_beta)

        if self.train_aug:
            image, semantic, instance = self._apply_augmentations(
                image, semantic, instance
            )

        targets: dict[str, torch.Tensor] = {}
        if "semantic" in self.return_targets:
            targets["semantic"] = torch.from_numpy(semantic.astype(np.int64))
        if "affinity" in self.return_targets:
            affinity = build_affinity_mask(semantic, char_metadata)
            targets["affinity"] = torch.from_numpy(affinity.astype(np.int64))
        if "direction" in self.return_targets:
            direction = build_direction_field(instance, char_metadata)
            # (H, W, 2) → (2, H, W)
            targets["direction"] = torch.from_numpy(direction).permute(2, 0, 1).float()

        return {
            "image": self._to_tensor_normalized(image),
            "targets": targets,
            "metadata": {"index": metadata.get("index", idx), "name": name},
        }

    # ── Internal ────────────────────────────────────────────────────────────

    def _in_split(self, stem: str, val_fraction: float) -> bool:
        # zlib.crc32, not hash(): str hashing is randomized per process, which
        # would reshuffle train/val on every run (and leak on --resume).
        bucket = (zlib.crc32(stem.encode("utf-8")) % 1000) / 1000.0
        is_val = bucket < val_fraction
        return is_val if self.split == "val" else not is_val

    def _load_image(self, path: Path) -> np.ndarray:
        img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise OSError(f"Failed to read image {path}")
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    def _load_mask(self, path: Path, dtype: type) -> np.ndarray:
        flag = cv2.IMREAD_GRAYSCALE if dtype is np.uint8 else cv2.IMREAD_UNCHANGED
        mask = cv2.imread(str(path), flag)
        if mask is None:
            raise OSError(f"Failed to read mask {path}")
        return mask.astype(dtype)

    def _to_tensor_normalized(self, image: np.ndarray) -> torch.Tensor:
        arr = image.astype(np.float32) / 255.0
        for c in range(3):
            arr[..., c] = (arr[..., c] - IMAGENET_MEAN[c]) / IMAGENET_STD[c]
        return torch.from_numpy(arr).permute(2, 0, 1).float()

    def _apply_augmentations(
        self,
        image: np.ndarray,
        semantic: np.ndarray,
        instance: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply training augmentations consistently to image and masks."""
        rng = np.random.default_rng()

        # Random horizontal flip (image + mask)
        if rng.random() < 0.5:
            image = image[:, ::-1].copy()
            semantic = semantic[:, ::-1].copy()
            instance = instance[:, ::-1].copy()

        # Color jitter (image only)
        if rng.random() < 0.8:
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.int32)
            hsv[..., 0] = (hsv[..., 0] + rng.integers(-15, 16)) % 180
            hsv[..., 1] = np.clip(hsv[..., 1] * rng.uniform(0.7, 1.3), 0, 255)
            hsv[..., 2] = np.clip(hsv[..., 2] * rng.uniform(0.7, 1.3), 0, 255)
            image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        # Random grayscale
        if rng.random() < 0.1:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            image = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

        return image, semantic, instance


def collate_fn(batch: list[dict]) -> dict:
    """Custom collate that stacks images + targets and lists metadata."""
    images = torch.stack([b["image"] for b in batch])
    targets: dict[str, torch.Tensor] = {}
    if batch:
        for key in batch[0]["targets"]:
            targets[key] = torch.stack([b["targets"][key] for b in batch])
    metadata = [b["metadata"] for b in batch]
    return {"image": images, "targets": targets, "metadata": metadata}
