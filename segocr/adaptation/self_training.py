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

Pseudo-labels are written in the generator's on-disk layout so
``SegOCRDataset`` loads them unchanged. Unconfident pixels and rejected
instances get ``IGNORE_INDEX`` and contribute no loss; the affinity head
is not supervised on real images (no word grouping), the direction head
is supervised toward each accepted instance's centroid.
"""
from __future__ import annotations

import copy
import json
import logging
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from torch import nn

from segocr.models.losses import IGNORE_INDEX

logger = logging.getLogger(__name__)

REAL_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp")
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class SelfTrainer:
    """Drives the self-training loop end-to-end.

    Args:
        config: the full SegOCR config. Reads ``adaptation.self_training``
            for thresholds / ratios, ``generator.image_size`` for the
            pseudo-label resolution, and the rest for fine-tuning.
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        st = config["adaptation"]["self_training"]
        self.confidence_threshold: float = float(st["confidence_threshold"])
        self.instance_coverage_threshold: float = float(st["instance_coverage_threshold"])
        self.image_coverage_threshold: float = float(st["image_coverage_threshold"])
        self.real_ratio: float = float(st.get("real_ratio", 0.3))
        self.finetune_iters: int = int(st.get("finetune_iters", 20000))
        self.unlabeled_dir: str | None = st.get("unlabeled_real_dir")
        self.image_size: tuple[int, int] = tuple(config["generator"]["image_size"])  # (H, W)

    # ── Filtering ───────────────────────────────────────────────────────────

    def filter_prediction(
        self,
        probs: np.ndarray,
        valid: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, list[dict]] | None:
        """Turn a softmax map into a filtered pseudo-label.

        Args:
            probs: (C, H, W) softmax probabilities.
            valid: optional (H, W) bool mask of real-image pixels (False on
                letterbox padding). Padding is always ignored.

        Returns:
            ``None`` if the image fails the coverage gate, else
            ``(semantic, instance, characters)``: uint8 class map with
            ``IGNORE_INDEX`` on uncertain pixels, uint16 instance map, and
            per-instance metadata (class_id, bbox, centroid).
        """
        pred = probs.argmax(axis=0)
        confident = probs.max(axis=0) >= self.confidence_threshold
        if valid is None:
            valid = np.ones(pred.shape, dtype=bool)
        confident &= valid
        if confident.sum() / max(int(valid.sum()), 1) < self.image_coverage_threshold:
            return None

        semantic = np.where(confident, pred, IGNORE_INDEX).astype(np.uint8)
        instance = np.zeros(pred.shape, dtype=np.uint16)
        characters: list[dict] = []
        for class_id in np.unique(pred[valid & (pred > 0)]):
            class_mask = ((pred == class_id) & valid).astype(np.uint8)
            n_labels, labels = cv2.connectedComponents(class_mask, connectivity=8)
            for i in range(1, n_labels):
                region = labels == i
                if confident[region].mean() < self.instance_coverage_threshold:
                    semantic[region] = IGNORE_INDEX
                    continue
                keep = region & confident
                ys, xs = np.nonzero(keep)
                characters.append({
                    "class_id": int(class_id),
                    "bbox": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
                    "centroid": [float(xs.mean()), float(ys.mean())],
                })
                instance[keep] = len(characters)
        return semantic, instance, characters

    # ── Pseudo-label generation ─────────────────────────────────────────────

    @torch.no_grad()
    def generate_pseudo_labels(
        self,
        model: nn.Module,
        unlabeled_dir: str | Path,
        output_dir: str | Path,
    ) -> int:
        """Run inference on unlabeled real images, write filtered pseudo-labels
        to ``output_dir``. Returns count of accepted images."""
        paths = sorted(
            p for p in Path(unlabeled_dir).rglob("*") if p.suffix.lower() in REAL_IMAGE_SUFFIXES
        )
        if not paths:
            raise FileNotFoundError(f"No unlabeled images found under {unlabeled_dir}")
        out = Path(output_dir)
        for sub in ("images", "semantic", "instance", "metadata"):
            (out / sub).mkdir(parents=True, exist_ok=True)

        model.eval()
        device = next(model.parameters()).device
        accepted = 0
        for path in paths:
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if bgr is None:
                logger.warning("Skipping unreadable image %s", path)
                continue
            image, valid = _letterbox(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), self.image_size)
            x = (image.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
            x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(device)
            probs = torch.softmax(model(x)["semantic"].float(), dim=1)[0].cpu().numpy()

            result = self.filter_prediction(probs, valid)
            if result is None:
                continue
            semantic, instance, characters = result
            name = f"{accepted:06d}"
            cv2.imwrite(str(out / "images" / f"{name}.png"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(out / "semantic" / f"{name}.png"), semantic)
            cv2.imwrite(str(out / "instance" / f"{name}.png"), instance)
            with open(out / "metadata" / f"{name}.json", "w", encoding="utf-8") as f:
                json.dump({
                    "index": accepted,
                    "pseudo_label": True,
                    "source": str(path),
                    "characters": characters,
                }, f)
            accepted += 1
        logger.info("Pseudo-labels: accepted %d / %d images", accepted, len(paths))
        return accepted

    # ── Full loop ───────────────────────────────────────────────────────────

    def run(
        self,
        init_checkpoint: str | Path,
        work_dir: str | Path,
        iterations: int = 3,
        seed: int | None = None,
    ) -> Path:
        """Run the full self-training loop.

        Each round pseudo-labels ``unlabeled_real_dir`` with the current
        weights, then fine-tunes from them for ``finetune_iters`` on
        synthetic + pseudo-labeled data. Returns the final weights path.
        """
        # Imported here: segocr.training imports segocr.adaptation (FDA).
        from segocr.models.unet import build_model
        from segocr.training.train import train

        if not self.unlabeled_dir:
            raise ValueError("adaptation.self_training.unlabeled_real_dir is not set")
        work = Path(work_dir)
        current = Path(init_checkpoint)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for round_idx in range(1, iterations + 1):
            round_dir = work / f"round_{round_idx}"
            model = build_model(self.config["model"]).to(device)
            ckpt = torch.load(current, map_location=device)
            model.load_state_dict(ckpt.get("ema", ckpt["model"]))
            n = self.generate_pseudo_labels(model, self.unlabeled_dir, round_dir / "pseudo")
            del model
            if n == 0:
                logger.warning("Round %d: no image passed the filters; stopping.", round_idx)
                break

            cfg = copy.deepcopy(self.config)
            cfg["training"]["output_dir"] = str(round_dir / "weights")
            cfg["training"]["total_iters"] = self.finetune_iters
            cfg["training"]["pseudo_label_dir"] = str(round_dir / "pseudo")
            cfg["training"]["pseudo_label_ratio"] = self.real_ratio
            cfg_path = round_dir / "config.yaml"
            cfg_path.write_text(yaml.safe_dump(cfg))

            best = train(cfg_path, seed=seed, init_from=current)
            if best is None:
                raise RuntimeError(
                    f"Round {round_idx} produced no checkpoint; make finetune_iters "
                    ">= training.eval_interval"
                )
            logger.info("Round %d done (%d pseudo-labeled images) → %s", round_idx, n, best)
            current = best
        return current


def _letterbox(image: np.ndarray, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Resize keeping aspect ratio, pad bottom/right with black to ``size`` (H, W).

    Returns the padded image and a bool mask of real (non-padding) pixels.
    """
    th, tw = size
    h, w = image.shape[:2]
    scale = min(th / h, tw / w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    out = np.zeros((th, tw, 3), dtype=np.uint8)
    out[:nh, :nw] = cv2.resize(image, (nw, nh), interpolation=interp)
    valid = np.zeros((th, tw), dtype=bool)
    valid[:nh, :nw] = True
    return out, valid
