"""Generator orchestrator.

Implementation Guide §3.8. Wires the seven sub-components together and
drives parallel dataset generation via multiprocessing.

Per-image contract:
    image           (H, W, 3)    uint8 RGB
    semantic_mask   (H, W)       uint8 class IDs (0 = bg, 1..62 chars)
    instance_mask   (H, W)       uint16 unique character instance IDs
    affinity_mask   (H, W)       uint16 word-group IDs
    direction_field (H, W, 2)    float32 unit-vector to character centroid
    metadata        dict         per-character + generation params

Disk layout written by ``generate_dataset``:
    output_dir/
    ├── images/      XXXXXX.png      lossless RGB
    ├── semantic/    XXXXXX.png      uint8 single-channel class IDs
    ├── instance/    XXXXXX.png      uint16 single-channel instance IDs
    └── metadata/    XXXXXX.json     per-char metadata + generation params

Note: affinity and direction are NOT written to disk — they're cheap to
recompute from (semantic, metadata) at training time, and the disk cost
of saving a float32 (H, W, 2) per sample dominates everything else.
"""
from __future__ import annotations

import json
import logging
import multiprocessing as mp
import os
import random
import time
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
from tqdm import tqdm

from segocr.generator.background import BackgroundGenerator
from segocr.generator.compositor import Compositor
from segocr.generator.degradation import DegradationPipeline
from segocr.generator.font_manager import FontManager
from segocr.generator.layout import LayoutEngine
from segocr.generator.renderer import CharacterRenderer
from segocr.generator.targets import (
    build_affinity_mask,
    build_direction_field,
    build_instance_mask,
)
from segocr.generator.text_sampler import TextSampler
from segocr.utils.config import load_config

logger = logging.getLogger(__name__)

# Per-worker engine instance, populated by _worker_init for multiprocessing.
_worker_engine: GeneratorEngine | None = None
_worker_save_root: Path | None = None
_worker_mode: str = "ocr"


class GeneratorEngine:
    """Main orchestrator for synthetic data generation."""

    def __init__(self, config_path: str | Path) -> None:
        self.config_path = Path(config_path)
        self.config: dict = load_config(self.config_path)

        gen_cfg = self.config["generator"]
        self.image_size: tuple[int, int] = tuple(gen_cfg["image_size"])
        self.tier: int = int(gen_cfg.get("character_set", {}).get("tier", 1))

        self.font_manager = FontManager(gen_cfg["fonts"])
        self.text_sampler = TextSampler(gen_cfg["text"])
        self.renderer = CharacterRenderer(
            config=gen_cfg, font_manager=self.font_manager, tier=self.tier
        )
        self.layout = LayoutEngine(gen_cfg["layout"])
        self.background = BackgroundGenerator(gen_cfg["background"])
        self.compositor = Compositor(gen_cfg["compositing"])
        self.degradation = DegradationPipeline(gen_cfg["degradation"])

    # ── Per-sample generation ──────────────────────────────────────────────

    def generate_one(
        self,
        index: int,
        mode: Literal["ocr", "noise_removal"] = "ocr",
    ) -> dict[str, Any]:
        """Generate a single training sample.

        Args:
            index: sample index. Used to seed random / np.random
                   deterministically so the same index always produces the
                   same image.
            mode:  ``"ocr"`` returns multi-class semantic mask (default).
                   ``"noise_removal"`` collapses classes 1..N → 1 for the
                   binary char-vs-background formulation.
        """
        _seed_for_index(index)

        layout_mode = self.layout.sample_mode()
        text_rgba, text_mask, metadata = self._render_text_for_layout(layout_mode)

        bg = self.background.generate(self.image_size)
        composited, semantic_mask = self.compositor.composite(text_rgba, text_mask, bg)
        composited = self.degradation.apply(composited)

        instance_mask = build_instance_mask(semantic_mask, metadata)
        affinity_mask = build_affinity_mask(semantic_mask, metadata)
        direction_field = build_direction_field(instance_mask, metadata)

        if mode == "noise_removal":
            semantic_mask = (semantic_mask > 0).astype(np.uint8)

        # Update text-frequency tracking for the rare-char boost
        text_chars = "".join(m["char"] for m in metadata)
        if text_chars:
            self.text_sampler.update_counts(text_chars)

        return {
            "index": index,
            "mode": mode,
            "image": composited,
            "semantic_mask": semantic_mask,
            "instance_mask": instance_mask,
            "affinity_mask": affinity_mask,
            "direction_field": direction_field,
            "metadata": {
                "characters": metadata,
                "layout_mode": layout_mode,
                "image_size": self.image_size,
                "tier": self.tier,
            },
        }

    # ── Dataset-scale generation ────────────────────────────────────────────

    def generate_dataset(
        self,
        num_images: int,
        output_dir: str | Path,
        mode: Literal["ocr", "noise_removal"] = "ocr",
        num_workers: int | None = None,
    ) -> None:
        """Generate ``num_images`` samples and write to ``output_dir``."""
        output_root = Path(output_dir)
        for sub in ("images", "semantic", "instance", "metadata"):
            (output_root / sub).mkdir(parents=True, exist_ok=True)

        if num_workers is None:
            num_workers = int(self.config["generator"].get("num_workers", 0))

        t0 = time.perf_counter()
        if num_workers <= 1:
            for index in tqdm(range(num_images), desc="generating"):
                sample = self.generate_one(index, mode=mode)
                _save_sample(sample, output_root)
        else:
            self._generate_parallel(num_images, output_root, mode, num_workers)
        elapsed = time.perf_counter() - t0
        logger.info(
            "Generated %d samples in %.1fs (%.1f img/s)",
            num_images,
            elapsed,
            num_images / max(1e-3, elapsed),
        )

    def _generate_parallel(
        self,
        num_images: int,
        output_root: Path,
        mode: str,
        num_workers: int,
    ) -> None:
        ctx = mp.get_context("spawn")
        with ctx.Pool(
            processes=num_workers,
            initializer=_worker_init,
            initargs=(str(self.config_path), str(output_root), mode),
        ) as pool:
            for _ in tqdm(
                pool.imap_unordered(_worker_generate, range(num_images), chunksize=4),
                total=num_images,
                desc="generating",
            ):
                pass

    # ── Internal ────────────────────────────────────────────────────────────

    def _render_text_for_layout(
        self, layout_mode: str
    ) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
        font, _ = self.font_manager.sample_font()
        size = self._sample_font_size_from_font(font)

        if layout_mode == "paragraph":
            lines = self.text_sampler.sample_paragraph()
            if not lines:
                lines = [self.text_sampler.sample_text()]
            rendered_lines = [
                self.renderer.render_text(line, font, size=size) for line in lines
            ]
            return self.layout.apply_paragraph(rendered_lines, self.image_size)

        text = self.text_sampler.sample_text()
        text_rgba, text_mask, metadata = self.renderer.render_text(
            text, font, size=size
        )
        return self.layout.apply_layout(
            text_rgba, text_mask, metadata, self.image_size, mode=layout_mode
        )

    def _sample_font_size_from_font(self, font) -> int:
        # Fonts come pre-sized from the FontManager — use their existing size
        # if available; otherwise fall back to a uniform-random pick.
        try:
            return int(font.size)
        except AttributeError:
            cfg = self.config["generator"]["fonts"]
            return random.randint(int(cfg["min_size"]), int(cfg["max_size"]))


# ── Multiprocessing worker functions (must be top-level for spawn) ─────────


def _worker_init(config_path: str, output_root: str, mode: str) -> None:
    global _worker_engine, _worker_save_root, _worker_mode
    # Quiet down sub-process logging unless the user explicitly enabled it.
    logging.basicConfig(level=os.environ.get("SEGOCR_LOG_LEVEL", "WARNING"))
    _worker_engine = GeneratorEngine(config_path)
    _worker_save_root = Path(output_root)
    _worker_mode = mode


def _worker_generate(index: int) -> int:
    assert _worker_engine is not None and _worker_save_root is not None
    sample = _worker_engine.generate_one(index, mode=_worker_mode)
    _save_sample(sample, _worker_save_root)
    return index


# ── Saving ──────────────────────────────────────────────────────────────────


def _save_sample(sample: dict[str, Any], output_root: Path) -> None:
    """Write one sample to disk under the standard layout."""
    index = sample["index"]
    name = f"{index:06d}"

    image = sample["image"]
    if image.ndim == 3 and image.shape[2] == 3:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        image_bgr = image
    cv2.imwrite(str(output_root / "images" / f"{name}.png"), image_bgr)
    cv2.imwrite(
        str(output_root / "semantic" / f"{name}.png"),
        sample["semantic_mask"].astype(np.uint8),
    )
    cv2.imwrite(
        str(output_root / "instance" / f"{name}.png"),
        sample["instance_mask"].astype(np.uint16),
    )

    meta_out = {
        **sample["metadata"],
        "index": index,
        "mode": sample["mode"],
    }
    with open(output_root / "metadata" / f"{name}.json", "w", encoding="utf-8") as f:
        json.dump(meta_out, f, default=_json_default)


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, tuple):
        return list(o)
    raise TypeError(f"Not JSON serializable: {type(o)}")


def _seed_for_index(index: int) -> None:
    """Make per-sample generation deterministic given the index."""
    random.seed(index)
    np.random.seed(index % (2**32))
