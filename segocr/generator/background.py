"""4-tier background generator.

Implementation Guide §3.5 + Research Proposal §4.5.

Tier 1 (10%) — solid colors and gradients. Pure compute.
Tier 2 (20%) — procedural textures: low-res Perlin grid upsampled with
                bilinear, optional colormap remapping. Pure compute.
Tier 3 (50%) — random crops from natural-image directories (COCO, DTD,
                Places365). Bottleneck for generation speed; we keep an
                in-memory preload buffer so disk I/O happens once per
                ``buffer_size`` samples instead of per-sample.
Tier 4 (20%) — adversarial: documents with other text, faintly-rendered
                text-like patterns, color-matched Tier 3, dense clutter.

If a tier's data dependencies are missing (e.g. no COCO images on disk),
the tier silently falls back to the most-similar available tier so the
generator keeps producing samples. This makes the module usable before
``setup_data.ps1`` has finished.
"""
from __future__ import annotations

import logging
import random
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


class BackgroundGenerator:
    """Generates backgrounds at 4 complexity tiers."""

    def __init__(self, config: dict) -> None:
        self.config = config
        self.tier_probs: dict[str, float] = dict(config["tier_distribution"])
        self.preload_buffer_size: int = int(config.get("preload_buffer_size", 200))

        self.natural_image_paths: list[Path] = []
        for root in config.get("natural_image_dirs", []):
            root_path = Path(root).expanduser()
            if root_path.exists():
                for ext in IMAGE_EXTENSIONS:
                    self.natural_image_paths.extend(root_path.rglob(f"*{ext}"))
        self.natural_image_paths.sort()

        if not self.natural_image_paths:
            logger.info(
                "No natural-image dirs found (looked in %s). "
                "Tier 3 / 4 will fall back to procedural.",
                config.get("natural_image_dirs", []),
            )

        self.preload_buffer: list[np.ndarray] = []
        self._buffer_index: int = 0

    # ── Public API ──────────────────────────────────────────────────────────

    def generate(self, size: tuple[int, int]) -> np.ndarray:
        """Generate a background sized (H, W). Returns (H, W, 3) uint8 RGB."""
        tier = self._sample_tier()
        try:
            if tier == "tier1_solid":
                return self._tier1_solid(size)
            if tier == "tier2_procedural":
                return self._tier2_procedural(size)
            if tier == "tier3_natural":
                return self._tier3_natural(size)
            return self._tier4_adversarial(size)
        except Exception as exc:  # noqa: BLE001 — never let a tier break dataset gen
            logger.warning("Tier %s failed (%s); falling back to Tier 1.", tier, exc)
            return self._tier1_solid(size)

    # ── Tiers ───────────────────────────────────────────────────────────────

    def _tier1_solid(self, size: tuple[int, int]) -> np.ndarray:
        h, w = size
        kind = random.choice(("solid", "linear_gradient", "radial_gradient"))
        if kind == "solid":
            color = _random_color()
            return np.full((h, w, 3), color, dtype=np.uint8)

        c1 = np.array(_random_color(), dtype=np.float32)
        c2 = np.array(_random_color(), dtype=np.float32)

        if kind == "linear_gradient":
            angle = random.uniform(0, 2 * np.pi)
            ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
            t = (xs * np.cos(angle) + ys * np.sin(angle)).astype(np.float32)
            t = (t - t.min()) / max(1.0, t.max() - t.min())
            t = t[..., None]
            arr = c1 * (1 - t) + c2 * t
            return np.clip(arr, 0, 255).astype(np.uint8)

        # radial_gradient
        cx = random.randint(0, w - 1)
        cy = random.randint(0, h - 1)
        ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        d = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
        t = (d / max(1.0, d.max()))[..., None]
        arr = c1 * (1 - t) + c2 * t
        return np.clip(arr, 0, 255).astype(np.uint8)

    def _tier2_procedural(self, size: tuple[int, int]) -> np.ndarray:
        """Multi-octave low-frequency noise.

        Implementation: generate small random grids at octaves and bilinearly
        upsample-and-add. Faster and visually equivalent to per-pixel Perlin
        for our purposes. ``noise.pnoise2`` is available but Python-level
        looping over all pixels would dominate generation time.
        """
        h, w = size
        n_octaves = random.randint(1, 4)
        accum = np.zeros((h, w), dtype=np.float32)
        amplitude = 1.0
        norm = 0.0
        for octave in range(n_octaves):
            grid_size = 4 * (2 ** octave)
            grid = np.random.rand(grid_size, grid_size).astype(np.float32)
            upsampled = cv2.resize(grid, (w, h), interpolation=cv2.INTER_CUBIC)
            accum += amplitude * upsampled
            norm += amplitude
            amplitude *= 0.5
        accum /= max(1e-6, norm)

        # Apply a random colormap with 50% probability; otherwise use as
        # a 3-channel grayscale bias around a random base color.
        if random.random() < 0.5:
            cmap_id = random.choice(
                [
                    cv2.COLORMAP_VIRIDIS,
                    cv2.COLORMAP_INFERNO,
                    cv2.COLORMAP_PLASMA,
                    cv2.COLORMAP_TWILIGHT,
                    cv2.COLORMAP_TURBO,
                    cv2.COLORMAP_OCEAN,
                ]
            )
            gray = (accum * 255).astype(np.uint8)
            colored_bgr = cv2.applyColorMap(gray, cmap_id)
            return cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)

        base = np.array(_random_color(), dtype=np.float32)
        accent = np.array(_random_color(), dtype=np.float32)
        t = accum[..., None]
        arr = base * (1 - t) + accent * t
        return np.clip(arr, 0, 255).astype(np.uint8)

    def _tier3_natural(self, size: tuple[int, int]) -> np.ndarray:
        if not self.natural_image_paths:
            return self._tier2_procedural(size)
        img = self._next_buffer_image()
        if img is None:
            return self._tier2_procedural(size)
        return _random_crop_or_scale(img, size)

    def _tier4_adversarial(self, size: tuple[int, int]) -> np.ndarray:
        kind = random.choice(("text_like", "color_matched", "dense_clutter"))

        if kind == "text_like":
            base = self._tier2_procedural(size)
            return _add_text_like_pattern(base)

        if kind == "color_matched" and self.natural_image_paths:
            tier3 = self._tier3_natural(size)
            target_hue = random.randint(0, 179)
            return _shift_hue(tier3, target_hue, saturation_scale=0.6)

        # dense_clutter / fallback
        return self._tier3_natural(size)

    # ── Internal helpers ────────────────────────────────────────────────────

    def _sample_tier(self) -> str:
        tiers = list(self.tier_probs.keys())
        weights = list(self.tier_probs.values())
        return random.choices(tiers, weights=weights, k=1)[0]

    def _next_buffer_image(self) -> np.ndarray | None:
        """Return a freshly-loaded image from the preload buffer.

        Refreshes the buffer when exhausted so disk I/O is amortized over
        ``preload_buffer_size`` samples.
        """
        if not self.preload_buffer or self._buffer_index >= len(self.preload_buffer):
            self._refresh_preload_buffer()
        if not self.preload_buffer:
            return None
        img = self.preload_buffer[self._buffer_index]
        self._buffer_index += 1
        return img

    def _refresh_preload_buffer(self) -> None:
        if not self.natural_image_paths:
            self.preload_buffer = []
            self._buffer_index = 0
            return
        n = min(self.preload_buffer_size, len(self.natural_image_paths))
        sampled = random.sample(self.natural_image_paths, n)
        loaded: list[np.ndarray] = []
        for path in sampled:
            try:
                img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if img_bgr is None:
                    continue
                loaded.append(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
            except Exception as exc:  # noqa: BLE001
                logger.debug("Failed to load %s: %s", path, exc)
        self.preload_buffer = loaded
        self._buffer_index = 0


# ── Module-level helpers ────────────────────────────────────────────────────


def _random_color() -> tuple[int, int, int]:
    return (
        random.randint(0, 255),
        random.randint(0, 255),
        random.randint(0, 255),
    )


def _random_crop_or_scale(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Random-crop ``img`` to ``size`` if it's larger; else upscale-then-crop.

    Always returns (H, W, 3) uint8 RGB regardless of input dimensions.
    """
    h, w = size
    src_h, src_w = img.shape[:2]
    scale = max(h / src_h, w / src_w, 1.0)
    if scale > 1.0:
        new_w = int(np.ceil(src_w * scale))
        new_h = int(np.ceil(src_h * scale))
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        src_h, src_w = img.shape[:2]
    y_max = max(0, src_h - h)
    x_max = max(0, src_w - w)
    y0 = random.randint(0, y_max) if y_max > 0 else 0
    x0 = random.randint(0, x_max) if x_max > 0 else 0
    crop = img[y0 : y0 + h, x0 : x0 + w]
    if random.random() < 0.5:
        crop = crop[:, ::-1]  # horizontal flip
    return np.ascontiguousarray(crop)


def _add_text_like_pattern(base: np.ndarray, density: float = 0.05) -> np.ndarray:
    """Overlay faintly-rendered random characters as adversarial texture.

    Uses Pillow's default bitmap font (no font dependency required) and
    composites at low alpha so the model learns to ignore text-like
    structure that isn't real text.
    """
    from PIL import Image, ImageDraw, ImageFont

    h, w, _ = base.shape
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    try:
        font = ImageFont.load_default()
    except OSError:
        return base

    n_chars = int(w * h * density / 200)
    chars = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    for _ in range(n_chars):
        char = random.choice(chars)
        x = random.randint(0, w)
        y = random.randint(0, h)
        color = (*_random_color(), random.randint(40, 110))
        draw.text((x, y), char, font=font, fill=color)

    base_pil = Image.fromarray(base).convert("RGBA")
    composed = Image.alpha_composite(base_pil, overlay).convert("RGB")
    return np.array(composed)


def _shift_hue(img: np.ndarray, target_hue: int, saturation_scale: float = 1.0) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.int32)
    hsv[..., 0] = target_hue
    hsv[..., 1] = np.clip(hsv[..., 1] * saturation_scale, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
