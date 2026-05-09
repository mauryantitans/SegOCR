"""Degradation pipeline — realistic capture-condition augmentations.

Implementation Guide §3.7 + Research Proposal §4.7.

Applied AFTER compositing. Operates on the RGB image only — the class
mask is never modified by degradations. Callers pass the image; the
mask is held elsewhere and is invariant.

Albumentations covers ~80% of what we need; only the optional local-blur
trick from CharSeg Dataset Brainstorming #10 (SynthText-style) is
custom.

JPEG compression is the single most impactful degradation for real-world
performance — without it the model exploits high-frequency details that
JPEG destroys. Do not lower its probability below ~0.5.
"""
from __future__ import annotations

import logging
import random

import albumentations as A  # noqa: N812 — A is the canonical albumentations alias
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class DegradationPipeline:
    """Image-only degradation pipeline driven by config thresholds."""

    def __init__(self, config: dict) -> None:
        self.config = config
        self.pipeline: A.Compose = self._build_pipeline(config)
        self.shadow_prob: float = float(config.get("lighting", {}).get("probability", 0.0)) * 0.3
        self.occlusion_cfg: dict = dict(config.get("occlusion", {}))

    # ── Public API ──────────────────────────────────────────────────────────

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Run the full pipeline on an (H, W, 3) uint8 RGB image."""
        result = self.pipeline(image=image)["image"]
        if random.random() < self.shadow_prob:
            result = self._apply_random_shadow(result)
        if random.random() < float(self.occlusion_cfg.get("probability", 0.0)):
            result = self._apply_random_occlusion(result)
        return result

    def apply_local_blur(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        bbox: tuple[int, int, int, int],
        kernel: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """SynthText-style local gaussian blur (dataset brainstorming #10).

        Blurs both the image AND a binary mask of the bbox region with the
        same kernel, then re-thresholds the mask so the polygon shape
        reflects the blurred edge. Used for downstream polygon extraction;
        NOT used in the standard OCR class-mask pipeline (blurring class
        IDs creates fractional values — gotcha #1).

        Args:
            image: (H, W, 3) RGB.
            mask:  (H, W) binary mask (0 / 255).
            bbox:  (x0, y0, x1, y1) region to blur.
            kernel: odd gaussian kernel size.

        Returns:
            (blurred_image, blurred_binary_mask).
        """
        if kernel % 2 == 0:
            kernel += 1
        x0, y0, x1, y1 = bbox
        h, w = image.shape[:2]
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, x1), min(h, y1)
        if x1 <= x0 or y1 <= y0:
            return image, mask

        out_image = image.copy()
        out_mask = mask.copy()
        out_image[y0:y1, x0:x1] = cv2.GaussianBlur(
            out_image[y0:y1, x0:x1], (kernel, kernel), 0
        )
        bin_region = (mask[y0:y1, x0:x1] > 0).astype(np.uint8) * 255
        blurred_region = cv2.GaussianBlur(bin_region, (kernel, kernel), 0)
        out_mask[y0:y1, x0:x1] = (blurred_region >= 128).astype(np.uint8) * 255
        return out_image, out_mask

    # ── Internal — pipeline construction ────────────────────────────────────

    def _build_pipeline(self, config: dict) -> A.Compose:
        blur_cfg = config.get("blur", {})
        noise_cfg = config.get("noise", {})
        compression_cfg = config.get("compression", {})
        lighting_cfg = config.get("lighting", {})
        geometric_cfg = config.get("geometric", {})

        transforms: list = []

        if blur_cfg.get("probability", 0) > 0:
            blur_max = int(max(*blur_cfg.get("motion_kernel", (3, 15))))
            blur_max = blur_max + (1 if blur_max % 2 == 0 else 0)
            transforms.append(
                A.OneOf(
                    [
                        A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                        A.MotionBlur(blur_limit=(3, blur_max), p=1.0),
                    ],
                    p=float(blur_cfg["probability"]),
                )
            )

        if noise_cfg.get("probability", 0) > 0:
            var_limit = tuple(noise_cfg.get("gaussian_sigma", (5, 30)))
            # var_limit is squared sigma in albumentations terms
            transforms.append(
                A.OneOf(
                    [
                        A.GaussNoise(
                            var_limit=(var_limit[0] ** 2, var_limit[1] ** 2),
                            p=1.0,
                        ),
                        A.ISONoise(p=1.0),
                    ],
                    p=float(noise_cfg["probability"]),
                )
            )

        if compression_cfg.get("probability", 0) > 0:
            qlow, qhigh = compression_cfg.get("jpeg_quality", (20, 95))
            transforms.append(
                A.ImageCompression(
                    quality_lower=int(qlow),
                    quality_upper=int(qhigh),
                    p=float(compression_cfg["probability"]),
                )
            )

        if lighting_cfg.get("probability", 0) > 0:
            brightness = float(lighting_cfg.get("brightness_shift", 0.3))
            contrast = float(lighting_cfg.get("contrast_factor", (0.5, 1.5))[1] - 1.0)
            transforms.append(
                A.RandomBrightnessContrast(
                    brightness_limit=brightness,
                    contrast_limit=contrast,
                    p=float(lighting_cfg["probability"]),
                )
            )
            gamma_lo, gamma_hi = lighting_cfg.get("gamma_range", (0.5, 2.0))
            transforms.append(
                A.RandomGamma(
                    gamma_limit=(int(gamma_lo * 100), int(gamma_hi * 100)),
                    p=float(lighting_cfg["probability"]) * 0.5,
                )
            )

        if geometric_cfg.get("probability", 0) > 0:
            distort = abs(float(geometric_cfg.get("distortion_k1", (-0.3, 0.3))[1]))
            transforms.append(
                A.OpticalDistortion(
                    distort_limit=distort,
                    p=float(geometric_cfg["probability"]),
                )
            )

        return A.Compose(transforms)

    # ── Custom degradations ─────────────────────────────────────────────────

    def _apply_random_shadow(self, image: np.ndarray) -> np.ndarray:
        """Overlay a smooth random gradient as a global shadow."""
        h, w = image.shape[:2]
        # Random linear gradient mask in [0.4, 1.0] — multiplicative shading
        ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        angle = random.uniform(0, 2 * np.pi)
        t = (xs * np.cos(angle) + ys * np.sin(angle)).astype(np.float32)
        t = (t - t.min()) / max(1.0, t.max() - t.min())
        shading = 0.4 + 0.6 * t
        if random.random() < 0.5:
            shading = shading[:, ::-1]
        return np.clip(image.astype(np.float32) * shading[..., None], 0, 255).astype(
            np.uint8
        )

    def _apply_random_occlusion(self, image: np.ndarray) -> np.ndarray:
        """Place 1..max_patches random rectangles/ellipses over the image,
        each covering at most ``max_coverage / max_patches`` of the area.
        """
        h, w = image.shape[:2]
        max_patches = int(self.occlusion_cfg.get("max_patches", 3))
        max_total_coverage = float(self.occlusion_cfg.get("max_coverage", 0.20))
        n_patches = random.randint(1, max(1, max_patches))
        per_patch_area_frac = max_total_coverage / max_patches

        out = image.copy()
        for _ in range(n_patches):
            patch_area = int(h * w * per_patch_area_frac)
            patch_w = random.randint(int(patch_area**0.5 * 0.5), int(patch_area**0.5 * 2))
            patch_h = max(1, patch_area // max(1, patch_w))
            patch_w = min(patch_w, w - 1)
            patch_h = min(patch_h, h - 1)
            x = random.randint(0, max(0, w - patch_w))
            y = random.randint(0, max(0, h - patch_h))
            color = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255),
            )
            shape = random.choice(("rect", "ellipse"))
            if shape == "rect":
                cv2.rectangle(out, (x, y), (x + patch_w, y + patch_h), color, -1)
            else:
                cx = x + patch_w // 2
                cy = y + patch_h // 2
                cv2.ellipse(
                    out,
                    (cx, cy),
                    (patch_w // 2, patch_h // 2),
                    angle=random.uniform(0, 360),
                    startAngle=0,
                    endAngle=360,
                    color=color,
                    thickness=-1,
                )
        return out
