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
    """Image + (optionally) mask degradation pipeline.

    Blur is the only degradation that meaningfully changes the visible
    silhouette of text. To keep the segmentation mask consistent with
    the rendered image after blur, we pull blur OUT of the
    albumentations Compose and apply it explicitly here, then dilate
    the per-class mask by an equivalent amount via ``apply_with_mask``.

    All other degradations (noise, JPEG compression, brightness/contrast,
    gamma, optical distortion, shadow, occlusion) leave the silhouette
    unchanged and only need to operate on the image.
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        self.blur_cfg: dict = dict(config.get("blur", {}))
        self.pipeline: A.Compose = self._build_pipeline_no_blur(config)
        self.shadow_prob: float = float(config.get("lighting", {}).get("probability", 0.0)) * 0.3
        self.occlusion_cfg: dict = dict(config.get("occlusion", {}))

    # ── Public API ──────────────────────────────────────────────────────────

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Run the full pipeline on an (H, W, 3) uint8 RGB image.

        The mask is unchanged. For mask-aware behavior (recommended for
        the standard generator), use ``apply_with_mask`` instead.
        """
        image, _kernel = self._maybe_blur(image)
        return self._apply_post_blur(image)

    def apply_with_mask(
        self,
        image: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run all degradations on the image AND propagate blur to the
        per-class mask via dilation, so the ground-truth shape matches
        the visible 'ghost' / spread the model sees.

        Returns:
            (degraded_image, dilated_mask). When no blur was sampled,
            the mask is returned unchanged (other transforms don't
            change the silhouette).
        """
        image, kernel = self._maybe_blur(image)
        if kernel >= 3:
            # GaussianBlur with kernel K spreads each pixel by ~K/2 in
            # each direction. Dilate per-class by that amount so the
            # mask covers the visible ghost.
            radius = max(1, kernel // 2)
            mask = _dilate_per_class(mask, 2 * radius + 1)
        image = self._apply_post_blur(image)
        return image, mask

    def _apply_post_blur(self, image: np.ndarray) -> np.ndarray:
        """All non-blur degradations: noise, JPEG, lighting, geometric,
        plus the custom shadow + occlusion."""
        image = self.pipeline(image=image)["image"]
        if random.random() < self.shadow_prob:
            image = self._apply_random_shadow(image)
        if random.random() < float(self.occlusion_cfg.get("probability", 0.0)):
            image = self._apply_random_occlusion(image)
        return image

    def _maybe_blur(
        self, image: np.ndarray
    ) -> tuple[np.ndarray, int]:
        """Sample a gaussian blur per ``self.blur_cfg``. Returns the
        possibly-blurred image and the actual odd kernel used (0 if no
        blur applied).
        """
        prob = float(self.blur_cfg.get("probability", 0))
        if random.random() >= prob:
            return image, 0
        k_min, k_max = self.blur_cfg.get("motion_kernel", (3, 7))
        k_min, k_max = int(k_min), int(k_max)
        if k_max < 3:
            return image, 0
        kernel = random.randint(max(3, k_min), max(3, k_max))
        if kernel % 2 == 0:
            kernel += 1
        return cv2.GaussianBlur(image, (kernel, kernel), 0), kernel

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

    def _build_pipeline_no_blur(self, config: dict) -> A.Compose:
        """Build the albumentations Compose for all NON-blur degradations.

        Blur is handled separately by ``_maybe_blur`` so the dilation in
        ``apply_with_mask`` can match the actually-applied blur kernel.
        """
        noise_cfg = config.get("noise", {})
        compression_cfg = config.get("compression", {})
        lighting_cfg = config.get("lighting", {})
        geometric_cfg = config.get("geometric", {})

        transforms: list = []

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


# ── Module-level helpers ────────────────────────────────────────────────────


def _dilate_per_class(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    """Dilate each non-background class's region by ``kernel_size`` pixels.

    Used by ``apply_with_mask`` to grow the per-class mask so it covers
    the visible 'ghost' spread that GaussianBlur produces. Last
    (highest) class wins on overlap, matching the renderer's z-order
    convention.

    Args:
        mask: (H, W) integer class-id mask.
        kernel_size: odd int; if even, the next-smaller odd value is
            used (cv2.dilate requires odd kernels).

    Returns:
        (H, W) dilated class-id mask, same dtype as input.
    """
    if kernel_size <= 1:
        return mask
    if kernel_size % 2 == 0:
        kernel_size = max(3, kernel_size - 1)
    classes = np.unique(mask)
    classes = classes[classes != 0]
    if len(classes) == 0:
        return mask
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    out = np.zeros_like(mask)
    for cls in classes:
        binary = (mask == cls).astype(np.uint8)
        dilated = cv2.dilate(binary, kernel, iterations=1)
        out[dilated > 0] = cls
    return out
