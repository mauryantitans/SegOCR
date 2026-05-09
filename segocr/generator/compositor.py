"""Text-onto-background compositor.

Implementation Guide §3.6 + Research Proposal §4.6.

Compositing modifies only the RGB image — the class mask is invariant
under all compositing operations because it represents ground-truth
character locations regardless of visual styling. Caller passes the mask
through unchanged; we accept and return it for symmetry.

Modes:
    standard         — alpha composite of solid-fill text onto background.
    semi_transparent — same as standard but with reduced fill alpha (50–90%).
    textured_fill    — text alpha used as a cookie-cutter on a texture pattern.
    outline          — text rendered as a stroke (border only).
    shadow           — drop shadow behind the text.
    emboss           — directional-lighting emboss using the mask as a
                       height-map: pixels' brightness is modulated by the
                       gradient of the mask shaded by a configurable light
                       angle. Simulates engraved / raised text.

Color strategy:
    contrast_aware — pick a foreground color with ≥4.5:1 luminance ratio
                     against the local background mean.
    random         — uniform RGB.
    low_contrast   — pick a color close to the background mean. Trains
                     the model to find text under near-camouflage
                     conditions.
"""
from __future__ import annotations

import random

import cv2
import numpy as np

LUMINANCE_TARGET_RATIO = 4.5
MAX_COLOR_TRIES = 20


class Compositor:
    """Composites rendered text onto backgrounds."""

    def __init__(self, config: dict) -> None:
        self.config = config
        self.mode_probs: dict[str, float] = dict(config["modes"])
        self.color_strategy_probs: dict[str, float] = dict(config["color_strategy"])

    # ── Public API ──────────────────────────────────────────────────────────

    def composite(
        self,
        text_rgba: np.ndarray,
        text_mask: np.ndarray,
        background: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Composite ``text_rgba`` (shape-matched to ``background``) onto the
        background. Returns (final_rgb, final_class_mask).

        The mask is passed through unchanged — caller may rely on it.
        """
        if text_rgba.shape[:2] != background.shape[:2]:
            raise ValueError(
                f"text_rgba {text_rgba.shape[:2]} != background "
                f"{background.shape[:2]} — re-canvas the text first."
            )

        bbox = _mask_bbox(text_mask)
        text_rgba = self._recolor_text(text_rgba, text_mask, background, bbox)

        mode = self._sample_mode()
        if mode == "shadow":
            background = self._apply_shadow(text_rgba, background)
        elif mode == "emboss":
            return self._apply_emboss(text_mask, background), text_mask
        elif mode == "textured_fill":
            text_rgba = self._apply_textured_fill(text_rgba, text_mask)
        elif mode == "outline":
            text_rgba = self._apply_outline(text_rgba, text_mask)
        elif mode == "semi_transparent":
            text_rgba = self._scale_alpha(text_rgba, random.uniform(0.5, 0.9))

        final_rgb = _alpha_composite(text_rgba, background)
        return final_rgb, text_mask

    # ── Color selection ─────────────────────────────────────────────────────

    def _recolor_text(
        self,
        text_rgba: np.ndarray,
        text_mask: np.ndarray,
        background: np.ndarray,
        bbox: tuple[int, int, int, int] | None,
    ) -> np.ndarray:
        """Pick a fresh foreground color for the text per the configured
        strategy and re-stamp it onto the alpha mask.

        Implementation note: the renderer fills text with an arbitrary color
        — usually black — that gets ignored at composite time in favor of
        the strategy here. We multiply the text RGB by the new color
        normalized so the alpha-shaping is preserved.
        """
        strategy = self._sample_color_strategy()
        if bbox is None:
            color = _random_color()
        elif strategy == "contrast_aware":
            color = _contrast_aware_color(background, bbox)
        elif strategy == "low_contrast":
            color = _low_contrast_color(background, bbox)
        else:
            color = _random_color()

        new_rgba = text_rgba.copy()
        # Where alpha > 0, set RGB to the chosen color (scaled by current
        # luminance for nicer AA edges).
        alpha = text_rgba[..., 3:4].astype(np.float32) / 255.0
        target = np.array(color, dtype=np.float32).reshape(1, 1, 3)
        new_rgba[..., :3] = (target * alpha + 0).astype(np.uint8)
        return new_rgba

    def _sample_color_strategy(self) -> str:
        keys = list(self.color_strategy_probs.keys())
        weights = list(self.color_strategy_probs.values())
        return random.choices(keys, weights=weights, k=1)[0]

    # ── Mode samplers / effects ─────────────────────────────────────────────

    def _sample_mode(self) -> str:
        modes = list(self.mode_probs.keys())
        weights = list(self.mode_probs.values())
        return random.choices(modes, weights=weights, k=1)[0]

    def _apply_shadow(
        self,
        text_rgba: np.ndarray,
        background: np.ndarray,
        offset: tuple[int, int] | None = None,
        blur_kernel: int | None = None,
        opacity: float = 0.5,
    ) -> np.ndarray:
        """Stamp a blurred shadow of the text onto the background and
        return the modified background. The text itself is composited
        afterwards in the main pipeline.
        """
        if offset is None:
            offset = (random.randint(2, 6), random.randint(2, 6))
        if blur_kernel is None:
            blur_kernel = random.choice([3, 5, 7])

        h, w = background.shape[:2]
        shadow_alpha = text_rgba[..., 3].astype(np.float32) * opacity
        shadow_alpha_img = np.zeros((h, w), dtype=np.float32)
        dy, dx = offset
        y_src_start = max(0, -dy)
        x_src_start = max(0, -dx)
        y_dst_start = max(0, dy)
        x_dst_start = max(0, dx)
        copy_h = min(h - y_dst_start, h - y_src_start)
        copy_w = min(w - x_dst_start, w - x_src_start)
        if copy_h > 0 and copy_w > 0:
            shadow_alpha_img[
                y_dst_start : y_dst_start + copy_h,
                x_dst_start : x_dst_start + copy_w,
            ] = shadow_alpha[
                y_src_start : y_src_start + copy_h,
                x_src_start : x_src_start + copy_w,
            ]
        if blur_kernel > 1:
            shadow_alpha_img = cv2.GaussianBlur(
                shadow_alpha_img, (blur_kernel, blur_kernel), 0
            )
        a = (shadow_alpha_img / 255.0)[..., None]
        return ((1.0 - a) * background.astype(np.float32)).astype(np.uint8)

    def _apply_emboss(
        self,
        text_mask: np.ndarray,
        background: np.ndarray,
        light_angle_deg: float | None = None,
        depth: float = 1.0,
    ) -> np.ndarray:
        """Simulate engraved / raised text by treating the binary mask as a
        height map and applying directional shading.

        Returns an RGB image (no separate text-color compositing) — the
        text becomes a brightness modulation of the local background.
        """
        if light_angle_deg is None:
            light_angle_deg = random.uniform(0, 360)

        height = (text_mask > 0).astype(np.float32) * depth
        # Soften the height map so the "edges" produce gradients to shade
        height = cv2.GaussianBlur(height, (5, 5), 1.0)
        gx = cv2.Sobel(height, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(height, cv2.CV_32F, 0, 1, ksize=3)
        rad = np.deg2rad(light_angle_deg)
        light = gx * np.cos(rad) + gy * np.sin(rad)

        # Scale to a brightness offset in [-100, +100]
        max_abs = max(1e-3, float(np.max(np.abs(light))))
        offset = (light / max_abs) * 80.0
        out = background.astype(np.float32) + offset[..., None]
        return np.clip(out, 0, 255).astype(np.uint8)

    def _apply_textured_fill(
        self, text_rgba: np.ndarray, text_mask: np.ndarray
    ) -> np.ndarray:
        """Replace the solid-color fill with a procedural texture clipped
        to the alpha cookie-cutter."""
        h, w = text_rgba.shape[:2]
        # Cheap "texture": low-frequency noise multiplied with two random
        # accent colors. Keeps us free of disk dependencies.
        coarse = np.random.rand(8, 8).astype(np.float32)
        upsampled = cv2.resize(coarse, (w, h), interpolation=cv2.INTER_CUBIC)
        c1 = np.array(_random_color(), dtype=np.float32)
        c2 = np.array(_random_color(), dtype=np.float32)
        t = upsampled[..., None]
        texture = (c1 * (1 - t) + c2 * t).astype(np.uint8)

        out = text_rgba.copy()
        alpha = text_rgba[..., 3:4].astype(np.float32) / 255.0
        out[..., :3] = (texture.astype(np.float32) * alpha).astype(np.uint8)
        return out

    def _apply_outline(self, text_rgba: np.ndarray, text_mask: np.ndarray) -> np.ndarray:
        """Replace the solid fill with a stroke: keep a 1–2 px outline,
        clear the interior."""
        kernel = np.ones((3, 3), np.uint8)
        eroded = cv2.erode((text_mask > 0).astype(np.uint8), kernel, iterations=1)
        outline = (text_mask > 0).astype(np.uint8) - eroded
        out = text_rgba.copy()
        out[..., 3] = (out[..., 3].astype(np.float32) * outline).astype(np.uint8)
        return out

    def _scale_alpha(self, text_rgba: np.ndarray, factor: float) -> np.ndarray:
        out = text_rgba.copy()
        out[..., 3] = np.clip(out[..., 3].astype(np.float32) * factor, 0, 255).astype(
            np.uint8
        )
        return out


# ── Module-level helpers ────────────────────────────────────────────────────


def _alpha_composite(rgba: np.ndarray, rgb: np.ndarray) -> np.ndarray:
    """Standard alpha-over compositing: result = fg·a + bg·(1-a)."""
    a = (rgba[..., 3:4].astype(np.float32) / 255.0)
    fg = rgba[..., :3].astype(np.float32)
    bg = rgb.astype(np.float32)
    return (fg * a + bg * (1.0 - a)).astype(np.uint8)


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _random_color() -> tuple[int, int, int]:
    return (
        random.randint(0, 255),
        random.randint(0, 255),
        random.randint(0, 255),
    )


def _luminance(color: np.ndarray | tuple[int, int, int]) -> float:
    """Relative luminance per WCAG 2.1 (sRGB → linear)."""
    c = np.asarray(color, dtype=np.float32) / 255.0
    c = np.where(c <= 0.03928, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return float(0.2126 * c[..., 0] + 0.7152 * c[..., 1] + 0.0722 * c[..., 2])


def _contrast_ratio(c1: np.ndarray | tuple, c2: np.ndarray | tuple) -> float:
    l1 = _luminance(c1)
    l2 = _luminance(c2)
    hi, lo = max(l1, l2), min(l1, l2)
    return (hi + 0.05) / (lo + 0.05)


def _contrast_aware_color(
    background: np.ndarray, bbox: tuple[int, int, int, int]
) -> tuple[int, int, int]:
    x0, y0, x1, y1 = bbox
    region = background[y0:y1, x0:x1]
    bg_mean = region.mean(axis=(0, 1))
    for _ in range(MAX_COLOR_TRIES):
        c = _random_color()
        if _contrast_ratio(c, bg_mean) >= LUMINANCE_TARGET_RATIO:
            return c
    # Couldn't find a high-contrast random color quickly — flip to
    # black/white based on background luminance.
    return (0, 0, 0) if _luminance(bg_mean) > 0.5 else (255, 255, 255)


def _low_contrast_color(
    background: np.ndarray, bbox: tuple[int, int, int, int]
) -> tuple[int, int, int]:
    x0, y0, x1, y1 = bbox
    region = background[y0:y1, x0:x1]
    bg_mean = region.mean(axis=(0, 1))
    delta = np.random.randint(-30, 31, size=3)
    return tuple(int(c) for c in np.clip(bg_mean + delta, 0, 255))
