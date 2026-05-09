"""Character renderer — the oracle.

Implementation Guide §3.3 + Research Proposal §4.3.

This module is where the *oracle property* originates. Every character is
programmatically rendered onto a transparent RGBA canvas, and the alpha
channel — thresholded at 128 — IS the ground-truth segmentation mask.
There is no annotation, no labelling tool, no human in the loop.

Implementation details that matter:

  Supersampling.   Render at 2× the target size and downsample with LANCZOS
                   for the RGB image, NEAREST for the mask. Without this,
                   anti-aliased edges in real photographs are easy to spot;
                   the model latches onto the absence of AA in synthetic
                   data and refuses to generalize.

  Mask resampling. ALWAYS use Image.NEAREST for the class mask. Bilinear
                   or bicubic on a class-id mask creates fractional class
                   IDs (e.g. 1.7) that no longer correspond to any
                   character. This is gotcha #1 in the Implementation
                   Guide; getting it wrong corrupts training silently.

  String rendering. Pillow's ``draw.text`` handles kerning automatically
                    when you draw the full string in one call. To keep
                    that benefit while still producing per-character
                    masks, we render the full string once for the RGB
                    image, then render each character *individually at
                    its kerned position* to extract its mask. The mask of
                    a character is exactly the alpha that character
                    contributed at that position.

  Z-ordering on overlap. When two character masks overlap (rare for
                         horizontal text, common after layout transforms),
                         the later character in the string wins. This
                         matches the visual rendering order.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image, ImageDraw

from segocr.utils.charset import CHARSET_TIER1, char_to_class_id

if TYPE_CHECKING:
    from PIL.ImageFont import FreeTypeFont

    from segocr.generator.font_manager import FontManager

SUPERSAMPLE_FACTOR = 2
ALPHA_THRESHOLD = 128
DEFAULT_PADDING = 4


class CharacterRenderer:
    """Renders characters and extracts oracle segmentation masks.

    Stateless beyond the char→class-id map. Safe to share across worker
    processes.
    """

    def __init__(
        self,
        config: dict,
        font_manager: FontManager | None = None,
        tier: int = 1,
    ) -> None:
        self.config = config
        self.font_manager = font_manager
        self.tier = tier
        self.char_to_class: dict[str, int] = char_to_class_id(tier)
        self.charset = (
            CHARSET_TIER1 if tier == 1 else tuple(self.char_to_class.keys())
        )

    def get_class_id(self, char: str) -> int:
        """Map ``char`` → class-id. 0 = background, 1..N = characters."""
        return self.char_to_class.get(char, 0)

    # ── Per-character rendering ────────────────────────────────────────────

    def render_character(
        self,
        char: str,
        font: FreeTypeFont,
        size: int,
        color: tuple[int, int, int] = (0, 0, 0),
        padding: int = DEFAULT_PADDING,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Render a single character on a tightly-cropped RGBA canvas.

        Args:
            char:    a single-character string.
            font:    Pillow FreeTypeFont (any size — we'll re-derive at
                     ``size`` × supersample for AA).
            size:    target output pixel size for the character height.
            color:   foreground RGB.
            padding: extra pixels around the rendered character (in
                     output-resolution units, not supersampled).

        Returns:
            rgba_image: (H, W, 4) uint8 — anti-aliased RGBA.
            mask:       (H, W)    uint8 — binary {0, 1} alpha-thresholded
                                          at 128 in the supersampled
                                          domain, then nearest-neighbor
                                          downsampled.
        """
        if len(char) != 1:
            raise ValueError(f"render_character expects exactly one char, got: {char!r}")

        big_size = size * SUPERSAMPLE_FACTOR
        big_pad = padding * SUPERSAMPLE_FACTOR
        big_font = font.font_variant(size=big_size)

        # Bbox at supersampled resolution; bbox can have negative origin so we
        # offset our draw call to make sure (0, 0) is the top-left.
        bbox = big_font.getbbox(char)
        if bbox is None:
            return _empty_rgba_mask()
        x0, y0, x1, y1 = bbox
        if x1 - x0 <= 0 or y1 - y0 <= 0:
            return _empty_rgba_mask()

        canvas_w = (x1 - x0) + big_pad * 2
        canvas_h = (y1 - y0) + big_pad * 2

        canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (big_pad - x0, big_pad - y0),
            char,
            font=big_font,
            fill=(*color, 255),
        )

        big_arr = np.array(canvas)
        big_mask = (big_arr[..., 3] >= ALPHA_THRESHOLD).astype(np.uint8)

        out_w = max(1, canvas_w // SUPERSAMPLE_FACTOR)
        out_h = max(1, canvas_h // SUPERSAMPLE_FACTOR)

        rgba = np.array(
            Image.fromarray(big_arr).resize((out_w, out_h), Image.LANCZOS)
        )
        # Multiply by 255 only for the resize — Pillow needs uint8 input
        # in {0, 255} so NEAREST keeps a clean binary output.
        mask_resized = np.array(
            Image.fromarray(big_mask * 255).resize(
                (out_w, out_h), Image.NEAREST
            )
        )
        mask = (mask_resized > 0).astype(np.uint8)

        return rgba, mask

    # ── Multi-character text rendering ─────────────────────────────────────

    def render_text(
        self,
        text: str,
        font: FreeTypeFont,
        size: int,
        color: tuple[int, int, int] = (0, 0, 0),
        char_spacing: float = 1.0,  # noqa: ARG002 — kerning handled by Pillow
        padding: int = DEFAULT_PADDING,
    ) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
        """Render a text string as a horizontal strip with per-char masks.

        The visible RGBA image is rendered in a single Pillow call so
        kerning is preserved. Per-character masks are extracted by
        re-rendering each character individually at its kerned x-offset
        — the character's mask is the set of pixels its alpha contributed
        in that solo render.

        Args:
            text:         the string to render (chars not in the active
                          charset will have class_id 0 and be excluded
                          from metadata; their visible pixels still go
                          into the RGB image).
            font:         Pillow FreeTypeFont.
            size:         target output character pixel size.
            color:        foreground RGB.
            char_spacing: currently ignored (kept for API stability) —
                          Pillow's native kerning is preferred over a
                          manually-added per-character gap.
            padding:      extra pixels around the rendered string.

        Returns:
            text_image:    (H, W, 4) uint8 RGBA — kerned, anti-aliased.
            class_mask:    (H, W)    uint8 — pixel = char class-id (0 = bg).
            char_metadata: list of per-character dicts containing
                           ``char``, ``class_id``, ``bbox``,
                           ``centroid``, ``area``. Bboxes / centroids in
                           output-resolution coordinates.
        """
        if not text:
            return _empty_text_rgba_mask()

        big_size = size * SUPERSAMPLE_FACTOR
        big_pad = padding * SUPERSAMPLE_FACTOR
        big_font = font.font_variant(size=big_size)

        full_bbox = big_font.getbbox(text)
        if full_bbox is None:
            return _empty_text_rgba_mask()
        x0, y0, x1, y1 = full_bbox
        if x1 - x0 <= 0 or y1 - y0 <= 0:
            return _empty_text_rgba_mask()

        canvas_w = (x1 - x0) + big_pad * 2
        canvas_h = (y1 - y0) + big_pad * 2
        # Anchor used for the full string render — also the anchor for
        # individual character renders, so masks line up exactly.
        anchor_x = big_pad - x0
        anchor_y = big_pad - y0

        # Full-string RGBA render (kerning preserved).
        full_canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
        ImageDraw.Draw(full_canvas).text(
            (anchor_x, anchor_y), text, font=big_font, fill=(*color, 255)
        )

        # Per-character class mask (supersampled, before downsample).
        class_mask_big = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
        metadata_supersampled: list[dict[str, Any]] = []

        for i, char in enumerate(text):
            cls_id = self.char_to_class.get(char, 0)
            if cls_id == 0:
                continue

            # x-offset of char i within the full string, accounting for
            # kerning of all previous characters.
            prefix = text[:i]
            try:
                prefix_w = big_font.getlength(prefix) if prefix else 0
            except AttributeError:
                # Older Pillow: fall back to bbox width
                if prefix:
                    pb = big_font.getbbox(prefix)
                    prefix_w = (pb[2] - pb[0]) if pb else 0
                else:
                    prefix_w = 0

            char_canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
            ImageDraw.Draw(char_canvas).text(
                (anchor_x + prefix_w, anchor_y),
                char,
                font=big_font,
                fill=(*color, 255),
            )
            char_alpha = np.array(char_canvas)[..., 3]
            char_mask = char_alpha >= ALPHA_THRESHOLD
            if not char_mask.any():
                continue

            # z-order: later chars overwrite earlier ones on overlap.
            class_mask_big[char_mask] = cls_id

            ys, xs = np.where(char_mask)
            metadata_supersampled.append(
                {
                    "char": char,
                    "class_id": cls_id,
                    "bbox_big": (
                        int(xs.min()),
                        int(ys.min()),
                        int(xs.max()) + 1,
                        int(ys.max()) + 1,
                    ),
                    "centroid_big": (float(xs.mean()), float(ys.mean())),
                    "area_big": int(char_mask.sum()),
                }
            )

        out_w = max(1, canvas_w // SUPERSAMPLE_FACTOR)
        out_h = max(1, canvas_h // SUPERSAMPLE_FACTOR)
        scale = 1.0 / SUPERSAMPLE_FACTOR

        rgba = np.array(
            Image.fromarray(np.array(full_canvas)).resize(
                (out_w, out_h), Image.LANCZOS
            )
        )
        class_mask = np.array(
            Image.fromarray(class_mask_big).resize(
                (out_w, out_h), Image.NEAREST
            )
        )

        char_metadata: list[dict[str, Any]] = []
        for m in metadata_supersampled:
            x0_b, y0_b, x1_b, y1_b = m["bbox_big"]
            cx_b, cy_b = m["centroid_big"]
            char_metadata.append(
                {
                    "char": m["char"],
                    "class_id": m["class_id"],
                    "bbox": (
                        int(x0_b * scale),
                        int(y0_b * scale),
                        max(int(x0_b * scale) + 1, int(x1_b * scale)),
                        max(int(y0_b * scale) + 1, int(y1_b * scale)),
                    ),
                    "centroid": (cx_b * scale, cy_b * scale),
                    "area": int(m["area_big"] * scale * scale),
                }
            )

        return rgba, class_mask, char_metadata


# ── Helpers ─────────────────────────────────────────────────────────────────


def _empty_rgba_mask() -> tuple[np.ndarray, np.ndarray]:
    return (
        np.zeros((1, 1, 4), dtype=np.uint8),
        np.zeros((1, 1), dtype=np.uint8),
    )


def _empty_text_rgba_mask() -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    rgba, mask = _empty_rgba_mask()
    return rgba, mask, []
