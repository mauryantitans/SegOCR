"""Layout engine — arranges rendered text in 6 spatial configurations.

Implementation Guide §3.4 + Mode 6 (paragraph) added per brainstorming #11.

The contract: take a rendered text strip (RGBA image + class-id mask +
per-character metadata) and a target image size, apply one of 6 spatial
transforms, place the result onto the canvas, and return the final
(rgba, mask, metadata) for the canvas-sized image.

Critical invariant: the same transform must be applied to both the RGB
image (cv2.INTER_LINEAR) and the class mask (cv2.INTER_NEAREST). Any
other interpolation on the mask creates fractional class IDs.

Modes:
    horizontal  — placed flat with random offset.
    rotated     — full text block rotated by a random angle.
    curved      — sinusoidal y-displacement applied via cv2.remap.
    perspective — random 4-point perspective warp.
    deformed    — coarse-grid displacement field, bilinear-upsampled, then
                  cv2.remap. Approximates thin-plate-spline elastic warp.
    paragraph   — multiple pre-rendered lines stacked vertically.
"""
from __future__ import annotations

import random
from typing import Any

import cv2
import numpy as np

LayoutResult = tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]


class LayoutEngine:
    """Arranges rendered text using one of 6 layout modes."""

    def __init__(self, config: dict) -> None:
        self.config = config
        self.mode_probs: dict[str, float] = dict(config["modes"])
        self.rotation_range: tuple[float, float] = tuple(config["rotation_range"])
        self.curve_types: list[str] = list(config.get("curve_types", ["sinusoidal"]))
        self.perspective_strength: tuple[float, float] = tuple(
            config["perspective_strength"]
        )
        self.deformation_strength: tuple[float, float] = tuple(
            config["deformation_strength"]
        )
        self.paragraph_cfg: dict = config.get("paragraph", {}) or {}

    # ── Public API ──────────────────────────────────────────────────────────

    def sample_mode(self) -> str:
        modes = list(self.mode_probs.keys())
        weights = list(self.mode_probs.values())
        return random.choices(modes, weights=weights, k=1)[0]

    def apply_layout(
        self,
        text_rgba: np.ndarray,
        text_mask: np.ndarray,
        char_metadata: list[dict[str, Any]],
        image_size: tuple[int, int],
        mode: str | None = None,
        placement_score_map: np.ndarray | None = None,
    ) -> LayoutResult:
        """Apply a (sampled or explicit) layout mode and place the result on
        an ``image_size`` canvas.

        Args:
            placement_score_map: optional (H, W) float32 saliency map from
                ``segocr.generator.saliency.compute_placement_score``. When
                provided, the canvas-placement step picks the best region
                instead of a random offset.

        For paragraph mode, prefer ``apply_paragraph`` directly — calling
        this with mode='paragraph' falls back to horizontal because we
        don't have multiple rendered lines to stack.
        """
        if mode is None:
            mode = self.sample_mode()
        if mode == "paragraph":
            mode = "horizontal"  # caller should use apply_paragraph instead

        if mode == "rotated":
            text_rgba, text_mask, char_metadata = self._rotated(
                text_rgba, text_mask, char_metadata
            )
        elif mode == "curved":
            text_rgba, text_mask, char_metadata = self._curved(
                text_rgba, text_mask, char_metadata
            )
        elif mode == "perspective":
            text_rgba, text_mask, char_metadata = self._perspective(
                text_rgba, text_mask, char_metadata
            )
        elif mode == "deformed":
            text_rgba, text_mask, char_metadata = self._deformed(
                text_rgba, text_mask, char_metadata
            )
        # horizontal — no transform

        return _place_on_canvas(
            text_rgba, text_mask, char_metadata, image_size, placement_score_map
        )

    def apply_paragraph(
        self,
        rendered_lines: list[LayoutResult],
        image_size: tuple[int, int],
        placement_score_map: np.ndarray | None = None,
    ) -> LayoutResult:
        """Stack pre-rendered lines vertically with sampled line / word
        spacing, then place onto an image-sized canvas.

        Each element of ``rendered_lines`` is a (text_rgba, text_mask,
        per-line metadata) triple as produced by ``CharacterRenderer.render_text``.
        """
        if not rendered_lines:
            return _empty_canvas(image_size)

        spacing_range = self.paragraph_cfg.get("line_spacing", [1.1, 1.6])
        line_factor = random.uniform(*spacing_range)

        max_w = max(line[0].shape[1] for line in rendered_lines)
        line_heights = [line[0].shape[0] for line in rendered_lines]
        line_pitch = int(max(line_heights) * line_factor)

        total_h = line_pitch * (len(rendered_lines) - 1) + line_heights[-1]
        total_h = max(total_h, max(line_heights))

        align = random.choice(self.paragraph_cfg.get("align", ["left", "center"]))

        para_rgba = np.zeros((total_h, max_w, 4), dtype=np.uint8)
        para_mask = np.zeros((total_h, max_w), dtype=np.uint8)
        para_metadata: list[dict[str, Any]] = []

        for i, (rgba, mask, meta) in enumerate(rendered_lines):
            y_off = i * line_pitch
            line_w = rgba.shape[1]
            if align == "center":
                x_off = (max_w - line_w) // 2
            elif align == "justified":
                # v0: same as left — full justification needs whitespace
                # detection. Defer to v2.
                x_off = 0
            else:
                x_off = 0

            line_h = rgba.shape[0]
            if y_off + line_h > total_h:
                line_h = total_h - y_off
                if line_h <= 0:
                    break
                rgba = rgba[:line_h]
                mask = mask[:line_h]

            _alpha_paste(para_rgba, rgba, x_off, y_off)
            _mask_paste(para_mask, mask, x_off, y_off)
            for m in meta:
                bbox = m["bbox"]
                cx, cy = m["centroid"]
                para_metadata.append(
                    {
                        **m,
                        "bbox": (
                            bbox[0] + x_off,
                            bbox[1] + y_off,
                            bbox[2] + x_off,
                            bbox[3] + y_off,
                        ),
                        "centroid": (cx + x_off, cy + y_off),
                        "line_index": i,
                    }
                )

        return _place_on_canvas(
            para_rgba, para_mask, para_metadata, image_size, placement_score_map
        )

    # ── Mode implementations ────────────────────────────────────────────────

    def _rotated(
        self,
        rgba: np.ndarray,
        mask: np.ndarray,
        metadata: list[dict[str, Any]],
    ) -> LayoutResult:
        h, w = rgba.shape[:2]
        angle_deg = random.uniform(*self.rotation_range)
        center = (w / 2.0, h / 2.0)

        # Compute the bounding box of the rotated rectangle and re-center
        # so the rotated content fits without cropping.
        rad = np.deg2rad(angle_deg)
        cos_a, sin_a = abs(np.cos(rad)), abs(np.sin(rad))
        new_w = int(np.ceil(h * sin_a + w * cos_a))
        new_h = int(np.ceil(h * cos_a + w * sin_a))
        m = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
        m[0, 2] += (new_w - w) / 2.0
        m[1, 2] += (new_h - h) / 2.0

        rotated_rgba = cv2.warpAffine(
            rgba, m, (new_w, new_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )
        rotated_mask = cv2.warpAffine(
            mask, m, (new_w, new_h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        return rotated_rgba, rotated_mask, _retransform_metadata_affine(metadata, m)

    def _curved(
        self,
        rgba: np.ndarray,
        mask: np.ndarray,
        metadata: list[dict[str, Any]],
    ) -> LayoutResult:
        h, w = rgba.shape[:2]
        amplitude = random.uniform(0.05, 0.20) * h
        frequency = random.uniform(0.5, 2.0)
        curve = random.choice(self.curve_types)

        x_coords, y_coords = np.meshgrid(
            np.arange(w, dtype=np.float32),
            np.arange(h, dtype=np.float32),
        )
        if curve == "circular":
            radius = max(w, 1) / random.uniform(np.pi / 2, np.pi * 1.5)
            displacement = radius - np.sqrt(
                np.maximum(0.0, radius**2 - (x_coords - w / 2.0) ** 2)
            )
            displacement -= displacement.mean()
            displacement *= np.sign(random.uniform(-1, 1))
        elif curve == "bezier":
            # Approximate with two-arc sine wave.
            displacement = amplitude * np.sin(np.pi * x_coords / max(w, 1))
        else:  # sinusoidal
            displacement = amplitude * np.sin(2 * np.pi * frequency * x_coords / max(w, 1))

        # Pad vertically so the curved strip doesn't get clipped.
        pad = int(np.ceil(np.max(np.abs(displacement)) + 4))
        padded_h = h + 2 * pad
        rgba_padded = cv2.copyMakeBorder(
            rgba, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0, 0)
        )
        mask_padded = cv2.copyMakeBorder(
            mask, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=0
        )

        x_coords_p, y_coords_p = np.meshgrid(
            np.arange(w, dtype=np.float32),
            np.arange(padded_h, dtype=np.float32),
        )
        # source_y for output row i is i - displacement(x), mapped into
        # padded coordinates.
        disp_p = np.tile(displacement[0:1, :], (padded_h, 1))
        map_x = x_coords_p
        map_y = y_coords_p - disp_p

        warped_rgba = cv2.remap(
            rgba_padded, map_x, map_y, cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0),
        )
        warped_mask = cv2.remap(
            mask_padded, map_x, map_y, cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )

        # Update metadata: shift y by pad, then by -disp_at_centroid_x.
        new_metadata = []
        for m in metadata:
            cx, cy = m["centroid"]
            x0, y0, x1, y1 = m["bbox"]
            disp_at_cx = float(displacement[0, int(min(max(cx, 0), w - 1))])
            new_metadata.append(
                {
                    **m,
                    "bbox": (x0, y0 + pad + int(disp_at_cx), x1, y1 + pad + int(disp_at_cx)),
                    "centroid": (cx, cy + pad + disp_at_cx),
                }
            )

        return warped_rgba, warped_mask, new_metadata

    def _perspective(
        self,
        rgba: np.ndarray,
        mask: np.ndarray,
        metadata: list[dict[str, Any]],
    ) -> LayoutResult:
        h, w = rgba.shape[:2]
        strength = random.uniform(*self.perspective_strength)
        max_dx = strength * w
        max_dy = strength * h

        src = np.array(
            [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32
        )
        dst = src + np.random.uniform(
            -1, 1, size=src.shape
        ).astype(np.float32) * np.array([max_dx, max_dy], dtype=np.float32)

        # Clamp to non-degenerate quad (re-order corners ascending if a
        # random perturbation flips the quad inside-out).
        dst[:, 0] = np.clip(dst[:, 0], -max_dx, w - 1 + max_dx)
        dst[:, 1] = np.clip(dst[:, 1], -max_dy, h - 1 + max_dy)

        # Re-anchor dst to non-negative corner so output canvas size is
        # exactly the bounding rectangle of the destination quad.
        offset = dst.min(axis=0)
        dst = dst - offset
        new_w = int(np.ceil(dst[:, 0].max())) + 1
        new_h = int(np.ceil(dst[:, 1].max())) + 1

        m = cv2.getPerspectiveTransform(src, dst)
        warped_rgba = cv2.warpPerspective(
            rgba, m, (new_w, new_h), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0),
        )
        warped_mask = cv2.warpPerspective(
            mask, m, (new_w, new_h), flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
        return warped_rgba, warped_mask, _retransform_metadata_perspective(metadata, m)

    def _deformed(
        self,
        rgba: np.ndarray,
        mask: np.ndarray,
        metadata: list[dict[str, Any]],
    ) -> LayoutResult:
        h, w = rgba.shape[:2]
        strength = random.uniform(*self.deformation_strength)

        grid_size = 4
        dx_coarse = (np.random.randn(grid_size, grid_size) * strength * w / 6.0).astype(
            np.float32
        )
        dy_coarse = (np.random.randn(grid_size, grid_size) * strength * h / 6.0).astype(
            np.float32
        )
        dx = cv2.resize(dx_coarse, (w, h), interpolation=cv2.INTER_CUBIC)
        dy = cv2.resize(dy_coarse, (w, h), interpolation=cv2.INTER_CUBIC)

        x_coords, y_coords = np.meshgrid(
            np.arange(w, dtype=np.float32),
            np.arange(h, dtype=np.float32),
        )
        map_x = x_coords + dx
        map_y = y_coords + dy

        warped_rgba = cv2.remap(
            rgba, map_x, map_y, cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0),
        )
        warped_mask = cv2.remap(
            mask, map_x, map_y, cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )

        # Approximate metadata update — shift centroid by displacement at
        # the original centroid location.
        new_metadata = []
        for m in metadata:
            cx, cy = m["centroid"]
            ix = int(min(max(cx, 0), w - 1))
            iy = int(min(max(cy, 0), h - 1))
            new_cx = cx - float(dx[iy, ix])
            new_cy = cy - float(dy[iy, ix])
            x0, y0, x1, y1 = m["bbox"]
            shift_x = new_cx - cx
            shift_y = new_cy - cy
            new_metadata.append(
                {
                    **m,
                    "bbox": (
                        int(x0 + shift_x),
                        int(y0 + shift_y),
                        int(x1 + shift_x),
                        int(y1 + shift_y),
                    ),
                    "centroid": (new_cx, new_cy),
                }
            )

        return warped_rgba, warped_mask, new_metadata


# ── Module-level helpers ────────────────────────────────────────────────────


def _empty_canvas(image_size: tuple[int, int]) -> LayoutResult:
    h, w = image_size
    return (
        np.zeros((h, w, 4), dtype=np.uint8),
        np.zeros((h, w), dtype=np.uint8),
        [],
    )


def _place_on_canvas(
    text_rgba: np.ndarray,
    text_mask: np.ndarray,
    char_metadata: list[dict[str, Any]],
    image_size: tuple[int, int],
    placement_score_map: np.ndarray | None = None,
) -> LayoutResult:
    """Place a transformed text strip onto an image-sized canvas, scaling
    it down first if it exceeds the canvas. If ``placement_score_map`` is
    provided, the offset is picked via saliency; otherwise random.
    """
    canvas_h, canvas_w = image_size
    text_h, text_w = text_rgba.shape[:2]

    if text_h > canvas_h or text_w > canvas_w:
        scale = min(canvas_h / text_h, canvas_w / text_w) * 0.95
        new_w = max(1, int(text_w * scale))
        new_h = max(1, int(text_h * scale))
        text_rgba = cv2.resize(text_rgba, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        text_mask = cv2.resize(text_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        char_metadata = [_scale_metadata(m, scale) for m in char_metadata]
        text_h, text_w = new_h, new_w

    if placement_score_map is not None:
        from segocr.generator.saliency import find_best_position

        if placement_score_map.shape != (canvas_h, canvas_w):
            placement_score_map = cv2.resize(
                placement_score_map, (canvas_w, canvas_h), interpolation=cv2.INTER_LINEAR
            )
        y_offset, x_offset = find_best_position(
            placement_score_map, (text_h, text_w)
        )
    else:
        y_max = max(0, canvas_h - text_h)
        x_max = max(0, canvas_w - text_w)
        y_offset = random.randint(0, y_max) if y_max > 0 else 0
        x_offset = random.randint(0, x_max) if x_max > 0 else 0

    canvas_rgba = np.zeros((canvas_h, canvas_w, 4), dtype=np.uint8)
    canvas_mask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    canvas_rgba[y_offset : y_offset + text_h, x_offset : x_offset + text_w] = text_rgba
    canvas_mask[y_offset : y_offset + text_h, x_offset : x_offset + text_w] = text_mask

    updated_metadata = [
        _translate_metadata(m, x_offset, y_offset) for m in char_metadata
    ]
    return canvas_rgba, canvas_mask, updated_metadata


def _clip_paste_region(
    canvas_shape: tuple[int, int],
    src_shape: tuple[int, int],
    x: int,
    y: int,
) -> tuple[int, int, int, int, int, int, int, int] | None:
    """Compute clipped (dst, src) slice indices for a (x, y) paste of
    ``src_shape`` onto a ``canvas_shape`` canvas. Returns None if the
    paste lands entirely off-canvas."""
    th, tw = canvas_shape
    sh, sw = src_shape
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(tw, x + sw)
    y1 = min(th, y + sh)
    if x1 <= x0 or y1 <= y0:
        return None
    sx0 = x0 - x
    sy0 = y0 - y
    sx1 = sx0 + (x1 - x0)
    sy1 = sy0 + (y1 - y0)
    return x0, y0, x1, y1, sx0, sy0, sx1, sy1


def _alpha_paste(
    target_rgba: np.ndarray, src_rgba: np.ndarray, x: int, y: int
) -> None:
    """Alpha-composite ``src_rgba`` onto ``target_rgba`` at (x, y) in place."""
    region = _clip_paste_region(
        target_rgba.shape[:2], src_rgba.shape[:2], x, y
    )
    if region is None:
        return
    x0, y0, x1, y1, sx0, sy0, sx1, sy1 = region

    src = src_rgba[sy0:sy1, sx0:sx1].astype(np.float32)
    dst = target_rgba[y0:y1, x0:x1].astype(np.float32)
    sa = src[..., 3:4] / 255.0
    da = dst[..., 3:4] / 255.0
    out_a = sa + da * (1.0 - sa)
    out_rgb = (src[..., :3] * sa + dst[..., :3] * da * (1.0 - sa)) / np.maximum(out_a, 1e-6)
    target_rgba[y0:y1, x0:x1, :3] = np.clip(out_rgb, 0, 255).astype(np.uint8)
    target_rgba[y0:y1, x0:x1, 3] = np.clip(out_a[..., 0] * 255.0, 0, 255).astype(np.uint8)


def _mask_paste(target_mask: np.ndarray, src_mask: np.ndarray, x: int, y: int) -> None:
    """Stamp src_mask onto target_mask at (x, y); later writes win on overlap."""
    region = _clip_paste_region(
        target_mask.shape[:2], src_mask.shape[:2], x, y
    )
    if region is None:
        return
    x0, y0, x1, y1, sx0, sy0, sx1, sy1 = region
    src = src_mask[sy0:sy1, sx0:sx1]
    target_mask[y0:y1, x0:x1] = np.where(src > 0, src, target_mask[y0:y1, x0:x1])


def _translate_metadata(
    m: dict[str, Any], dx: int, dy: int
) -> dict[str, Any]:
    x0, y0, x1, y1 = m["bbox"]
    cx, cy = m["centroid"]
    return {
        **m,
        "bbox": (x0 + dx, y0 + dy, x1 + dx, y1 + dy),
        "centroid": (cx + dx, cy + dy),
    }


def _scale_metadata(m: dict[str, Any], scale: float) -> dict[str, Any]:
    x0, y0, x1, y1 = m["bbox"]
    cx, cy = m["centroid"]
    return {
        **m,
        "bbox": (
            int(x0 * scale),
            int(y0 * scale),
            int(x1 * scale),
            int(y1 * scale),
        ),
        "centroid": (cx * scale, cy * scale),
        "area": int(m.get("area", 0) * scale * scale),
    }


def _retransform_metadata_affine(
    metadata: list[dict[str, Any]], m: np.ndarray
) -> list[dict[str, Any]]:
    """Apply a 2×3 affine to centroids and bboxes (axis-aligned bbox of
    the transformed corners — slight overestimate, fine for our use)."""
    out = []
    for entry in metadata:
        cx, cy = entry["centroid"]
        c = np.array([cx, cy, 1.0])
        new_cx, new_cy = m @ c
        x0, y0, x1, y1 = entry["bbox"]
        corners = np.array(
            [[x0, y0, 1.0], [x1, y0, 1.0], [x1, y1, 1.0], [x0, y1, 1.0]]
        )
        new_corners = (m @ corners.T).T
        nx0, ny0 = new_corners.min(axis=0)
        nx1, ny1 = new_corners.max(axis=0)
        out.append(
            {
                **entry,
                "bbox": (int(nx0), int(ny0), int(nx1), int(ny1)),
                "centroid": (float(new_cx), float(new_cy)),
            }
        )
    return out


def _retransform_metadata_perspective(
    metadata: list[dict[str, Any]], m: np.ndarray
) -> list[dict[str, Any]]:
    out = []
    for entry in metadata:
        cx, cy = entry["centroid"]
        x0, y0, x1, y1 = entry["bbox"]
        pts = np.array(
            [[cx, cy], [x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32
        ).reshape(-1, 1, 2)
        new_pts = cv2.perspectiveTransform(pts, m).reshape(-1, 2)
        new_cx, new_cy = new_pts[0]
        corners = new_pts[1:]
        nx0, ny0 = corners.min(axis=0)
        nx1, ny1 = corners.max(axis=0)
        out.append(
            {
                **entry,
                "bbox": (int(nx0), int(ny0), int(nx1), int(ny1)),
                "centroid": (float(new_cx), float(new_cy)),
            }
        )
    return out
