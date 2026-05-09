"""LayoutEngine tests — verifies the mask-class-id-preservation contract."""
from __future__ import annotations

import numpy as np
import pytest

from segocr.generator.font_manager import FontManager
from segocr.generator.layout import LayoutEngine
from segocr.generator.renderer import CharacterRenderer
from segocr.utils.charset import char_to_class_id


@pytest.fixture
def layout_config() -> dict:
    return {
        "modes": {
            "horizontal": 1.0 / 6,
            "rotated": 1.0 / 6,
            "curved": 1.0 / 6,
            "perspective": 1.0 / 6,
            "deformed": 1.0 / 6,
            "paragraph": 1.0 / 6,
        },
        "rotation_range": [-45, 45],
        "curve_types": ["sinusoidal", "circular", "bezier"],
        "perspective_strength": [0.05, 0.20],
        "deformation_strength": [0.05, 0.20],
        "paragraph": {
            "lines": [2, 4],
            "line_spacing": [1.1, 1.4],
            "word_spacing": [0.6, 1.2],
            "align": ["left", "center"],
        },
    }


@pytest.fixture
def layout_engine(layout_config):
    return LayoutEngine(layout_config)


@pytest.fixture
def rendered_text(font_manager: FontManager):
    renderer = CharacterRenderer(config={}, font_manager=font_manager, tier=1)
    font, _ = font_manager.sample_font(size=24)
    return renderer.render_text("ABCDE", font, size=24)


# ── Common invariants across all modes ─────────────────────────────────────


@pytest.mark.parametrize(
    "mode", ["horizontal", "rotated", "curved", "perspective", "deformed"]
)
def test_layout_returns_canvas_sized_output(
    layout_engine: LayoutEngine, rendered_text, mode: str
) -> None:
    text_rgba, text_mask, metadata = rendered_text
    image_size = (256, 256)
    out_rgba, out_mask, out_meta = layout_engine.apply_layout(
        text_rgba, text_mask, metadata, image_size, mode=mode
    )
    assert out_rgba.shape == (256, 256, 4)
    assert out_mask.shape == (256, 256)
    assert out_rgba.dtype == np.uint8
    assert out_mask.dtype == np.uint8


@pytest.mark.parametrize(
    "mode", ["horizontal", "rotated", "curved", "perspective", "deformed"]
)
def test_layout_mask_only_contains_valid_class_ids(
    layout_engine: LayoutEngine, rendered_text, mode: str
) -> None:
    """No fractional class IDs should appear after any transform — proves
    that NEAREST interpolation is used on the mask everywhere."""
    text_rgba, text_mask, metadata = rendered_text
    expected = set(np.unique(text_mask).tolist())
    out_rgba, out_mask, _ = layout_engine.apply_layout(
        text_rgba, text_mask, metadata, (256, 256), mode=mode
    )
    actual = set(np.unique(out_mask).tolist())
    # After transforms, some class IDs may be lost (mapped outside canvas)
    # but no NEW (invalid) class IDs should appear.
    invalid = actual - expected
    assert not invalid, f"Mode {mode} introduced invalid class IDs: {invalid}"


@pytest.mark.parametrize(
    "mode", ["horizontal", "rotated", "curved", "perspective", "deformed"]
)
def test_layout_metadata_count_preserved(
    layout_engine: LayoutEngine, rendered_text, mode: str
) -> None:
    text_rgba, text_mask, metadata = rendered_text
    _, _, out_meta = layout_engine.apply_layout(
        text_rgba, text_mask, metadata, (256, 256), mode=mode
    )
    assert len(out_meta) == len(metadata)


@pytest.mark.parametrize(
    "mode", ["horizontal", "rotated", "curved", "perspective", "deformed"]
)
def test_layout_preserves_character_class_assignments(
    layout_engine: LayoutEngine, rendered_text, mode: str
) -> None:
    """Every metadata entry's class_id should still match its char."""
    text_rgba, text_mask, metadata = rendered_text
    _, _, out_meta = layout_engine.apply_layout(
        text_rgba, text_mask, metadata, (256, 256), mode=mode
    )
    char_map = char_to_class_id(1)
    for m in out_meta:
        assert m["class_id"] == char_map[m["char"]]


# ── Per-mode specifics ─────────────────────────────────────────────────────


def test_horizontal_no_rotation_artifacts(
    layout_engine: LayoutEngine, rendered_text
) -> None:
    """Horizontal mode shouldn't change text shape — mask pixel count
    should equal input pixel count when canvas is large enough."""
    text_rgba, text_mask, metadata = rendered_text
    in_count = (text_mask > 0).sum()
    _, out_mask, _ = layout_engine.apply_layout(
        text_rgba, text_mask, metadata, (512, 512), mode="horizontal"
    )
    assert (out_mask > 0).sum() == in_count


def test_rotation_at_180_inverts_centroid_order(
    layout_engine: LayoutEngine, rendered_text, monkeypatch
) -> None:
    """Rotating 180° should reverse the x-order of character centroids."""
    text_rgba, text_mask, metadata = rendered_text
    monkeypatch.setattr(
        "segocr.generator.layout.random.uniform", lambda a, b: 180.0
    )
    monkeypatch.setattr(
        "segocr.generator.layout.random.randint", lambda a, b: 0
    )
    _, _, out_meta = layout_engine.apply_layout(
        text_rgba, text_mask, metadata, (512, 512), mode="rotated"
    )
    in_x = [m["centroid"][0] for m in metadata]
    out_x = [m["centroid"][0] for m in out_meta]
    # Original was monotonic increasing; after 180° rotation should be
    # monotonic decreasing.
    assert in_x == sorted(in_x)
    assert out_x == sorted(out_x, reverse=True), (
        f"180° rotation should reverse char order, got {out_x}"
    )


def test_oversized_text_scaled_down(
    layout_engine: LayoutEngine, rendered_text
) -> None:
    """A text strip larger than the canvas must be scaled to fit."""
    text_rgba, text_mask, metadata = rendered_text
    # Force a tiny canvas
    _, out_mask, _ = layout_engine.apply_layout(
        text_rgba, text_mask, metadata, (32, 32), mode="horizontal"
    )
    assert out_mask.shape == (32, 32)
    # Mask should still contain *some* non-background pixels
    assert (out_mask > 0).sum() > 0


# ── Paragraph mode ─────────────────────────────────────────────────────────


def test_apply_paragraph_returns_canvas_sized(
    layout_engine: LayoutEngine, font_manager: FontManager
) -> None:
    renderer = CharacterRenderer(config={}, font_manager=font_manager, tier=1)
    font, _ = font_manager.sample_font(size=20)
    lines = [
        renderer.render_text(line, font, size=20)
        for line in ["First", "Second", "Third"]
    ]
    out_rgba, out_mask, out_meta = layout_engine.apply_paragraph(lines, (256, 256))
    assert out_rgba.shape == (256, 256, 4)
    assert out_mask.shape == (256, 256)
    # All chars from all lines should still be there
    assert len(out_meta) == sum(len(line[2]) for line in lines)
    # Each metadata entry should carry a line_index
    for m in out_meta:
        assert "line_index" in m
        assert 0 <= m["line_index"] < 3


def test_apply_paragraph_empty_input(layout_engine: LayoutEngine) -> None:
    rgba, mask, meta = layout_engine.apply_paragraph([], (64, 64))
    assert rgba.shape == (64, 64, 4)
    assert mask.shape == (64, 64)
    assert meta == []


def test_apply_paragraph_lines_stacked_vertically(
    layout_engine: LayoutEngine, font_manager: FontManager
) -> None:
    """Line-1 chars should have smaller y-centroids than line-2 chars."""
    renderer = CharacterRenderer(config={}, font_manager=font_manager, tier=1)
    font, _ = font_manager.sample_font(size=20)
    lines = [
        renderer.render_text("AAAA", font, size=20),
        renderer.render_text("BBBB", font, size=20),
    ]
    _, _, out_meta = layout_engine.apply_paragraph(lines, (256, 256))
    line0_y = [m["centroid"][1] for m in out_meta if m["line_index"] == 0]
    line1_y = [m["centroid"][1] for m in out_meta if m["line_index"] == 1]
    assert max(line0_y) < min(line1_y), (
        "Paragraph lines should be stacked top-to-bottom"
    )
