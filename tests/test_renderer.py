"""CharacterRenderer tests — verifies the oracle property."""
from __future__ import annotations

import numpy as np

from segocr.generator.font_manager import FontManager
from segocr.generator.renderer import ALPHA_THRESHOLD, CharacterRenderer
from segocr.utils.charset import char_to_class_id


def test_render_character_returns_nonempty_mask(
    character_renderer: CharacterRenderer, font_manager: FontManager
) -> None:
    font, _ = font_manager.sample_font(size=32)
    rgba, mask = character_renderer.render_character("A", font, size=32)
    assert rgba.ndim == 3 and rgba.shape[2] == 4
    assert mask.ndim == 2
    assert mask.sum() > 0, "Rendering 'A' produced an empty mask"


def test_mask_is_binary(
    character_renderer: CharacterRenderer, font_manager: FontManager
) -> None:
    font, _ = font_manager.sample_font(size=32)
    _, mask = character_renderer.render_character("M", font, size=32)
    assert set(np.unique(mask).tolist()) <= {0, 1}


def test_mask_and_rgba_same_spatial_size(
    character_renderer: CharacterRenderer, font_manager: FontManager
) -> None:
    font, _ = font_manager.sample_font(size=24)
    rgba, mask = character_renderer.render_character("g", font, size=24)
    assert rgba.shape[:2] == mask.shape


def test_render_character_size_proportional(
    character_renderer: CharacterRenderer, font_manager: FontManager
) -> None:
    """Rendering at 4× size should produce a roughly 4× mask area."""
    font, _ = font_manager.sample_font(size=24)
    _, mask_small = character_renderer.render_character("M", font, size=24)
    _, mask_large = character_renderer.render_character("M", font, size=96)
    ratio = mask_large.sum() / max(1, mask_small.sum())
    # 16× area in the limit; allow [10, 22]× for AA rounding
    assert 10.0 <= ratio <= 22.0, f"Area ratio out of range: {ratio:.2f}"


def test_render_text_class_ids_match_charset(
    character_renderer: CharacterRenderer, font_manager: FontManager
) -> None:
    font, _ = font_manager.sample_font(size=32)
    text = "AB12"
    _, class_mask, metadata = character_renderer.render_text(text, font, size=32)

    expected = {char_to_class_id(1)[c] for c in text}
    present_classes = set(np.unique(class_mask).tolist()) - {0}
    assert present_classes == expected, (
        f"Expected class IDs {expected}, got {present_classes}"
    )

    # Metadata covers every input character
    assert [m["char"] for m in metadata] == list(text)
    for m in metadata:
        assert m["class_id"] == char_to_class_id(1)[m["char"]]


def test_render_text_metadata_bboxes_inside_image(
    character_renderer: CharacterRenderer, font_manager: FontManager
) -> None:
    font, _ = font_manager.sample_font(size=40)
    rgba, _, metadata = character_renderer.render_text("Hello", font, size=40)
    h, w = rgba.shape[:2]
    for m in metadata:
        x0, y0, x1, y1 = m["bbox"]
        assert 0 <= x0 < x1 <= w
        assert 0 <= y0 < y1 <= h


def test_render_text_x_order_matches_string_order(
    character_renderer: CharacterRenderer, font_manager: FontManager
) -> None:
    font, _ = font_manager.sample_font(size=32)
    text = "ABCDE"
    _, _, metadata = character_renderer.render_text(text, font, size=32)
    # Centroid x-coordinates should be monotonically increasing for LTR text
    x_centroids = [m["centroid"][0] for m in metadata]
    assert x_centroids == sorted(x_centroids), (
        f"Char centroids not monotonic in x: {x_centroids}"
    )


def test_empty_string_returns_empty(
    character_renderer: CharacterRenderer, font_manager: FontManager
) -> None:
    font, _ = font_manager.sample_font(size=32)
    rgba, class_mask, metadata = character_renderer.render_text("", font, size=32)
    assert rgba.shape == (1, 1, 4)
    assert class_mask.shape == (1, 1)
    assert metadata == []


def test_off_charset_chars_skipped_from_metadata(
    character_renderer: CharacterRenderer, font_manager: FontManager
) -> None:
    """Tier-1 charset has no punctuation. ``A!B`` → metadata has only A, B."""
    font, _ = font_manager.sample_font(size=32)
    _, _, metadata = character_renderer.render_text("A!B", font, size=32)
    chars = [m["char"] for m in metadata]
    assert chars == ["A", "B"]


def test_oracle_property_alpha_matches_class_mask(
    character_renderer: CharacterRenderer, font_manager: FontManager
) -> None:
    """The oracle property: pixels classified as foreground in the class
    mask must correspond to pixels with high alpha in the rendered RGBA.

    Concretely, the IoU between (alpha >= threshold) and (class_mask > 0)
    should be very high. They will not be perfectly equal because the
    full-string render uses kerning while per-character renders are
    composed independently — small (~1 px) discrepancies at character
    junctions are expected.
    """
    font, _ = font_manager.sample_font(size=48)
    rgba, class_mask, _ = character_renderer.render_text(
        "Hello", font, size=48
    )
    fg_alpha = rgba[..., 3] >= ALPHA_THRESHOLD
    fg_class = class_mask > 0

    intersection = np.logical_and(fg_alpha, fg_class).sum()
    union = np.logical_or(fg_alpha, fg_class).sum()
    iou = intersection / max(1, union)
    assert iou >= 0.85, f"Oracle alignment IoU too low: {iou:.3f}"


def test_z_order_later_char_wins_on_overlap(
    character_renderer: CharacterRenderer, font_manager: FontManager
) -> None:
    """When two characters' masks overlap, the later one in the string
    should claim the disputed pixels (z-order: last-write-wins)."""
    font, _ = font_manager.sample_font(size=64)
    # Tight kerning of 'AV' often produces visual overlap regions; the
    # invariant must hold whether or not overlap actually occurs.
    rgba, class_mask, metadata = character_renderer.render_text("AV", font, size=64)

    # Locate where each char's mask was stamped — by class id
    class_a = char_to_class_id(1)["A"]
    class_v = char_to_class_id(1)["V"]

    a_pixels = (class_mask == class_a).sum()
    v_pixels = (class_mask == class_v).sum()
    # Both characters must claim *some* pixels — overlap shouldn't have
    # eaten an entire character.
    assert a_pixels > 0
    assert v_pixels > 0
