"""FontManager tests."""
from __future__ import annotations

from pathlib import Path

from PIL.ImageFont import FreeTypeFont

from segocr.generator.font_manager import FontManager


def test_loads_validated_fonts(font_manager: FontManager) -> None:
    assert font_manager.num_fonts >= 1


def test_sample_returns_valid_font_and_category(font_manager: FontManager) -> None:
    font, category = font_manager.sample_font(size=32)
    assert isinstance(font, FreeTypeFont)
    # The fixture only places fonts under serif/ and sans-serif/.
    assert category in {"serif", "sans-serif"}


def test_sample_respects_size_argument(font_manager: FontManager) -> None:
    font, _ = font_manager.sample_font(size=24)
    # FreeTypeFont exposes size as the second tuple element of getmetrics()
    # context — the simplest invariant is that it actually rendered at the
    # requested size, which we verify by bbox height being in a sensible
    # range proportional to size.
    bbox = font.getbbox("M")
    assert bbox is not None
    height = bbox[3] - bbox[1]
    assert 8 <= height <= 48  # 24px nominal — Arial 'M' is ~17px tall


def test_sample_size_in_configured_range(font_manager: FontManager) -> None:
    # When size is None, it should be drawn from [min_size, max_size]
    for _ in range(20):
        font, _ = font_manager.sample_font()
        bbox = font.getbbox("M")
        assert bbox is not None
        height = bbox[3] - bbox[1]
        # M-cap-height is roughly 0.7 * font size; min_size=16, max_size=64
        # → cap height roughly in [10, 50] but with rendering padding could be wider
        assert 4 <= height <= 80


def test_get_char_bbox_returns_4tuple(font_manager: FontManager) -> None:
    fonts = font_manager.get_all_fonts()
    assert fonts, "No validated fonts to test against"
    _, font_path = fonts[0]
    bbox = font_manager.get_char_bbox(font_path, "A", size=48)
    assert isinstance(bbox, tuple) and len(bbox) == 4
    x0, y0, x1, y1 = bbox
    assert x1 > x0 and y1 > y0


def test_get_char_bbox_scales_with_size(font_manager: FontManager) -> None:
    _, font_path = font_manager.get_all_fonts()[0]
    bbox_small = font_manager.get_char_bbox(font_path, "M", size=24)
    bbox_large = font_manager.get_char_bbox(font_path, "M", size=96)
    h_small = bbox_small[3] - bbox_small[1]
    h_large = bbox_large[3] - bbox_large[1]
    # 4× size → ~4× height (allow ±20% for rounding/font metrics)
    ratio = h_large / max(1, h_small)
    assert 3.0 <= ratio <= 5.0


def test_cache_persists_across_instances(font_manager_config: dict) -> None:
    fm1 = FontManager(font_manager_config)
    n1 = fm1.num_fonts
    cache_path = Path(font_manager_config["cache_path"])
    assert cache_path.exists()

    # Second instance should populate from the cache without rescanning
    fm2 = FontManager(font_manager_config)
    assert fm2.num_fonts == n1
    assert fm2.bbox_cache  # populated from disk


def test_handles_missing_root_dir(tmp_path: Path) -> None:
    config = {
        "root_dir": str(tmp_path / "does_not_exist"),
        "cache_path": str(tmp_path / "cache.json"),
        "min_size": 16,
        "max_size": 64,
        "categories": {"sans-serif": 1.0},
    }
    fm = FontManager(config)
    assert fm.num_fonts == 0


def test_categorize_assigns_to_existing_category(font_manager: FontManager) -> None:
    # Every loaded font should be in one of the configured categories
    for cat, paths in font_manager.fonts_by_category.items():
        if paths:
            assert cat in font_manager.category_weights
