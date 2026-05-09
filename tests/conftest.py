"""Shared pytest fixtures.

The generator tests need a real .ttf to render against. We don't ship
fonts (license + binary churn), so we sniff the host system's default
font directory and copy the first match into a tmp fixture dir. If no
suitable font exists, every test that depends on the fixture is skipped
rather than failing.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

SYSTEM_FONT_CANDIDATES = [
    # Windows
    Path("C:/Windows/Fonts/arial.ttf"),
    Path("C:/Windows/Fonts/calibri.ttf"),
    Path("C:/Windows/Fonts/segoeui.ttf"),
    # macOS
    Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
    Path("/Library/Fonts/Arial.ttf"),
    # Linux
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    Path("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"),
    Path("/usr/share/fonts/dejavu/DejaVuSans.ttf"),
]


def _find_system_font() -> Path | None:
    for p in SYSTEM_FONT_CANDIDATES:
        if p.exists():
            return p
    return None


@pytest.fixture(scope="session")
def system_font_path() -> Path:
    p = _find_system_font()
    if p is None:
        pytest.skip("No system font found for renderer tests.")
    return p


@pytest.fixture(scope="session")
def font_fixture_dir(tmp_path_factory, system_font_path: Path) -> Path:
    """A tiny fonts/ tree with two categorized copies of the system font."""
    fixture_dir = tmp_path_factory.mktemp("fonts")
    for cat in ("sans-serif", "serif"):
        cat_dir = fixture_dir / cat
        cat_dir.mkdir()
        shutil.copy(system_font_path, cat_dir / system_font_path.name)
    return fixture_dir


@pytest.fixture
def font_manager_config(font_fixture_dir: Path, tmp_path: Path) -> dict:
    return {
        "root_dir": str(font_fixture_dir),
        "cache_path": str(tmp_path / "font_cache.json"),
        "min_size": 16,
        "max_size": 64,
        "categories": {
            "serif": 0.5,
            "sans-serif": 0.5,
        },
    }


@pytest.fixture
def font_manager(font_manager_config):
    from segocr.generator.font_manager import FontManager

    return FontManager(font_manager_config)


@pytest.fixture
def text_sampler_config() -> dict:
    return {
        "corpus_path": None,
        "min_length": 3,
        "max_length": 20,
        "min_words_per_line": 1,
        "max_words_per_line": 5,
        "max_lines": 4,
        "case_distribution": {
            "lower": 0.4,
            "upper": 0.3,
            "mixed": 0.2,
            "title": 0.1,
        },
        "rare_char_boost": 2.0,
    }


@pytest.fixture
def text_sampler(text_sampler_config):
    from segocr.generator.text_sampler import TextSampler

    return TextSampler(text_sampler_config)


@pytest.fixture
def character_renderer(font_manager):
    from segocr.generator.renderer import CharacterRenderer

    return CharacterRenderer(config={}, font_manager=font_manager, tier=1)
