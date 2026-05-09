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


@pytest.fixture
def engine_config_path(tmp_path: Path, font_fixture_dir: Path) -> Path:
    """A fully-populated YAML config pointing at the fixture fonts and
    using a small image size so engine tests run fast.
    """
    import yaml

    config = {
        "generator": {
            "output_dir": str(tmp_path / "generated"),
            "image_size": [128, 128],
            "num_images": 4,
            "num_workers": 0,  # in-process for tests
            "fonts": {
                "root_dir": str(font_fixture_dir),
                "cache_path": str(tmp_path / "font_cache.json"),
                "min_size": 16,
                "max_size": 32,
                "categories": {"serif": 0.5, "sans-serif": 0.5},
            },
            "text": {
                "corpus_path": None,
                "min_length": 2,
                "max_length": 6,
                "min_words_per_line": 1,
                "max_words_per_line": 2,
                "max_lines": 2,
                "case_distribution": {
                    "lower": 0.4,
                    "upper": 0.3,
                    "mixed": 0.2,
                    "title": 0.1,
                },
                "rare_char_boost": 1.5,
            },
            "character_set": {"tier": 1},
            "layout": {
                "modes": {
                    "horizontal": 1.0,
                    "rotated": 0.0,
                    "curved": 0.0,
                    "perspective": 0.0,
                    "deformed": 0.0,
                    "paragraph": 0.0,
                },
                "rotation_range": [-30, 30],
                "curve_types": ["sinusoidal"],
                "perspective_strength": [0.05, 0.15],
                "deformation_strength": [0.05, 0.15],
                "paragraph": {
                    "lines": [2, 3],
                    "line_spacing": [1.1, 1.4],
                    "word_spacing": [0.6, 1.2],
                    "align": ["left"],
                },
            },
            "background": {
                "tier_distribution": {
                    "tier1_solid": 0.5,
                    "tier2_procedural": 0.5,
                    "tier3_natural": 0.0,
                    "tier4_adversarial": 0.0,
                },
                "natural_image_dirs": [],
                "preload_buffer_size": 8,
            },
            "compositing": {
                "modes": {
                    "standard": 0.7,
                    "semi_transparent": 0.1,
                    "textured_fill": 0.05,
                    "outline": 0.05,
                    "shadow": 0.05,
                    "emboss": 0.05,
                },
                "color_strategy": {
                    "contrast_aware": 0.5,
                    "random": 0.3,
                    "low_contrast": 0.2,
                },
            },
            "degradation": {
                "blur": {"probability": 0.2, "motion_kernel": [3, 7]},
                "noise": {"probability": 0.2, "gaussian_sigma": [5, 15]},
                "compression": {"probability": 0.3, "jpeg_quality": [50, 95]},
                "lighting": {
                    "probability": 0.3,
                    "gamma_range": [0.7, 1.3],
                    "brightness_shift": 0.2,
                    "contrast_factor": [0.8, 1.2],
                },
                "geometric": {"probability": 0.0, "distortion_k1": [-0.1, 0.1]},
                "occlusion": {
                    "probability": 0.0,
                    "max_patches": 2,
                    "max_coverage": 0.10,
                },
            },
        },
        "model": {"num_classes": 63, "input_size": [128, 128], "heads": {}, "loss": {}},
        "training": {},
        "adaptation": {},
        "evaluation": {"benchmarks": [], "metrics": ["miou"]},
    }
    config_path = tmp_path / "test_config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f)
    return config_path
