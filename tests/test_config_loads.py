"""Sanity test: default config loads, has all required top-level sections."""
from __future__ import annotations

from pathlib import Path

from segocr.utils.config import apply_overrides, load_config

CONFIG_PATH = Path(__file__).parent.parent / "segocr" / "configs" / "default.yaml"


def test_default_config_loads() -> None:
    config = load_config(CONFIG_PATH)
    for section in ("generator", "model", "training", "adaptation", "evaluation"):
        assert section in config, f"missing top-level section: {section}"


def test_layout_modes_sum_to_one() -> None:
    config = load_config(CONFIG_PATH)
    modes = config["generator"]["layout"]["modes"]
    assert abs(sum(modes.values()) - 1.0) < 1e-6


def test_background_tiers_sum_to_one() -> None:
    config = load_config(CONFIG_PATH)
    tiers = config["generator"]["background"]["tier_distribution"]
    assert abs(sum(tiers.values()) - 1.0) < 1e-6


def test_overrides_work() -> None:
    config = load_config(CONFIG_PATH)
    config = apply_overrides(config, ["model.num_classes=78"])
    assert config["model"]["num_classes"] == 78
