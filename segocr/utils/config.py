"""YAML config loading + override helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def apply_overrides(config: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply ``--override key.path=value`` style CLI overrides.

    Values are parsed as YAML so ints / floats / bools / lists work.
    """
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid override (missing '='): {override}")
        key_path, value_str = override.split("=", 1)
        value = yaml.safe_load(value_str)
        keys = key_path.split(".")
        target = config
        for key in keys[:-1]:
            target = target.setdefault(key, {})
        target[keys[-1]] = value
    return config
