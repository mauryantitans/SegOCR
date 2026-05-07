"""Train a SegOCR model.

Usage:
    .venv/Scripts/python -m scripts.train_model --config segocr/configs/default.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

from segocr.training import train
from segocr.utils.config import apply_overrides, load_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("segocr/configs/default.yaml"))
    p.add_argument("--override", nargs="*", default=[])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config = apply_overrides(config, args.override)
    train(args.config)


if __name__ == "__main__":
    main()
