"""Generate a synthetic SegOCR training dataset.

Usage:
    .venv/Scripts/python -m scripts.generate_dataset \
        --config segocr/configs/default.yaml \
        --num-images 10000 \
        --output data/generated \
        --mode ocr
"""
from __future__ import annotations

import argparse
from pathlib import Path

from segocr.generator import GeneratorEngine
from segocr.utils.config import apply_overrides, load_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("segocr/configs/default.yaml"))
    p.add_argument("--num-images", type=int, default=None)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument(
        "--mode",
        choices=["ocr", "noise_removal"],
        default="ocr",
        help="ocr → multi-class masks; noise_removal → binary char/bg",
    )
    p.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="key.path=value YAML overrides",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config = apply_overrides(config, args.override)
    if args.num_images is not None:
        config["generator"]["num_images"] = args.num_images
    if args.output is not None:
        config["generator"]["output_dir"] = str(args.output)

    engine = GeneratorEngine(args.config)
    engine.generate_dataset(
        num_images=config["generator"]["num_images"],
        output_dir=config["generator"]["output_dir"],
        mode=args.mode,
    )


if __name__ == "__main__":
    main()
