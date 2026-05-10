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
    p.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume from a checkpoint .pth file (loads model/EMA/optimizer/iter).",
    )
    p.add_argument(
        "--resume-latest",
        type=Path,
        default=None,
        help="Resume from the latest checkpoint_*.pth in this directory, if any.",
    )
    p.add_argument("--override", nargs="*", default=[])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config = apply_overrides(config, args.override)

    resume_from = args.resume
    if resume_from is None and args.resume_latest is not None:
        ckpts = sorted(Path(args.resume_latest).glob("checkpoint_*.pth"))
        if ckpts:
            resume_from = ckpts[-1]
            print(f"Auto-resuming from: {resume_from}")
        else:
            print(f"No checkpoints found in {args.resume_latest}; starting fresh.")

    train(args.config, resume_from=resume_from)


if __name__ == "__main__":
    main()
