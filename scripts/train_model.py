"""Train a SegOCR model.

Usage:
    .venv/Scripts/python -m scripts.train_model --config segocr/configs/default.yaml
"""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import yaml

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
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Set torch + numpy + Python random seeds before training. "
             "Use to make multi-worker runs land in different basins of "
             "the loss landscape so their checkpoints average usefully. "
             "Also seeds DataLoader workers so num_workers > 0 stays "
             "reproducible.",
    )
    p.add_argument(
        "--reproducible",
        action="store_true",
        help="Enable bit-exact GPU determinism (cudnn.deterministic=True, "
             "benchmark=False). ~10–20%% slower; only enable when you "
             "specifically need bitwise reproducibility.",
    )
    p.add_argument("--override", nargs="*", default=[])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config = apply_overrides(config, args.override)

    if args.seed is not None:
        import random

        import numpy as np
        import torch
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        print(f"Seeded random / numpy / torch with seed={args.seed}")

    resume_from = args.resume
    if resume_from is None and args.resume_latest is not None:
        ckpts = sorted(Path(args.resume_latest).glob("checkpoint_*.pth"))
        if ckpts:
            resume_from = ckpts[-1]
            print(f"Auto-resuming from: {resume_from}")
        else:
            print(f"No checkpoints found in {args.resume_latest}; starting fresh.")

    # If --override was used, persist the overridden config to a temp YAML
    # so train() (which re-loads from a path) actually sees the overrides.
    config_path = args.config
    if args.override:
        tmp_path = Path(tempfile.mkstemp(suffix="_config.yaml")[1])
        tmp_path.write_text(yaml.safe_dump(config))
        config_path = tmp_path
        print(f"Applied {len(args.override)} override(s); using {config_path}")

    train(
        config_path,
        resume_from=resume_from,
        seed=args.seed,
        reproducible=args.reproducible,
    )


if __name__ == "__main__":
    main()
