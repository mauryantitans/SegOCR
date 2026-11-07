"""Self-training on unlabeled real images, starting from a trained checkpoint.

Each round pseudo-labels ``adaptation.self_training.unlabeled_real_dir``
with the current weights, then fine-tunes on synthetic + pseudo-labeled
data (see ``segocr/adaptation/self_training.py``).

Usage:
    python -m scripts.self_train \\
        --init weights/main_unet/averaged_best.pth \\
        --work-dir weights/self_train \\
        --override model.architecture=unet model.encoder=resnet18 \\
                   adaptation.self_training.unlabeled_real_dir=data/unlabeled_real
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from segocr.adaptation.self_training import SelfTrainer
from segocr.utils.config import apply_overrides, load_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("segocr/configs/default.yaml"))
    p.add_argument("--init", type=Path, required=True, help="Starting checkpoint .pth")
    p.add_argument("--work-dir", type=Path, required=True, help="Per-round outputs go here")
    p.add_argument("--iterations", type=int, default=None,
                   help="Rounds (default: adaptation.self_training.iterations)")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--override", nargs="*", default=[])
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    config = apply_overrides(load_config(args.config), args.override)
    iterations = args.iterations or int(config["adaptation"]["self_training"].get("iterations", 3))
    final = SelfTrainer(config).run(args.init, args.work_dir, iterations=iterations, seed=args.seed)
    print(f"Final weights: {final}")


if __name__ == "__main__":
    main()
