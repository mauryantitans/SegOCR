"""Evaluate a trained model on benchmarks.

Usage:
    .venv/Scripts/python -m scripts.evaluate \
        --config segocr/configs/default.yaml \
        --checkpoint weights/best_model.pth \
        --benchmark hard_cases
"""
from __future__ import annotations

import argparse
from pathlib import Path

from segocr.evaluation import run_benchmark
from segocr.utils.config import load_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("segocr/configs/default.yaml"))
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--benchmark", type=str, default=None,
                   help="Benchmark name; if omitted, runs all benchmarks in config.")
    p.add_argument("--output", type=Path, default=Path("evaluation_results"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    # TODO Week 11 — load model + checkpoint, iterate benchmarks
    raise NotImplementedError("scripts/evaluate.py — Week 11")


if __name__ == "__main__":
    main()
