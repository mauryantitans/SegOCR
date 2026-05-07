"""Benchmark runner.

Drives evaluation of a trained model on the full benchmark set
(ICDAR2013/2015, Total-Text, COCO-Text, custom hard-cases) and emits a
comparison table against baselines (Tesseract / PaddleOCR / EasyOCR /
TrOCR / CRAFT+CRNN). Research Proposal §8.3–§8.4.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import nn


def run_benchmark(
    model: "nn.Module",
    benchmark_name: str,
    benchmark_path: str | Path,
    metrics: list[str],
    output_dir: str | Path,
) -> dict[str, float]:
    """Evaluate ``model`` on the named benchmark.

    Returns a metric→value dict and writes per-image predictions +
    confusion matrices into ``output_dir``.
    """
    raise NotImplementedError("run_benchmark — Week 11")
