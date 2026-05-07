"""Smoke test — every package + module imports without error.

Modules requiring torch are skipped when torch isn't installed (Phase 1
environment) and re-enabled automatically in Phase 2.
"""
from __future__ import annotations

import importlib
import importlib.util

import pytest

# No torch / no segmentation_models_pytorch needed.
PHASE1_MODULES = [
    "segocr",
    "segocr.utils",
    "segocr.utils.charset",
    "segocr.utils.config",
    "segocr.generator.font_manager",
    "segocr.generator.text_sampler",
    "segocr.generator.renderer",
    "segocr.generator.layout",
    "segocr.generator.background",
    "segocr.generator.compositor",
    "segocr.generator.degradation",
    "segocr.generator.placement",
    "segocr.generator.engine",
    "segocr.generator",  # __init__ imports the above
    "segocr.evaluation.metrics",
    "segocr.evaluation.visualize",
]

# Importable only once torch (and friends) are installed.
PHASE2_MODULES = [
    "segocr.models",
    "segocr.models.heads",
    "segocr.models.losses",
    "segocr.models.segformer",
    "segocr.models.unet",
    "segocr.training",
    "segocr.training.dataset",
    "segocr.training.train",
    "segocr.training.evaluator",
    "segocr.adaptation",
    "segocr.adaptation.cyclegan",
    "segocr.adaptation.dann",
    "segocr.adaptation.fda",
    "segocr.adaptation.self_training",
    "segocr.postprocessing",
    "segocr.postprocessing.cleanup",
    "segocr.postprocessing.instance_extraction",
    "segocr.postprocessing.reading_order",
    "segocr.evaluation",
    "segocr.evaluation.benchmark",
]

_HAS_TORCH = importlib.util.find_spec("torch") is not None


@pytest.mark.parametrize("module_name", PHASE1_MODULES)
def test_phase1_module_importable(module_name: str) -> None:
    importlib.import_module(module_name)


@pytest.mark.skipif(not _HAS_TORCH, reason="Phase 2 module: torch not installed")
@pytest.mark.parametrize("module_name", PHASE2_MODULES)
def test_phase2_module_importable(module_name: str) -> None:
    importlib.import_module(module_name)
