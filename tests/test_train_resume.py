"""Resume-from-checkpoint smoke test.

Trains for a few iterations, verifies a checkpoint was saved, then calls
train() with resume_from pointing at it — confirms the loop picks up at
the right iteration without crashing.

Kept short: 8 iterations on 4 samples. ~10 seconds on CPU.
"""
from __future__ import annotations

from pathlib import Path

import torch
import yaml

from segocr.generator.engine import GeneratorEngine
from segocr.training.train import train


def _build_train_config(
    engine_config_path: Path,
    tmp_path: Path,
    total_iters: int = 4,
) -> Path:
    """Take the engine fixture config and add training section."""
    cfg = yaml.safe_load(engine_config_path.read_text())
    cfg["training"] = {
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "total_iters": total_iters,
        "warmup_iters": 1,
        "eval_interval": max(1, total_iters // 2),
        "save_interval": max(1, total_iters // 2),
        "log_interval": 1,
        "batch_size": 2,
        "num_workers": 0,
        "mixed_precision": False,
        "output_dir": str(tmp_path / "weights"),
        "ema": {"enabled": False},
        "checkpoint_averaging": {"enabled": False},
        "keep_best_n": 2,
        "wandb": {"project": None},
    }
    # Tiny model for fast tests
    cfg["model"]["encoder"] = "resnet18"
    cfg["model"]["encoder_weights"] = None  # no ImageNet download
    cfg["model"]["head_features"] = 16
    cfg["model"]["decoder_channels"] = [64, 32, 16, 16, 16]
    cfg["model"]["heads"] = {"semantic": True, "affinity": False, "direction": False}

    train_config_path = tmp_path / "train_config.yaml"
    train_config_path.write_text(yaml.safe_dump(cfg))
    return train_config_path


def test_train_saves_checkpoint(engine_config_path: Path, tmp_path: Path) -> None:
    """A normal short run should write at least one checkpoint."""
    GeneratorEngine(engine_config_path).generate_dataset(
        num_images=4, output_dir=Path(yaml.safe_load(engine_config_path.read_text())["generator"]["output_dir"]),
        num_workers=0,
    )
    train_config = _build_train_config(engine_config_path, tmp_path, total_iters=4)
    train(train_config)

    weights_dir = tmp_path / "weights"
    checkpoints = list(weights_dir.glob("checkpoint_*.pth"))
    assert checkpoints, f"No checkpoints in {weights_dir}"


def test_train_resumes_from_checkpoint(
    engine_config_path: Path, tmp_path: Path
) -> None:
    """After resume_from, the loop continues without resetting iteration."""
    GeneratorEngine(engine_config_path).generate_dataset(
        num_images=4, output_dir=Path(yaml.safe_load(engine_config_path.read_text())["generator"]["output_dir"]),
        num_workers=0,
    )

    # First short run (4 iters) — produces a checkpoint
    train_config = _build_train_config(engine_config_path, tmp_path, total_iters=4)
    train(train_config)
    weights_dir = tmp_path / "weights"
    checkpoints = sorted(weights_dir.glob("checkpoint_*.pth"))
    assert checkpoints, "First run should have produced a checkpoint"
    first_ckpt = checkpoints[-1]
    first_iter = torch.load(first_ckpt, map_location="cpu")["iteration"]
    assert first_iter > 0

    # Second run resumes — total_iters bumped to 8 so we actually progress
    train_config_2 = _build_train_config(engine_config_path, tmp_path, total_iters=8)
    train(train_config_2, resume_from=first_ckpt)

    new_checkpoints = sorted(weights_dir.glob("checkpoint_*.pth"))
    assert new_checkpoints, "Second run should also write checkpoints"
    last_iter = torch.load(new_checkpoints[-1], map_location="cpu")["iteration"]
    assert last_iter > first_iter, (
        f"Resume should advance the iteration counter; "
        f"started at {first_iter}, ended at {last_iter}"
    )


def test_train_writes_run_manifest(
    engine_config_path: Path, tmp_path: Path
) -> None:
    """A training run should drop a run_manifest.json in the output dir
    with seed, config snapshot, hardware info."""
    import json as _json

    GeneratorEngine(engine_config_path).generate_dataset(
        num_images=4,
        output_dir=Path(yaml.safe_load(engine_config_path.read_text())["generator"]["output_dir"]),
        num_workers=0,
    )
    train_config = _build_train_config(engine_config_path, tmp_path, total_iters=2)
    train(train_config, seed=42)

    manifest_path = tmp_path / "weights" / "run_manifest.json"
    assert manifest_path.exists(), "run_manifest.json should be written"
    manifest = _json.loads(manifest_path.read_text())

    assert manifest["seed"] == 42
    assert "config" in manifest
    assert "torch_version" in manifest
    assert "timestamp" in manifest
    # device key is present even on CPU
    assert "device" in manifest


def test_train_resume_from_missing_path_raises(
    engine_config_path: Path, tmp_path: Path
) -> None:
    """A bogus resume path should error early, not silently start fresh."""
    GeneratorEngine(engine_config_path).generate_dataset(
        num_images=4, output_dir=Path(yaml.safe_load(engine_config_path.read_text())["generator"]["output_dir"]),
        num_workers=0,
    )
    train_config = _build_train_config(engine_config_path, tmp_path, total_iters=4)
    bogus_path = tmp_path / "does_not_exist.pth"
    try:
        train(train_config, resume_from=bogus_path)
    except FileNotFoundError:
        return
    raise AssertionError("Expected FileNotFoundError for missing resume_from path")
