"""Tests for the multi-account ensemble path: index_offset + average_runs."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from scripts.average_runs import average_states
from segocr.generator.engine import GeneratorEngine


def test_index_offset_produces_disjoint_data(
    engine_config_path: Path, tmp_path: Path
) -> None:
    """Two workers with different index_offsets should produce different
    images for the same nominal index slot."""
    engine = GeneratorEngine(engine_config_path)
    out_a = tmp_path / "worker_a"
    out_b = tmp_path / "worker_b"
    engine.generate_dataset(
        num_images=2, output_dir=out_a, num_workers=0, index_offset=0
    )
    engine.generate_dataset(
        num_images=2, output_dir=out_b, num_workers=0, index_offset=100
    )

    # Worker A wrote 000000.png, 000001.png; Worker B wrote 000100.png, 000101.png.
    # Test that worker B did NOT produce file 000000 (proves indices are different).
    assert (out_a / "images" / "000000.png").exists()
    assert (out_b / "images" / "000100.png").exists()
    assert not (out_b / "images" / "000000.png").exists()
    assert not (out_a / "images" / "000100.png").exists()


def test_index_offset_seeds_distinctly(
    engine_config_path: Path, tmp_path: Path
) -> None:
    """Same nominal slot but different offsets must produce different images
    (proves seeding uses the absolute index, not the relative one)."""
    import cv2

    engine = GeneratorEngine(engine_config_path)
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    # Both produce file 000000.png but seeded at index 0 vs 100
    engine.generate_dataset(num_images=1, output_dir=out_a, num_workers=0, index_offset=0)
    # Re-init engine (so its text-sampler counter resets)
    engine_b = GeneratorEngine(engine_config_path)
    engine_b.generate_dataset(
        num_images=1, output_dir=out_b, num_workers=0, index_offset=100
    )

    img_a = cv2.imread(str(out_a / "images" / "000000.png"))
    img_b = cv2.imread(str(out_b / "images" / "000100.png"))
    assert not np.array_equal(img_a, img_b), (
        "Different index offsets should produce different images"
    )


def test_average_states_matches_manual_mean() -> None:
    """The averaging math should equal a hand-rolled mean over float tensors."""
    state_a = {
        "w1": torch.tensor([1.0, 2.0, 3.0]),
        "w2": torch.tensor([[10.0]]),
        "step": torch.tensor(5),  # int — should be passed through
    }
    state_b = {
        "w1": torch.tensor([3.0, 4.0, 5.0]),
        "w2": torch.tensor([[20.0]]),
        "step": torch.tensor(5),
    }
    state_c = {
        "w1": torch.tensor([5.0, 6.0, 7.0]),
        "w2": torch.tensor([[30.0]]),
        "step": torch.tensor(5),
    }
    avg = average_states([state_a, state_b, state_c])
    assert torch.allclose(avg["w1"], torch.tensor([3.0, 4.0, 5.0]))
    assert torch.allclose(avg["w2"], torch.tensor([[20.0]]))
    # Int tensor passed through verbatim
    assert avg["step"].item() == 5


def test_average_runs_script_writes_loadable_checkpoint(tmp_path: Path) -> None:
    """End-to-end: write fake worker checkpoints, run the script,
    verify the output has the expected structure."""
    import subprocess
    import sys

    # Write three fake checkpoints
    ckpt_paths: list[Path] = []
    for i in range(3):
        path = tmp_path / f"worker{i}_best.pth"
        torch.save(
            {"model": {"layer.weight": torch.full((4,), float(i))}, "iteration": -1},
            path,
        )
        ckpt_paths.append(path)

    output = tmp_path / "ensemble.pth"
    cmd = [
        sys.executable,
        "-m",
        "scripts.average_runs",
        "--checkpoints",
        *(str(p) for p in ckpt_paths),
        "--output",
        str(output),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"Script failed: {result.stderr}"
    assert output.exists()

    loaded = torch.load(output, map_location="cpu")
    assert "model" in loaded
    assert "ema" in loaded  # script writes both keys
    assert loaded["n_runs"] == 3
    # Average of 0, 1, 2 = 1.0
    assert torch.allclose(loaded["model"]["layer.weight"], torch.full((4,), 1.0))
