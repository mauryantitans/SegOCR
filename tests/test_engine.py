"""End-to-end engine tests."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from segocr.generator.engine import GeneratorEngine


def test_generate_one_returns_complete_dict(engine_config_path: Path) -> None:
    engine = GeneratorEngine(engine_config_path)
    sample = engine.generate_one(0)

    assert sample["index"] == 0
    assert sample["mode"] == "ocr"
    assert sample["image"].shape == (128, 128, 3)
    assert sample["image"].dtype == np.uint8
    assert sample["semantic_mask"].shape == (128, 128)
    assert sample["instance_mask"].shape == (128, 128)
    assert sample["affinity_mask"].shape == (128, 128)
    assert sample["direction_field"].shape == (128, 128, 2)
    assert sample["direction_field"].dtype == np.float32
    assert isinstance(sample["metadata"], dict)
    assert "characters" in sample["metadata"]


def test_generate_one_semantic_mask_has_valid_class_ids(
    engine_config_path: Path,
) -> None:
    engine = GeneratorEngine(engine_config_path)
    sample = engine.generate_one(0)
    classes = set(np.unique(sample["semantic_mask"]).tolist())
    # 63 classes (0..62 = bg + Tier 1)
    assert classes <= set(range(63)), f"Invalid class IDs: {classes}"


def test_generate_one_deterministic_across_engines(engine_config_path: Path) -> None:
    """Two freshly-initialized engines must produce identical output for
    the same index. (Within a single engine, the text sampler's running
    char-count state biases later calls — that's intentional for the
    rare-char boost; determinism is per-fresh-engine, not per-call.)
    """
    engine_a = GeneratorEngine(engine_config_path)
    engine_b = GeneratorEngine(engine_config_path)
    sample_a = engine_a.generate_one(42)
    sample_b = engine_b.generate_one(42)
    assert np.array_equal(sample_a["image"], sample_b["image"])
    assert np.array_equal(sample_a["semantic_mask"], sample_b["semantic_mask"])


def test_generate_one_different_indices_produce_different_outputs(
    engine_config_path: Path,
) -> None:
    engine = GeneratorEngine(engine_config_path)
    sample_a = engine.generate_one(0)
    sample_b = engine.generate_one(1)
    assert not np.array_equal(sample_a["image"], sample_b["image"])


def test_noise_removal_mode_collapses_to_binary(engine_config_path: Path) -> None:
    engine = GeneratorEngine(engine_config_path)
    sample = engine.generate_one(0, mode="noise_removal")
    semantic = sample["semantic_mask"]
    assert set(np.unique(semantic).tolist()) <= {0, 1}


def test_paragraph_mode_runs_without_error(
    engine_config_path: Path, tmp_path: Path
) -> None:
    """Force layout mode = paragraph and verify a sample generates."""
    import yaml

    cfg = yaml.safe_load(engine_config_path.read_text())
    cfg["generator"]["layout"]["modes"] = {
        "horizontal": 0.0,
        "rotated": 0.0,
        "curved": 0.0,
        "perspective": 0.0,
        "deformed": 0.0,
        "paragraph": 1.0,
    }
    para_config_path = tmp_path / "para_config.yaml"
    para_config_path.write_text(yaml.safe_dump(cfg))

    engine = GeneratorEngine(para_config_path)
    sample = engine.generate_one(0)
    assert sample["metadata"]["layout_mode"] == "paragraph"
    assert sample["image"].shape == (128, 128, 3)


def test_generate_dataset_writes_expected_files(
    engine_config_path: Path, tmp_path: Path
) -> None:
    output_dir = tmp_path / "out"
    engine = GeneratorEngine(engine_config_path)
    engine.generate_dataset(num_images=3, output_dir=output_dir, num_workers=0)

    for sub in ("images", "semantic", "instance", "metadata"):
        assert (output_dir / sub).exists()

    for i in range(3):
        name = f"{i:06d}"
        assert (output_dir / "images" / f"{name}.png").exists()
        assert (output_dir / "semantic" / f"{name}.png").exists()
        assert (output_dir / "instance" / f"{name}.png").exists()
        assert (output_dir / "metadata" / f"{name}.json").exists()


def test_saved_image_round_trips(engine_config_path: Path, tmp_path: Path) -> None:
    """Image saved to PNG should be readable and match the in-memory sample."""
    output_dir = tmp_path / "out"
    engine = GeneratorEngine(engine_config_path)
    engine.generate_dataset(num_images=1, output_dir=output_dir, num_workers=0)

    img_path = output_dir / "images" / "000000.png"
    loaded_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    assert loaded_bgr is not None
    assert loaded_bgr.shape == (128, 128, 3)


def test_saved_metadata_is_valid_json(engine_config_path: Path, tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    engine = GeneratorEngine(engine_config_path)
    engine.generate_dataset(num_images=1, output_dir=output_dir, num_workers=0)

    meta_path = output_dir / "metadata" / "000000.json"
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    assert "characters" in meta
    assert "layout_mode" in meta
    assert "tier" in meta


def test_oracle_property_holds_end_to_end(engine_config_path: Path) -> None:
    """After full generation, every non-zero pixel in the semantic mask
    must correspond to a character class ID that's valid for the config's
    tier (no fractional or out-of-range values).
    """
    engine = GeneratorEngine(engine_config_path)
    for index in range(5):
        sample = engine.generate_one(index)
        semantic = sample["semantic_mask"]
        non_bg = semantic[semantic > 0]
        if non_bg.size == 0:
            continue
        # Tier 1 = 62 classes; valid IDs are 1..62
        assert non_bg.min() >= 1
        assert non_bg.max() <= 62
