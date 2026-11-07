"""SegOCRDataset + collate_fn tests.

Uses the engine fixture to generate a small dataset on disk, then loads
it back through the Dataset to verify the round-trip + augmentation.
"""
from __future__ import annotations

from pathlib import Path

import torch

from segocr.generator.engine import GeneratorEngine
from segocr.training.dataset import SegOCRDataset, collate_fn


def _make_dataset(engine_config_path: Path, output_dir: Path, n: int = 4) -> Path:
    engine = GeneratorEngine(engine_config_path)
    engine.generate_dataset(num_images=n, output_dir=output_dir, num_workers=0)
    return output_dir


def test_dataset_loads_generated_samples(
    engine_config_path: Path, tmp_path: Path
) -> None:
    data_dir = _make_dataset(engine_config_path, tmp_path / "ds", n=8)
    dataset = SegOCRDataset(data_dir, split="train", train_aug=False)
    assert len(dataset) > 0


def test_getitem_returns_expected_keys(
    engine_config_path: Path, tmp_path: Path
) -> None:
    data_dir = _make_dataset(engine_config_path, tmp_path / "ds", n=4)
    dataset = SegOCRDataset(data_dir, split="train", train_aug=False)
    sample = dataset[0]
    assert "image" in sample
    assert "targets" in sample
    assert "metadata" in sample


def test_getitem_image_shape_and_dtype(
    engine_config_path: Path, tmp_path: Path
) -> None:
    data_dir = _make_dataset(engine_config_path, tmp_path / "ds", n=4)
    dataset = SegOCRDataset(data_dir, split="train", train_aug=False)
    sample = dataset[0]
    image = sample["image"]
    assert image.shape == (3, 128, 128)
    assert image.dtype == torch.float32


def test_targets_have_correct_shapes(
    engine_config_path: Path, tmp_path: Path
) -> None:
    data_dir = _make_dataset(engine_config_path, tmp_path / "ds", n=4)
    dataset = SegOCRDataset(data_dir, split="train", train_aug=False)
    sample = dataset[0]
    targets = sample["targets"]
    assert targets["semantic"].shape == (128, 128)
    assert targets["semantic"].dtype == torch.long
    assert targets["affinity"].shape == (128, 128)
    assert targets["direction"].shape == (2, 128, 128)
    assert targets["direction"].dtype == torch.float32


def test_subset_targets_skips_unrequested(
    engine_config_path: Path, tmp_path: Path
) -> None:
    data_dir = _make_dataset(engine_config_path, tmp_path / "ds", n=4)
    dataset = SegOCRDataset(
        data_dir, split="train", train_aug=False, return_targets=("semantic",)
    )
    sample = dataset[0]
    assert "semantic" in sample["targets"]
    assert "affinity" not in sample["targets"]
    assert "direction" not in sample["targets"]


def test_train_val_split_disjoint(
    engine_config_path: Path, tmp_path: Path
) -> None:
    data_dir = _make_dataset(engine_config_path, tmp_path / "ds", n=20)
    train_ds = SegOCRDataset(data_dir, split="train", val_fraction=0.25)
    val_ds = SegOCRDataset(data_dir, split="val", val_fraction=0.25)
    train_names = {p.stem for p in train_ds.image_paths}
    val_names = {p.stem for p in val_ds.image_paths}
    assert not train_names & val_names
    assert len(train_names) + len(val_names) == 20


def test_collate_fn_stacks_batch(
    engine_config_path: Path, tmp_path: Path
) -> None:
    data_dir = _make_dataset(engine_config_path, tmp_path / "ds", n=4)
    dataset = SegOCRDataset(data_dir, split="train", train_aug=False)
    samples = [dataset[i] for i in range(min(2, len(dataset)))]
    batch = collate_fn(samples)
    assert batch["image"].shape == (len(samples), 3, 128, 128)
    assert batch["targets"]["semantic"].shape == (len(samples), 128, 128)
    assert isinstance(batch["metadata"], list)
    assert len(batch["metadata"]) == len(samples)


def test_image_normalized_to_imagenet_stats(
    engine_config_path: Path, tmp_path: Path
) -> None:
    """Loaded images should have channel means roughly in standard-normal range."""
    data_dir = _make_dataset(engine_config_path, tmp_path / "ds", n=8)
    dataset = SegOCRDataset(data_dir, split="train", train_aug=False)
    means = []
    for i in range(min(4, len(dataset))):
        means.append(dataset[i]["image"].mean(dim=(1, 2)))
    avg_mean = torch.stack(means).mean(dim=0).abs().max()
    # ImageNet-normalized random images have means in roughly [-2, 2]
    assert avg_mean < 5.0


def test_train_aug_changes_output(
    engine_config_path: Path, tmp_path: Path
) -> None:
    """Training augmentation should produce different outputs across calls."""
    data_dir = _make_dataset(engine_config_path, tmp_path / "ds", n=4)
    dataset = SegOCRDataset(data_dir, split="train", train_aug=True)
    first = dataset[0]["image"]
    # Each call is unaugmented with small probability; 5 calls → ~never all equal
    assert any(not torch.equal(first, dataset[0]["image"]) for _ in range(5))


def test_train_aug_never_changes_targets(
    engine_config_path: Path, tmp_path: Path
) -> None:
    """No geometric aug: a mirrored glyph would keep a now-wrong class label
    and desync the metadata-derived affinity / direction targets."""
    data_dir = _make_dataset(engine_config_path, tmp_path / "ds", n=4)
    plain = SegOCRDataset(data_dir, split="train", train_aug=False)[0]["targets"]
    aug_ds = SegOCRDataset(data_dir, split="train", train_aug=True)
    for _ in range(10):
        targets = aug_ds[0]["targets"]
        for key in ("semantic", "affinity", "direction"):
            assert torch.equal(targets[key], plain[key])


def test_train_val_split_stable_across_processes(tmp_path: Path) -> None:
    """The split must not depend on PYTHONHASHSEED, or --resume leaks val."""
    import os
    import subprocess
    import sys

    (tmp_path / "images").mkdir()
    for i in range(200):
        (tmp_path / "images" / f"{i:06d}.png").touch()
    code = (
        "import sys; from segocr.training.dataset import SegOCRDataset as D; "
        "print(sorted(p.stem for p in D(sys.argv[1], split='val').image_paths))"
    )
    outputs = set()
    for hash_seed in ("1", "2"):
        env = {**os.environ, "PYTHONHASHSEED": hash_seed}
        result = subprocess.run(
            [sys.executable, "-c", code, str(tmp_path)],
            env=env, capture_output=True, text=True, check=True,
        )
        outputs.add(result.stdout)
    assert len(outputs) == 1


def test_fda_stylizes_image_but_not_masks(
    engine_config_path: Path, tmp_path: Path
) -> None:
    import cv2
    import numpy as np

    data_dir = _make_dataset(engine_config_path, tmp_path / "ds", n=4)
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    cv2.imwrite(str(real_dir / "a.jpg"), np.full((64, 64, 3), (0, 0, 255), np.uint8))

    kwargs = {"fda_target_dir": real_dir, "fda_probability": 1.0, "fda_beta": 0.05}
    # Val / no-aug datasets must never be stylized.
    assert SegOCRDataset(data_dir, split="train", train_aug=False, **kwargs).fda_target_paths == []

    plain = SegOCRDataset(data_dir, split="train", train_aug=False)
    fda = SegOCRDataset(data_dir, split="train", train_aug=True, **kwargs)
    assert fda.fda_target_paths == [real_dir / "a.jpg"]
    # Disable flip/jitter so the only difference is FDA.
    fda._apply_augmentations = lambda img, sem, inst: (img, sem, inst)
    a, b = plain[0], fda[0]
    assert torch.equal(a["targets"]["semantic"], b["targets"]["semantic"])
    assert not torch.equal(a["image"], b["image"])
