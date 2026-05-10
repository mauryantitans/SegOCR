# SegOCR — Usage Guide

Step-by-step instructions for installing and running SegOCR on Windows, Linux/macOS, and Google Colab.

> **What is SegOCR?** A character-level semantic segmentation reformulation of OCR. Every pixel is classified as background or one of N character classes; detection and recognition happen in a single inference pass. See `SegOCR_Research_Proposal.docx` for the full paper outline.

---

## Table of contents

1. [Quickstart on Google Colab](#1-quickstart-on-google-colab) (no local GPU required)
2. [Local install (Windows / Linux / macOS)](#2-local-install)
3. [Generating a synthetic dataset](#3-generating-a-synthetic-dataset)
4. [Training a model](#4-training-a-model)
5. [Evaluating + visualizing predictions](#5-evaluating--visualizing-predictions)
6. [Configuration reference](#6-configuration-reference)
7. [Common workflows](#7-common-workflows)
8. [Troubleshooting](#8-troubleshooting)

---

## 1 · Quickstart on Google Colab

You don't need a local GPU. Open the bundled notebook in Colab:

> [`notebooks/segocr_colab_quickstart.ipynb`](../notebooks/segocr_colab_quickstart.ipynb)

To run it:
1. Push this repo to your GitHub (or use any fork).
2. Visit https://colab.research.google.com → **File → Open Notebook → GitHub** → paste the notebook URL.
3. **Runtime → Change runtime type → T4 GPU** (free) or A100 (paid).
4. Run all cells in order. Total wall time: ~10–15 minutes on T4 for an end-to-end demo (generate 500 samples, train 500 iterations, visualize predictions).

The notebook handles:
- Cloning the repo
- Installing dependencies (Colab already has PyTorch + CUDA)
- Generating a small dataset using bundled corpora and Linux system fonts (no Google Fonts download needed for the demo)
- Training a UNet+ResNet18 baseline
- Visualizing predictions on held-out validation samples

For a full-scale experiment on Colab:
- Use a paid runtime (A100/L4) with persistent storage.
- Run `bash scripts/setup_data.sh` to download Google Fonts + COCO + DTD (~25 GB).
- Increase `--num-images` to 100K and `total_iters` to 80K+.

---

## 2 · Local install

### 2.1 Prerequisites

- Python 3.10+
- Git
- (Optional) NVIDIA GPU with CUDA 11.8+ for training. CPU-only is fine for data generation but training is ~50× slower.

### 2.2 Windows (PowerShell)

```powershell
# Clone
git clone https://github.com/mauryantitans/SegOCR.git
cd segocr

# Create + activate venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install Phase 1 deps (data generation only)
pip install -r requirements\base.txt
pip install -e .

# Phase 2 deps for training
# CPU build (no GPU):
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cpu
# OR CUDA 11.8 build (with GPU):
# pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
pip install segmentation-models-pytorch wandb

# Verify
pytest tests/ -q
```

### 2.3 Linux / macOS

```bash
git clone https://github.com/mauryantitans/SegOCR.git
cd segocr

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements/base.txt
pip install -e .

# CUDA build:
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
pip install segmentation-models-pytorch wandb

pytest tests/ -q
```

### 2.4 Optional — download datasets

For full-scale training you need Google Fonts (~500 MB), COCO (~18 GB), DTD (~600 MB), and a text corpus. The bundled mini-corpora in `segocr/assets/corpora/` and your OS system fonts are enough for *quick experiments*; larger external datasets are needed for production-quality results.

```bash
# Linux / macOS / Colab
bash scripts/setup_data.sh
```

```powershell
# Windows
.\scripts\setup_data.ps1
```

Manual benchmark downloads (require registration):
- ICDAR 2013/2015 — https://rrc.cvc.uab.es/
- COCO-Text — https://bgshih.github.io/cocotext/
- Total-Text — https://github.com/cs-chan/Total-Text-Dataset

---

## 3 · Generating a synthetic dataset

The data generator is the **oracle**: every character is rendered programmatically, so the alpha channel of the rendered text *is* the pixel-perfect ground-truth segmentation mask. No manual annotation, no labelling tool.

### 3.1 Quickstart command

```bash
python -m scripts.generate_dataset \
    --config segocr/configs/default.yaml \
    --num-images 1000 \
    --output data/dev_run \
    --override generator.fonts.root_dir=/usr/share/fonts \
                generator.background.natural_image_dirs=[] \
                generator.num_workers=4
```

On Windows:
```powershell
python -m scripts.generate_dataset `
    --config segocr\configs\default.yaml `
    --num-images 1000 `
    --output data\dev_run `
    --override generator.fonts.root_dir=C:/Windows/Fonts `
                generator.background.natural_image_dirs=[] `
                generator.num_workers=4
```

### 3.2 Output format

```
data/dev_run/
├── images/      000000.png … 000999.png    # RGB
├── semantic/    000000.png … 000999.png    # uint8 class IDs (0=bg, 1..62 chars)
├── instance/    000000.png … 000999.png    # uint16 unique character instance IDs
└── metadata/    000000.json … 000999.json  # per-character metadata + generation params
```

Affinity (word-grouping) and direction (vector-to-centroid) targets are *not* saved to disk — they're cheap to recompute from `(semantic, instance, metadata)` at training time, saving ~1MB per sample.

### 3.3 Modes

The generator has two modes selectable via `--mode`:
- **`ocr`** (default) — multi-class semantic mask with class IDs 1..62.
- **`noise_removal`** — collapses classes 1..62 → 1 for the binary char-vs-background formulation. Useful when you only want text-as-noise removal, not character recognition.

### 3.4 What you get out of the box

Even without external datasets, the generator produces useful samples:
- **Fonts**: pulled from your OS system font directory (`C:/Windows/Fonts`, `/usr/share/fonts`, `/Library/Fonts`).
- **Text**: bundled mini-corpora (`signs`, `receipts`, `names`, `numbers`) — produces real words like "EXIT", "TOTAL", "SAMUEL", "10015" instead of random `ABCDEFG`.
- **Backgrounds**: tier-1 (solid/gradient) + tier-2 (procedural Perlin noise) auto-fall-back when no natural images are configured.

For paper-quality data, run `setup_data.sh` to enable tier-3 (COCO) and tier-4 (adversarial) backgrounds.

### 3.5 Layout modes

Six layout modes control how text is arranged spatially:

| Mode | Description | Default weight |
|---|---|---|
| `horizontal` | Plain left-to-right | 0.30 |
| `rotated` | Block rotated by random angle | 0.18 |
| `curved` | Sinusoidal/circular/bezier displacement | 0.18 |
| `perspective` | 4-point perspective warp | 0.13 |
| `deformed` | Elastic deformation (TPS-style) | 0.08 |
| `paragraph` | Multi-line document layout | 0.13 |

Tune via `generator.layout.modes` in the YAML config.

### 3.6 Smart placement

When `generator.layout.placement.realistic_fraction > 0`, that fraction of samples uses **edge-density saliency** to place text on flat, low-clutter regions (less likely to be obscured). The remainder uses uniform-random placement to preserve the **diversity** that domain randomization adaptation depends on. Default split: 30% saliency / 70% random.

---

## 4 · Training a model

### 4.1 Quickstart

After generating a dataset, train a UNet baseline:

```bash
python -m scripts.train_model \
    --config segocr/configs/default.yaml \
    --override generator.output_dir=data/dev_run \
                model.architecture=unet \
                model.encoder=resnet18 \
                training.total_iters=2000 \
                training.batch_size=8 \
                training.eval_interval=200 \
                training.output_dir=weights/dev_run
```

### 4.2 Architectures available

- **`unet`** (default) — UNet with a `segmentation_models_pytorch` encoder. Encoders supported: `resnet18`, `resnet34`, `resnet50`, `resnet101`, `efficientnet-b0..b7`, etc.
- **`segformer`** — MiT-B2 backbone, requires `mmsegmentation`. See `requirements/train.txt` for install instructions.

### 4.3 Multi-head outputs

Each architecture exposes three heads:

| Head | Output | Loss |
|---|---|---|
| `semantic` | `(B, num_classes, H, W)` | Focal CE (γ=2) + Dice |
| `affinity` | `(B, 1, H, W)` | BCE w/ pos-weight |
| `direction` | `(B, 2, H, W)` | SmoothL1 (foreground-only) |

Toggle individual heads via `model.heads.{semantic, affinity, direction}` in the config.

### 4.4 Stability mitigations

The training loop includes two mitigations for the **across-epoch instability** observed in the pilot study:

- **EMA of weights** (`training.ema.enabled = true`, `decay = 0.999`) — exponentially-averaged model weights, evaluated separately from the live model.
- **Top-N checkpoint averaging** (`training.checkpoint_averaging.enabled = true`, `top_n = 3`) — at end of training, average the top 3 checkpoints by validation mIoU.

Per the design note, try EMA + averaging first. Only investigate Nested Learning if both are insufficient.

### 4.5 What's logged

Every `eval_interval` iterations:
- Three mIoU variants:
  - `miou` — mean over all classes including background.
  - `fg_miou` — mean over foreground (non-background) classes only — the true OCR signal.
  - `binary_miou` — collapses `1..N → 1` — sanity floor and the noise-removal use-case deliverable.
- Per-class IoU (`iou_class_00` through `iou_class_62`) — surfaces rare-character regression that mean metrics hide.

When `training.wandb.project` is set, all these go to Weights & Biases. Otherwise they're logged to stdout.

---

## 5 · Evaluating + visualizing predictions

```bash
python -m scripts.evaluate \
    --config segocr/configs/default.yaml \
    --checkpoint weights/dev_run/checkpoint_002000.pth \
    --benchmark hard_cases
```

For programmatic visualization see the last cell of [`notebooks/segocr_colab_quickstart.ipynb`](../notebooks/segocr_colab_quickstart.ipynb), or roll your own:

```python
import torch
from segocr.models.unet import build_model
from segocr.training.dataset import SegOCRDataset
from segocr.utils.config import load_config

cfg = load_config('segocr/configs/default.yaml')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = build_model(cfg['model']).to(device).eval()
state = torch.load('weights/dev_run/checkpoint_002000.pth', map_location=device)
model.load_state_dict(state['model'])

val_ds = SegOCRDataset('data/dev_run', split='val', train_aug=False)
sample = val_ds[0]
with torch.no_grad():
    out = model(sample['image'].unsqueeze(0).to(device))
prediction = out['semantic'].argmax(dim=1).cpu().numpy()[0]
```

---

## 6 · Configuration reference

Every parameter lives in one YAML file. The default is [`segocr/configs/default.yaml`](../segocr/configs/default.yaml).

Override any value at the CLI without editing the file:

```bash
python -m scripts.train_model \
    --config segocr/configs/default.yaml \
    --override training.batch_size=16 \
                training.learning_rate=1e-4 \
                model.encoder=resnet50
```

Top-level sections:

| Section | What it controls |
|---|---|
| `generator.*` | Synthetic data generation (fonts, text, layout, background, compositing, degradation) |
| `model.*` | Architecture, encoder, output heads, loss weights |
| `training.*` | Optimizer, scheduler, AMP, EMA, eval cadence, wandb |
| `adaptation.*` | CycleGAN / FDA / self-training / DANN settings (Phase 3) |
| `evaluation.*` | Benchmark paths and metric selection |

---

## 7 · Common workflows

### 7.1 Quick iteration on the generator

Generate 100 samples, eyeball them:

```bash
python -m scripts.generate_dataset \
    --num-images 100 --output /tmp/eyeball \
    --override generator.fonts.root_dir=C:/Windows/Fonts \
                generator.background.natural_image_dirs=[]
# Then visualize via the notebook's "Inspect generated samples" cell
```

### 7.2 Full pre-training run

```bash
# 1. Download all data (~25 GB)
bash scripts/setup_data.sh

# 2. Generate 100K samples
python -m scripts.generate_dataset --num-images 100000 --output data/main

# 3. Train UNet baseline
python -m scripts.train_model \
    --config segocr/configs/default.yaml \
    --override generator.output_dir=data/main \
                training.output_dir=weights/main_unet

# 4. Compare with SegFormer (after installing mmsegmentation)
python -m scripts.train_model \
    --config segocr/configs/default.yaml \
    --override generator.output_dir=data/main \
                model.architecture=segformer \
                model.encoder=mit_b2 \
                training.output_dir=weights/main_segformer
```

### 7.3 Noise-removal-only training

```bash
python -m scripts.generate_dataset \
    --num-images 100000 --output data/noise_removal --mode noise_removal

# Train with num_classes=2 (background + foreground)
python -m scripts.train_model \
    --override generator.output_dir=data/noise_removal \
                model.num_classes=2
```

The model trained this way collapses character recognition to "is there text here?" — useful when text is noise to be removed rather than recognized.

### 7.4 Resume from a checkpoint

```bash
# Note: not yet wired into the CLI — open scripts/train_model.py to load
# state['model'] before training.
```

### 7.5 wandb logging

Set `training.wandb.project` and `training.wandb.entity` in the config, then run `wandb login` once. All loss components, learning rate, per-class IoU, and the three mIoU variants will be logged to your W&B project.

---

## 8 · Troubleshooting

### `No validated fonts available under …`

The font scanner found zero usable fonts. Check that `generator.fonts.root_dir` points to a directory containing `.ttf` / `.otf` files, and that the validation didn't reject all of them (Pillow can crash on some malformed fonts; we silently skip them).

```bash
# Test what fonts are visible
python -c "from pathlib import Path; \
    print(list(Path('/usr/share/fonts').rglob('*.ttf'))[:5])"
```

### Tests fail with `ModuleNotFoundError: torch`

Phase 2 deps not installed. Re-run the install step from §2. Phase 1 (data generation) does not need torch.

### `No images found under data/.../images`

Either you haven't generated a dataset yet, or `generator.output_dir` doesn't match the directory you generated to. Pass `--override generator.output_dir=…` matching your actual output path.

### CUDA out of memory during training

Reduce `training.batch_size` or use a smaller encoder (`resnet18` → `resnet50` is ~3× memory). With AMP enabled on T4 you should fit `batch_size=8` for `resnet50` at 512×512.

### Generation is slow on Colab

Two common causes:
- Tier 3 (natural images) re-loads from disk on every preload-buffer refresh. Pre-warming COCO into RAM helps.
- `num_workers=0` runs single-process. Set `--override generator.num_workers=4` on Colab.

### `mmsegmentation install fails on Windows`

Known issue. Use the UNet baseline (`model.architecture=unet`), which doesn't depend on mmsegmentation. SegFormer support is gated on Linux/Colab for now.

### Generated text is just `ABCDEFG`

You're hitting the random-only fallback. Check that the bundled corpora are loading:

```python
from segocr.generator.text_sampler import TextSampler
sampler = TextSampler(config['generator']['text'])
print([(tag, len(sentences)) for tag, sentences, _ in sampler.corpora])
# Should show all four bundled corpora with positive lengths
```

---

## Project structure (cheat-sheet)

```
segocr/
├── configs/default.yaml        # source of truth for all hyperparameters
├── assets/corpora/             # bundled mini-corpora (signs/receipts/names/numbers)
├── generator/                  # synthetic data pipeline (Phase 1)
│   ├── font_manager.py         # font scan + validation + sampling
│   ├── text_sampler.py         # multi-corpus weighted text sampling
│   ├── renderer.py             # the oracle — alpha channel = mask
│   ├── layout.py               # 6 spatial modes
│   ├── background.py           # 4-tier background generator
│   ├── compositor.py           # alpha compositing + visual modes
│   ├── degradation.py          # albumentations + custom degradations
│   ├── saliency.py             # edge-density-based placement scorer
│   ├── placement.py            # collision-mask tracker (multi-text instances)
│   ├── targets.py              # build instance/affinity/direction from semantic
│   └── engine.py               # orchestrator (single-process + multiprocessing)
├── models/                     # architectures (Phase 2)
│   ├── losses.py               # FocalLoss + DiceLoss + SegOCRLoss
│   ├── heads.py                # semantic + affinity + direction heads
│   ├── unet.py                 # UNet baseline (smp-backed)
│   └── segformer.py            # SegFormer (gated on mmsegmentation)
├── training/                   # training pipeline (Phase 2)
│   ├── dataset.py              # PyTorch Dataset for the generator's on-disk format
│   ├── train.py                # main training loop with EMA + AMP + ckpt averaging
│   └── evaluator.py            # streaming-confusion-matrix metrics
├── adaptation/                 # synthetic→real domain adaptation (Phase 3, stubs)
├── postprocessing/             # pixels→text post-processing (Phase 4, stubs)
├── evaluation/                 # benchmark runner (stubs)
└── utils/                      # config loader, charset definitions

scripts/
├── generate_dataset.py         # python -m scripts.generate_dataset
├── train_model.py              # python -m scripts.train_model
├── evaluate.py                 # python -m scripts.evaluate (stub)
├── setup_data.ps1              # data download (Windows)
└── setup_data.sh               # data download (Linux/macOS/Colab)

notebooks/
└── segocr_colab_quickstart.ipynb   # end-to-end demo on Colab
```
