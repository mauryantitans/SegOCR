"""Build 5 per-worker training notebooks for Kaggle from a single template.

Each output notebook differs only in the hardcoded WORKER_ID so the user
can just fork/import the one matching their account without risking a
parameter mix-up. Re-run this script after editing the template to
regenerate all 5 files.

    python kaggle/_build_train_notebooks.py
"""
from __future__ import annotations

import json
from pathlib import Path

KAGGLE_DIR = Path(__file__).parent
NUM_WORKERS = 5
# Two halves; each is ~14.6 GB and fits Kaggle's 20 GB /kaggle/working/ cap.
# Trainer notebooks attach BOTH and symlink-merge them into one local dir.
DATASET_SLUG_A = "segocr-ensemble-a"   # indices 0..39999
DATASET_SLUG_B = "segocr-ensemble-b"   # indices 40000..79999


def md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src}


def code(src: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src,
    }


def build_train_notebook(worker_id: int) -> dict:
    cells = [
        md(
            f"# SegOCR — Train Worker {worker_id} (Kaggle)\n\n"
            f"This notebook trains worker {worker_id} of a 5-worker parallel "
            "ensemble on Kaggle. `WORKER_ID` is hardcoded below — do not "
            "change it. Run this notebook on **one** Kaggle account; run the "
            "other four worker notebooks on different accounts in parallel.\n\n"
            "**Before running:**\n"
            f"1. Settings → Accelerator: **GPU T4**.\n"
            f"2. Add Data (sidebar) → attach **both** `{DATASET_SLUG_A}` "
            f"and `{DATASET_SLUG_B}`. They get mounted at "
            f"`/kaggle/input/{DATASET_SLUG_A}/` and "
            f"`/kaggle/input/{DATASET_SLUG_B}/`. The cell below "
            "symlink-merges them into one directory.\n"
            f"3. (Optional, for resume) Add Data → Notebook Output Files → "
            "attach your previous version's output of this notebook.\n"
            "4. Click **Save Version → Save & Run All** (runs server-side, "
            "survives connection issues, gives 9 hr).\n\n"
            "When the notebook finishes, your trained weights are at "
            "`/kaggle/working/weights/averaged_best.pth` and become a "
            "downloadable notebook output."
        ),
        md("## 1 / Setup — clone repo + install deps"),
        code(
            "import os\n"
            "if not os.path.isdir('/kaggle/working/segocr'):\n"
            "    !git clone https://github.com/mauryantitans/SegOCR.git /kaggle/working/segocr\n"
            "%cd /kaggle/working/segocr\n"
            "!git pull --quiet\n"
            "!pip install -q -e .\n"
            "!pip install -q -r requirements/base.txt\n"
            "!pip install -q segmentation-models-pytorch"
        ),
        md("## 2 / Verify GPU"),
        code(
            "import torch\n"
            "device = 'cuda' if torch.cuda.is_available() else 'cpu'\n"
            "print(f'Device: {device}')\n"
            "if device == 'cuda':\n"
            "    n_gpus = torch.cuda.device_count()\n"
            "    print(f'Number of GPUs visible: {n_gpus}')\n"
            "    for i in range(n_gpus):\n"
            "        name = torch.cuda.get_device_name(i)\n"
            "        mem_gb = torch.cuda.get_device_properties(i).total_memory / 1e9\n"
            "        print(f'  cuda:{i}  {name}  ({mem_gb:.1f} GB)')\n"
            "    if n_gpus > 1:\n"
            "        print(f'\\nTraining will use nn.DataParallel across all {n_gpus} GPUs.')\n"
            "else:\n"
            "    raise RuntimeError('No GPU. Settings → Accelerator → GPU T4 (or T4 x2).')"
        ),
        md(
            f"## 3 / Worker config (locked to WORKER_ID = {worker_id})\n\n"
            "These values are baked into this notebook. The 4 other worker "
            "notebooks have the same code but different `WORKER_ID`."
        ),
        code(
            f"WORKER_ID = {worker_id}     # hardcoded for this notebook — do not change\n"
            "TRAIN_SEED = WORKER_ID + 1   # different basin per worker\n"
            "\n"
            "import os, glob, shutil\n"
            f"DATASET_SLUG_A = '{DATASET_SLUG_A}'   # indices 0..39999\n"
            f"DATASET_SLUG_B = '{DATASET_SLUG_B}'   # indices 40000..79999\n"
            "\n"
            "def find_dataset_worker_dir(slug, worker_id):\n"
            "    '''Locate /kaggle/input/.../worker{N} for a given dataset slug.\n"
            "    Tries the standard mount path first, then nested variants that\n"
            "    notebook-output-as-dataset workflows can produce.'''\n"
            "    patterns = [\n"
            "        f'/kaggle/input/{slug}/worker{worker_id}',\n"
            "        f'/kaggle/input/{slug}/{slug}/worker{worker_id}',\n"
            "        f'/kaggle/input/datasets/*/{slug}/worker{worker_id}',\n"
            "        f'/kaggle/input/datasets/*/{slug}/{slug}/worker{worker_id}',\n"
            "    ]\n"
            "    for pat in patterns:\n"
            "        matches = sorted(glob.glob(pat))\n"
            "        if matches:\n"
            "            return matches[0]\n"
            "    raise FileNotFoundError(\n"
            "        f'Could not find worker{worker_id} in dataset {slug!r}. '\n"
            "        f'Tried: {patterns}. Did you attach the dataset via Add Data?'\n"
            "    )\n"
            "\n"
            "DATA_DIR_A = find_dataset_worker_dir(DATASET_SLUG_A, WORKER_ID)\n"
            "DATA_DIR_B = find_dataset_worker_dir(DATASET_SLUG_B, WORKER_ID)\n"
            "DATA_DIR = '/kaggle/working/segocr_data_merged'\n"
            "WEIGHTS_DIR = '/kaggle/working/weights'\n"
            "os.makedirs(WEIGHTS_DIR, exist_ok=True)\n"
            "\n"
            "# Symlink the two halves into one merged directory so SegOCRDataset\n"
            "# sees a single 16K-sample worker slice. Symlinks are ~50 bytes\n"
            "# each so the working-dir overhead is ~MB, not GB.\n"
            "if os.path.isdir(DATA_DIR):\n"
            "    shutil.rmtree(DATA_DIR)\n"
            "for subdir in ('images', 'semantic', 'instance', 'metadata'):\n"
            "    os.makedirs(f'{DATA_DIR}/{subdir}', exist_ok=True)\n"
            "    for src_base in (DATA_DIR_A, DATA_DIR_B):\n"
            "        src_dir = f'{src_base}/{subdir}'\n"
            "        if not os.path.isdir(src_dir):\n"
            "            continue\n"
            "        for fname in os.listdir(src_dir):\n"
            "            src = f'{src_dir}/{fname}'\n"
            "            dst = f'{DATA_DIR}/{subdir}/{fname}'\n"
            "            if not os.path.exists(dst):\n"
            "                os.symlink(src, dst)\n"
            "\n"
            "n_merged = len(os.listdir(f'{DATA_DIR}/images'))\n"
            "print(f'Worker {WORKER_ID} of 5 (parallel ensemble)')\n"
            "print(f'Dataset A: {DATA_DIR_A}')\n"
            "print(f'Dataset B: {DATA_DIR_B}')\n"
            "print(f'Merged:    {DATA_DIR}  ({n_merged} images via symlink)')\n"
            "print(f'Weights:   {WEIGHTS_DIR}')"
        ),
        md(
            "## 4 / Resume from previous version's output (if attached)\n\n"
            "If a previous run of this notebook was saved and re-attached "
            "via **Add Data → Notebook Output Files**, its checkpoints get "
            "mounted somewhere under `/kaggle/input/`. We auto-detect them "
            "and copy into `/kaggle/working/weights/` so `--resume-latest` "
            "picks them up.\n\n"
            "If this is your first run, this cell is a no-op."
        ),
        code(
            "import shutil, glob\n"
            "prev_checkpoints = []\n"
            "for path in glob.glob('/kaggle/input/*/weights/checkpoint_*.pth'):\n"
            "    prev_checkpoints.append(path)\n"
            "for path in glob.glob('/kaggle/input/*/weights/averaged_best.pth'):\n"
            "    prev_checkpoints.append(path)\n"
            "\n"
            "if prev_checkpoints:\n"
            "    print(f'Found {len(prev_checkpoints)} previous checkpoint(s); copying for resume...')\n"
            "    for src in prev_checkpoints:\n"
            "        dst = os.path.join(WEIGHTS_DIR, os.path.basename(src))\n"
            "        shutil.copy(src, dst)\n"
            "        print(f'  {src} -> {dst}')\n"
            "else:\n"
            "    print('No previous checkpoints attached; starting fresh.')"
        ),
        md("## 5 / Build the training config\n\nCalibrated for Kaggle T4 + 9 hr session."),
        code(
            "import yaml\n"
            "from pathlib import Path\n"
            "\n"
            "IMG_SIZE = 512\n"
            "TOTAL_ITERS = 30_000\n"
            "# DataParallel gathers (B, num_classes, H, W) to cuda:0; with 63 classes\n"
            "# at 512² each sample contributes ~63 MiB. cuda:0 also holds model copy,\n"
            "# optimizer state, EMA, and its own forward activations — so we keep the\n"
            "# *gathered* batch at 16. On 2× T4 that means PER_GPU_BATCH=8.\n"
            "n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1\n"
            "PER_GPU_BATCH = 16 if n_gpus <= 1 else 8\n"
            "BATCH_SIZE = PER_GPU_BATCH * max(1, n_gpus)\n"
            "\n"
            "cfg = yaml.safe_load(Path('segocr/configs/default.yaml').read_text())\n"
            "\n"
            "# Generator: fonts + paths\n"
            "cfg['generator']['fonts']['root_dir'] = '/usr/share/fonts'\n"
            "cfg['generator']['fonts']['cache_path'] = '/kaggle/working/font_cache.json'\n"
            "cfg['generator']['fonts']['min_size'] = 40\n"
            "cfg['generator']['fonts']['max_size'] = 128\n"
            "cfg['generator']['image_size'] = [IMG_SIZE, IMG_SIZE]\n"
            "cfg['generator']['num_workers'] = 2\n"
            "cfg['generator']['output_dir'] = DATA_DIR   # read from attached dataset\n"
            "\n"
            "# Generator: text + layout + bg + composite + degradation tweaks\n"
            "cfg['generator']['text']['min_length'] = 2\n"
            "cfg['generator']['text']['max_length'] = 20\n"
            "cfg['generator']['text']['min_words_per_line'] = 1\n"
            "cfg['generator']['text']['max_words_per_line'] = 3\n"
            "cfg['generator']['text']['max_lines'] = 1\n"
            "cfg['generator']['text']['case_distribution'] = {\n"
            "    'lower': 0.50, 'upper': 0.20, 'mixed': 0.20, 'title': 0.10,\n"
            "}\n"
            "cfg['generator']['text']['rare_char_boost'] = 4.0\n"
            "cfg['generator']['text']['corpus_paths'] = [\n"
            "    {'path': 'BUNDLED:signs', 'tag': 'signs', 'weight': 0.30},\n"
            "    {'path': 'BUNDLED:receipts', 'tag': 'receipts', 'weight': 0.20},\n"
            "    {'path': 'BUNDLED:names', 'tag': 'names', 'weight': 0.20},\n"
            "    {'path': 'BUNDLED:numbers', 'tag': 'numbers', 'weight': 0.30},\n"
            "]\n"
            "cfg['generator']['layout']['modes'] = {\n"
            "    'horizontal': 0.50, 'rotated': 0.20, 'curved': 0.10,\n"
            "    'perspective': 0.10, 'deformed': 0.10, 'paragraph': 0.0,\n"
            "}\n"
            "cfg['generator']['background']['natural_image_dirs'] = []\n"
            "cfg['generator']['background']['tier_distribution'] = {\n"
            "    'tier1_solid': 0.40, 'tier2_procedural': 0.30,\n"
            "    'tier3_natural': 0.25, 'tier4_adversarial': 0.05,\n"
            "}\n"
            "cfg['generator']['compositing']['color_strategy'] = {\n"
            "    'contrast_aware': 0.60, 'random': 0.30, 'low_contrast': 0.10,\n"
            "}\n"
            "cfg['generator']['degradation']['blur'] = {\n"
            "    'probability': 0.30, 'gaussian_sigma': [0.3, 1.0],\n"
            "    'motion_kernel': [3, 7], 'defocus_radius': [1, 3],\n"
            "}\n"
            "cfg['generator']['degradation']['noise']['probability'] = 0.40\n"
            "cfg['generator']['degradation']['noise']['gaussian_sigma'] = [5, 20]\n"
            "cfg['generator']['degradation']['occlusion']['probability'] = 0.05\n"
            "\n"
            "# Model\n"
            "cfg['model']['architecture'] = 'unet'\n"
            "cfg['model']['encoder'] = 'resnet18'\n"
            "cfg['model']['encoder_weights'] = 'imagenet'\n"
            "cfg['model']['head_features'] = 64\n"
            "cfg['model']['decoder_channels'] = [256, 128, 64, 32, 32]\n"
            "cfg['model']['heads'] = {'semantic': True, 'affinity': True, 'direction': True}\n"
            "cfg['model']['num_classes'] = 63\n"
            "cfg['model']['input_size'] = [IMG_SIZE, IMG_SIZE]\n"
            "\n"
            "# Training\n"
            "cfg['training']['learning_rate'] = 3e-4\n"
            "cfg['training']['weight_decay'] = 1e-4\n"
            "cfg['training']['total_iters'] = TOTAL_ITERS\n"
            "cfg['training']['warmup_iters'] = 1_000\n"
            "cfg['training']['eval_interval'] = 2_500\n"
            "cfg['training']['save_interval'] = 2_500\n"
            "cfg['training']['log_interval'] = 100\n"
            "cfg['training']['batch_size'] = BATCH_SIZE\n"
            "cfg['training']['num_workers'] = 2\n"
            "cfg['training']['mixed_precision'] = True\n"
            "cfg['training']['output_dir'] = WEIGHTS_DIR\n"
            "cfg['training']['ema'] = {'enabled': True, 'decay': 0.999}\n"
            "cfg['training']['checkpoint_averaging'] = {'enabled': True, 'top_n': 3}\n"
            "cfg['training']['keep_best_n'] = 5\n"
            "cfg['training']['wandb'] = {'project': None}\n"
            "\n"
            "config_path = '/kaggle/working/train_config.yaml'\n"
            "Path(config_path).write_text(yaml.safe_dump(cfg))\n"
            "n_images = len(os.listdir(DATA_DIR + '/images'))\n"
            "print(f'Config: {n_images} samples × {TOTAL_ITERS} iters @ {IMG_SIZE}px, batch {BATCH_SIZE}')"
        ),
        md(
            "## 6 / Train\n\n"
            "Training is the long step (~8 hr on T4). With **Save & Run All**, "
            "this runs server-side — close the tab if you want, your "
            "session continues. Checkpoints are written to "
            "`/kaggle/working/weights/` every 2,500 iters and the final "
            "top-3-averaged best is written there at the end."
        ),
        code(
            "!python -m scripts.train_model --config {config_path} --resume-latest {WEIGHTS_DIR} --seed {TRAIN_SEED}"
        ),
        md(
            "## 7 / Output files (download these)\n\n"
            "After training completes, the cell below shows what's in "
            "`/kaggle/working/weights/`. The key file is `averaged_best.pth` — "
            "that's the final trained model you'll combine with the other 4 "
            "workers' outputs.\n\n"
            "**To download:**\n\n"
            "1. **Easiest:** From this notebook's page, click the **Output** "
            "tab at top, then click the download icon next to "
            "`weights/averaged_best.pth`.\n"
            f"2. **Via Kaggle CLI** (on your laptop, after `pip install kaggle` "
            "and setting up `~/.kaggle/kaggle.json`):\n"
            "   ```bash\n"
            "   kaggle kernels output YOUR_USERNAME/YOUR_NOTEBOOK_SLUG "
            "-p ./downloads/worker"
            f"{worker_id}\n"
            "   ```\n\n"
            "After all 5 workers are downloaded, run "
            "`scripts/average_runs.py` locally to build the ensemble."
        ),
        code(
            "import os\n"
            "print('Contents of /kaggle/working/weights/:')\n"
            "for name in sorted(os.listdir(WEIGHTS_DIR)):\n"
            "    size_mb = os.path.getsize(os.path.join(WEIGHTS_DIR, name)) / 1e6\n"
            "    print(f'  {name:40s}  {size_mb:8.1f} MB')\n"
            "\n"
            "print()\n"
            "print('Next step: download averaged_best.pth from the Output tab')\n"
            "print('        : then run scripts/average_runs.py locally after collecting all 5 workers.')"
        ),
    ]

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    for worker_id in range(NUM_WORKERS):
        nb = build_train_notebook(worker_id)
        out_path = KAGGLE_DIR / f"{worker_id + 1:02d}_train_worker{worker_id}.ipynb"
        out_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote {out_path}  ({len(nb['cells'])} cells)")


if __name__ == "__main__":
    main()
