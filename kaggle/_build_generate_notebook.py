"""Build the dataset-generator notebook for Kaggle."""
from __future__ import annotations

import json
from pathlib import Path

KAGGLE_DIR = Path(__file__).parent


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


def build() -> dict:
    cells = [
        md(
            "# SegOCR — Generate 5-Worker Ensemble Dataset (Kaggle)\n\n"
            "Generates 80,000 synthetic samples (5 worker slices × 16K each) "
            "at 512², ready to publish as a Kaggle Dataset for the 5 parallel "
            "training notebooks.\n\n"
            "**Before running:**\n"
            "1. Settings → Accelerator: **None** (CPU is enough; generation "
            "is not GPU-bound).\n"
            "2. Settings → Persistence: **Files only** is fine.\n"
            "3. Click **Save Version → Save & Run All**.\n\n"
            "**Wall time:** ~4–5 hours on Kaggle CPU.\n\n"
            "**After it finishes:** click *New Dataset* from the output panel "
            "and publish as `segocr-ensemble-80k` (Public). The 5 training "
            "notebooks attach this dataset by that slug."
        ),
        md("## 1 / Setup — clone repo + install deps"),
        code(
            "import os\n"
            "if not os.path.isdir('/kaggle/working/segocr'):\n"
            "    !git clone https://github.com/mauryantitans/SegOCR.git /kaggle/working/segocr\n"
            "%cd /kaggle/working/segocr\n"
            "!git pull --quiet\n"
            "!pip install -q -e .\n"
            "!pip install -q -r requirements/base.txt"
        ),
        md("## 2 / Verify font availability + plan"),
        code(
            "import os\n"
            "NUM_WORKERS = 5\n"
            "SAMPLES_PER_WORKER = 16_000   # 5 × 16K = 80K total\n"
            "DATASET_OUTPUT = '/kaggle/working/segocr-ensemble-80k'\n"
            "os.makedirs(DATASET_OUTPUT, exist_ok=True)\n"
            "\n"
            "n_fonts = sum(\n"
            "    1 for root, _, files in os.walk('/usr/share/fonts')\n"
            "    for f in files if f.endswith(('.ttf', '.otf'))\n"
            ")\n"
            "print(f'System fonts available: {n_fonts}')\n"
            "print(f'Output: {DATASET_OUTPUT}')\n"
            "print(f'Plan:   {NUM_WORKERS} slices × {SAMPLES_PER_WORKER} = {NUM_WORKERS * SAMPLES_PER_WORKER} samples total')"
        ),
        md("## 3 / Build the generator config"),
        code(
            "import yaml\n"
            "from pathlib import Path\n"
            "\n"
            "cfg = yaml.safe_load(Path('segocr/configs/default.yaml').read_text())\n"
            "\n"
            "cfg['generator']['fonts']['root_dir'] = '/usr/share/fonts'\n"
            "cfg['generator']['fonts']['cache_path'] = '/kaggle/working/font_cache.json'\n"
            "cfg['generator']['fonts']['min_size'] = 40\n"
            "cfg['generator']['fonts']['max_size'] = 128\n"
            "cfg['generator']['image_size'] = [512, 512]\n"
            "cfg['generator']['num_workers'] = 2\n"
            "\n"
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
            "config_path = '/kaggle/working/gen_config.yaml'\n"
            "Path(config_path).write_text(yaml.safe_dump(cfg))\n"
            "print(f'Config: {config_path}')"
        ),
        md(
            "## 4 / Generate 5 worker slices serially\n\n"
            "Each slice gets a deterministic, disjoint range of indices "
            "(worker N uses indices `N * SAMPLES_PER_WORKER .. (N+1) * "
            "SAMPLES_PER_WORKER - 1`). Same code + same indices = same images, "
            "always — so any of the 5 training accounts will see exactly the "
            "data this notebook produces here.\n\n"
            "Generation runs ~5–10 img/s on Kaggle CPU at 512². Each 16K slice "
            "takes ~30–50 min. Total: ~3–4 hours."
        ),
        code(
            "import time\n"
            "for worker_id in range(NUM_WORKERS):\n"
            "    output = f'{DATASET_OUTPUT}/worker{worker_id}'\n"
            "    if os.path.isdir(f'{output}/images') and len(os.listdir(f'{output}/images')) >= SAMPLES_PER_WORKER:\n"
            "        print(f'[worker {worker_id}] already has {SAMPLES_PER_WORKER} samples — skipping')\n"
            "        continue\n"
            "    offset = worker_id * SAMPLES_PER_WORKER\n"
            "    print(f'\\n=== Generating worker {worker_id}: indices {offset}..{offset + SAMPLES_PER_WORKER - 1} ===')\n"
            "    t0 = time.time()\n"
            "    !python -m scripts.generate_dataset --config {config_path} --num-images {SAMPLES_PER_WORKER} --output {output} --index-offset {offset}\n"
            "    print(f'[worker {worker_id}] done in {(time.time() - t0)/60:.1f} min')\n"
            "\n"
            "print('\\nAll 5 worker slices generated.')"
        ),
        md("## 5 / Sanity-check output layout"),
        code(
            "for worker_id in range(NUM_WORKERS):\n"
            "    worker_dir = f'{DATASET_OUTPUT}/worker{worker_id}'\n"
            "    n_imgs = len(os.listdir(f'{worker_dir}/images')) if os.path.isdir(f'{worker_dir}/images') else 0\n"
            "    n_sem  = len(os.listdir(f'{worker_dir}/semantic')) if os.path.isdir(f'{worker_dir}/semantic') else 0\n"
            "    n_inst = len(os.listdir(f'{worker_dir}/instance')) if os.path.isdir(f'{worker_dir}/instance') else 0\n"
            "    n_meta = len(os.listdir(f'{worker_dir}/metadata')) if os.path.isdir(f'{worker_dir}/metadata') else 0\n"
            "    print(f'worker{worker_id}: {n_imgs} images / {n_sem} semantic / {n_inst} instance / {n_meta} metadata')"
        ),
        md(
            "## 6 / Publish as a Kaggle Dataset\n\n"
            "1. Click **Save Version → Save & Run All** at the top right (if "
            "you haven't already).\n"
            "2. When the run completes, the **Output** tab shows the "
            "`segocr-ensemble-80k/` folder.\n"
            "3. Click the **New Dataset** button in the Output panel.\n"
            "4. Slug: `segocr-ensemble-80k`. Visibility: **Public** (simplest "
            "for cross-account sharing).\n"
            "5. After publish, copy the dataset URL — each training notebook "
            "attaches it via *Add Data*.\n\n"
            "Total dataset size: ~15–25 GB (well within Kaggle's 100 GB per-"
            "dataset limit)."
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
    nb = build()
    out_path = KAGGLE_DIR / "00_generate_dataset.ipynb"
    out_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out_path}  ({len(nb['cells'])} cells)")


if __name__ == "__main__":
    main()
