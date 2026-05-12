"""Build the two dataset-generator notebooks (parts A and B) for Kaggle.

Splitting into two datasets keeps each generation run under the 20 GB
/kaggle/working/ cap (40K samples × ~365 KB ≈ 14.6 GB per dataset).
Each training worker attaches BOTH datasets and symlinks them into
one merged directory at training time.

Index ranges are disjoint so worker N's combined slice is deterministic:
  Part A → indices [0, 40000)         — worker N: [N*8000, N*8000+8000)
  Part B → indices [40000, 80000)     — worker N: [40000+N*8000, ...)
"""
from __future__ import annotations

import json
from pathlib import Path

KAGGLE_DIR = Path(__file__).parent

# Per-part config
PARTS = [
    {
        "letter": "a",
        "filename": "00a_generate_dataset_a.ipynb",
        "slug": "segocr-ensemble-a",
        "output_dir_name": "segocr-ensemble-a",
        "base_offset": 0,           # global index start for this part
    },
    {
        "letter": "b",
        "filename": "00b_generate_dataset_b.ipynb",
        "slug": "segocr-ensemble-b",
        "output_dir_name": "segocr-ensemble-b",
        "base_offset": 40_000,       # part B starts here so its indices are disjoint from A
    },
]
NUM_WORKERS = 5
SAMPLES_PER_WORKER = 8_000   # 5 × 8K × 2 parts = 80K total samples


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


def build(part: dict) -> dict:
    letter = part["letter"]
    slug = part["slug"]
    out_name = part["output_dir_name"]
    base_offset = part["base_offset"]
    total_in_this_part = NUM_WORKERS * SAMPLES_PER_WORKER

    cells = [
        md(
            f"# SegOCR — Generate Ensemble Dataset Part {letter.upper()} (Kaggle)\n\n"
            f"Generates **{total_in_this_part:,} synthetic samples** "
            f"(5 worker slices × {SAMPLES_PER_WORKER:,} each, ~14.6 GB) "
            f"at 512², for the second half of the 80K-total split.\n\n"
            f"Part A covers indices `[0, 40000)`. Part B covers `[40000, "
            "80000)`. Each training worker attaches **both** datasets and "
            "symlinks them into one merged directory — so worker N ends up "
            f"training on {2 * SAMPLES_PER_WORKER:,} samples total.\n\n"
            "**Before running:**\n"
            "1. Settings → Accelerator: **None** (CPU is enough).\n"
            "2. Click **Save Version → Save & Run All**.\n\n"
            "**Wall time:** ~1.5–2 hours on Kaggle CPU.\n\n"
            "**After it finishes:** click *New Dataset* from the output panel "
            f"and publish as **`{slug}`** (Public). All 5 training notebooks "
            f"attach this dataset by that slug (plus the matching part-other "
            "dataset)."
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
        md("## 2 / Plan + paths"),
        code(
            "import os\n"
            f"NUM_WORKERS = {NUM_WORKERS}\n"
            f"SAMPLES_PER_WORKER = {SAMPLES_PER_WORKER}   # 5 × 8K = 40K total per dataset (~14.6 GB)\n"
            f"BASE_OFFSET = {base_offset}                  # global index start for part {letter.upper()}\n"
            f"DATASET_OUTPUT = '/kaggle/working/{out_name}'\n"
            "os.makedirs(DATASET_OUTPUT, exist_ok=True)\n"
            "\n"
            f"print(f'Part {letter.upper()}: 5 × {{SAMPLES_PER_WORKER}} = {{NUM_WORKERS * SAMPLES_PER_WORKER}} samples')\n"
            "print(f'Global index range: [{BASE_OFFSET}, {BASE_OFFSET + NUM_WORKERS * SAMPLES_PER_WORKER})')\n"
            "print(f'Output: {DATASET_OUTPUT}')"
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
            "## 4 / Generate the 5 worker slices for this part\n\n"
            "Worker N in this part uses indices "
            "`[BASE_OFFSET + N*SAMPLES_PER_WORKER, "
            "BASE_OFFSET + (N+1)*SAMPLES_PER_WORKER)`. The deterministic "
            "per-index seed guarantees the other gen notebook produces "
            "different images for its disjoint range."
        ),
        code(
            "import time\n"
            "for worker_id in range(NUM_WORKERS):\n"
            "    output = f'{DATASET_OUTPUT}/worker{worker_id}'\n"
            "    if os.path.isdir(f'{output}/images') and len(os.listdir(f'{output}/images')) >= SAMPLES_PER_WORKER:\n"
            "        print(f'[worker {worker_id}] already has {SAMPLES_PER_WORKER} samples — skipping')\n"
            "        continue\n"
            "    offset = BASE_OFFSET + worker_id * SAMPLES_PER_WORKER\n"
            "    end = offset + SAMPLES_PER_WORKER\n"
            "    print(f'\\n=== Generating worker {worker_id}: indices {offset}..{end - 1} ===')\n"
            "    t0 = time.time()\n"
            "    !python -m scripts.generate_dataset --config {config_path} --num-images {SAMPLES_PER_WORKER} --output {output} --index-offset {offset}\n"
            "    print(f'[worker {worker_id}] done in {(time.time() - t0)/60:.1f} min')\n"
            "\n"
            f"print('\\nPart {letter.upper()} done.')"
        ),
        md("## 5 / Sanity-check output layout"),
        code(
            "for worker_id in range(NUM_WORKERS):\n"
            "    worker_dir = f'{DATASET_OUTPUT}/worker{worker_id}'\n"
            "    n_imgs = len(os.listdir(f'{worker_dir}/images')) if os.path.isdir(f'{worker_dir}/images') else 0\n"
            "    n_sem  = len(os.listdir(f'{worker_dir}/semantic')) if os.path.isdir(f'{worker_dir}/semantic') else 0\n"
            "    n_inst = len(os.listdir(f'{worker_dir}/instance')) if os.path.isdir(f'{worker_dir}/instance') else 0\n"
            "    n_meta = len(os.listdir(f'{worker_dir}/metadata')) if os.path.isdir(f'{worker_dir}/metadata') else 0\n"
            "    print(f'worker{worker_id}: {n_imgs} images / {n_sem} semantic / {n_inst} instance / {n_meta} metadata')\n"
            "\n"
            "# Total disk usage\n"
            "!du -sh {DATASET_OUTPUT}"
        ),
        md(
            f"## 6 / Publish as a Kaggle Dataset\n\n"
            f"1. Click **Save Version → Save & Run All** if you haven't.\n"
            f"2. **Output** tab → **New Dataset**.\n"
            f"3. Slug: **`{slug}`**. Visibility: **Public** (simplest "
            f"for cross-account sharing).\n"
            f"4. Note the dataset URL — the 5 training notebooks attach "
            f"this **plus** the other part."
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
    # Clean up the old single-part notebook if present
    old = KAGGLE_DIR / "00_generate_dataset.ipynb"
    if old.exists():
        old.unlink()
        print(f"Removed {old}")
    for part in PARTS:
        nb = build(part)
        out_path = KAGGLE_DIR / part["filename"]
        out_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote {out_path}  ({len(nb['cells'])} cells)")


if __name__ == "__main__":
    main()
