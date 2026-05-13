"""Build dual-worker training notebooks for Kaggle's 2× T4 setup.

Each notebook launches TWO independent training processes on the same
Kaggle session — one pinned to cuda:0, the other to cuda:1 — training
two different worker IDs of the 5-worker ensemble. They share no state:
no DataParallel sync, no gradient gather, no master GPU. Each runs at
full single-GPU speed, in parallel.

Net effect: one 9 hr Kaggle session produces two trained worker
checkpoints instead of one. For the 5-worker ensemble this means
3 accounts instead of 5 (or, with workers 1,2 and 3,4 paired and
worker 0 already done, just 2 more accounts).

    python kaggle/_build_dual_train_notebooks.py
"""
from __future__ import annotations

import json
from pathlib import Path

KAGGLE_DIR = Path(__file__).parent

# Worker pairs to build. Worker 0 is assumed already trained.
PAIRS = [
    {"filename": "06_train_workers_1_2.ipynb", "worker_a": 1, "worker_b": 2},
    {"filename": "07_train_workers_3_4.ipynb", "worker_a": 3, "worker_b": 4},
]

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


def build_dual_notebook(worker_a: int, worker_b: int) -> dict:
    cells = [
        md(
            f"# SegOCR — Train Workers {worker_a} + {worker_b} in Parallel (Kaggle, 2× T4)\n\n"
            f"This notebook trains **workers {worker_a} and {worker_b}** of the "
            "5-worker ensemble simultaneously — each pinned to one of Kaggle's "
            "two T4 GPUs as an independent training process. No DataParallel, "
            "no gradient sync. Both run at full single-GPU speed in parallel.\n\n"
            f"**One Kaggle session → two trained workers.** This is the efficient "
            f"way to use Kaggle's 2× T4 for an ensemble (vs DataParallel, which "
            f"actually runs *slower* than single-GPU for this model size).\n\n"
            "**Before running:**\n"
            f"1. Settings → Accelerator: **GPU T4 x2** (must be the two-GPU option).\n"
            f"2. Add Data (sidebar) → attach **both** `{DATASET_SLUG_A}` "
            f"and `{DATASET_SLUG_B}`.\n"
            f"3. (Optional, for resume) Add Data → Notebook Output Files → "
            "attach your previous version's output of this notebook.\n"
            "4. Click **Save Version → Save & Run All** (runs headless for ~8 hr).\n\n"
            "When done, you'll have:\n"
            f"- `/kaggle/working/weights_w{worker_a}/averaged_best.pth`\n"
            f"- `/kaggle/working/weights_w{worker_b}/averaged_best.pth`\n"
            f"Both appear as downloadable notebook outputs."
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
        md("## 2 / Verify 2 GPUs visible"),
        code(
            "import torch\n"
            "if not torch.cuda.is_available():\n"
            "    raise RuntimeError('No CUDA. Settings → Accelerator → GPU T4 x2.')\n"
            "n_gpus = torch.cuda.device_count()\n"
            "print(f'CUDA devices visible: {n_gpus}')\n"
            "for i in range(n_gpus):\n"
            "    name = torch.cuda.get_device_name(i)\n"
            "    mem_gb = torch.cuda.get_device_properties(i).total_memory / 1e9\n"
            "    print(f'  cuda:{i}  {name}  ({mem_gb:.1f} GB)')\n"
            "if n_gpus < 2:\n"
            "    raise RuntimeError(\n"
            "        f'Need 2 GPUs for this notebook (got {n_gpus}). '\n"
            "        'Settings → Accelerator → GPU T4 x2.'\n"
            "    )"
        ),
        md(
            f"## 3 / Worker config (locked: workers {worker_a} + {worker_b})\n\n"
            "Both workers attach the same two datasets but use disjoint slices. "
            "Each worker gets its own weights dir and seed."
        ),
        code(
            f"WORKER_A = {worker_a}    # trained on cuda:0\n"
            f"WORKER_B = {worker_b}    # trained on cuda:1\n"
            "WORKERS = [WORKER_A, WORKER_B]\n"
            "\n"
            "import os, glob, shutil\n"
            f"DATASET_SLUG_A = '{DATASET_SLUG_A}'   # indices 0..39999\n"
            f"DATASET_SLUG_B = '{DATASET_SLUG_B}'   # indices 40000..79999\n"
            "\n"
            "def find_dataset_worker_dir(slug, worker_id):\n"
            "    '''Locate /kaggle/input/.../worker{N} for a given dataset slug.'''\n"
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
            "DATA_DIRS = {}\n"
            "WEIGHTS_DIRS = {}\n"
            "for w in WORKERS:\n"
            "    a_dir = find_dataset_worker_dir(DATASET_SLUG_A, w)\n"
            "    b_dir = find_dataset_worker_dir(DATASET_SLUG_B, w)\n"
            "    merged = f'/kaggle/working/segocr_data_w{w}_merged'\n"
            "    if os.path.isdir(merged):\n"
            "        shutil.rmtree(merged)\n"
            "    for subdir in ('images', 'semantic', 'instance', 'metadata'):\n"
            "        os.makedirs(f'{merged}/{subdir}', exist_ok=True)\n"
            "        for src_base in (a_dir, b_dir):\n"
            "            src_dir = f'{src_base}/{subdir}'\n"
            "            if not os.path.isdir(src_dir):\n"
            "                continue\n"
            "            for fname in os.listdir(src_dir):\n"
            "                src = f'{src_dir}/{fname}'\n"
            "                dst = f'{merged}/{subdir}/{fname}'\n"
            "                if not os.path.exists(dst):\n"
            "                    os.symlink(src, dst)\n"
            "    DATA_DIRS[w] = merged\n"
            "    WEIGHTS_DIRS[w] = f'/kaggle/working/weights_w{w}'\n"
            "    os.makedirs(WEIGHTS_DIRS[w], exist_ok=True)\n"
            "    n_merged = len(os.listdir(f'{merged}/images'))\n"
            "    print(f'Worker {w}: data={merged} ({n_merged} imgs), weights={WEIGHTS_DIRS[w]}')"
        ),
        md(
            "## 4 / Resume from previous version's output (if attached)\n\n"
            "If a previous run of this notebook was saved and re-attached "
            "via **Add Data → Notebook Output Files**, its checkpoints get "
            "auto-discovered and copied per worker."
        ),
        code(
            "import shutil, glob\n"
            "for w in WORKERS:\n"
            "    prev = []\n"
            "    for pat in (\n"
            "        f'/kaggle/input/*/weights_w{w}/checkpoint_*.pth',\n"
            "        f'/kaggle/input/*/weights_w{w}/averaged_best.pth',\n"
            "        f'/kaggle/input/*/weights_w{w}/snapshot_*.pth',\n"
            "    ):\n"
            "        prev.extend(glob.glob(pat))\n"
            "    if prev:\n"
            "        print(f'Worker {w}: found {len(prev)} previous checkpoint(s); copying...')\n"
            "        for src in prev:\n"
            "            dst = os.path.join(WEIGHTS_DIRS[w], os.path.basename(src))\n"
            "            shutil.copy(src, dst)\n"
            "            print(f'  {src} -> {dst}')\n"
            "    else:\n"
            "        print(f'Worker {w}: no previous checkpoints; starting fresh.')"
        ),
        md(
            "## 5 / Build per-worker training configs\n\n"
            "Same hyperparameters as the single-worker notebooks (batch=16, "
            "30K iters, 512²). Only data_dir + output_dir differ between "
            "the two configs."
        ),
        code(
            "import yaml\n"
            "from pathlib import Path\n"
            "\n"
            "IMG_SIZE = 512\n"
            "TOTAL_ITERS = 30_000\n"
            "BATCH_SIZE = 16\n"
            "\n"
            "def build_cfg_for_worker(w):\n"
            "    cfg = yaml.safe_load(Path('segocr/configs/default.yaml').read_text())\n"
            "    cfg['generator']['fonts']['root_dir'] = '/usr/share/fonts'\n"
            "    cfg['generator']['fonts']['cache_path'] = '/kaggle/working/font_cache.json'\n"
            "    cfg['generator']['fonts']['min_size'] = 40\n"
            "    cfg['generator']['fonts']['max_size'] = 128\n"
            "    cfg['generator']['image_size'] = [IMG_SIZE, IMG_SIZE]\n"
            "    cfg['generator']['num_workers'] = 2\n"
            "    cfg['generator']['output_dir'] = DATA_DIRS[w]\n"
            "    cfg['generator']['text']['min_length'] = 2\n"
            "    cfg['generator']['text']['max_length'] = 20\n"
            "    cfg['generator']['text']['min_words_per_line'] = 1\n"
            "    cfg['generator']['text']['max_words_per_line'] = 3\n"
            "    cfg['generator']['text']['max_lines'] = 1\n"
            "    cfg['generator']['text']['case_distribution'] = {\n"
            "        'lower': 0.50, 'upper': 0.20, 'mixed': 0.20, 'title': 0.10,\n"
            "    }\n"
            "    cfg['generator']['text']['rare_char_boost'] = 4.0\n"
            "    cfg['generator']['text']['corpus_paths'] = [\n"
            "        {'path': 'BUNDLED:signs', 'tag': 'signs', 'weight': 0.30},\n"
            "        {'path': 'BUNDLED:receipts', 'tag': 'receipts', 'weight': 0.20},\n"
            "        {'path': 'BUNDLED:names', 'tag': 'names', 'weight': 0.20},\n"
            "        {'path': 'BUNDLED:numbers', 'tag': 'numbers', 'weight': 0.30},\n"
            "    ]\n"
            "    cfg['generator']['layout']['modes'] = {\n"
            "        'horizontal': 0.50, 'rotated': 0.20, 'curved': 0.10,\n"
            "        'perspective': 0.10, 'deformed': 0.10, 'paragraph': 0.0,\n"
            "    }\n"
            "    cfg['generator']['background']['natural_image_dirs'] = []\n"
            "    cfg['generator']['background']['tier_distribution'] = {\n"
            "        'tier1_solid': 0.40, 'tier2_procedural': 0.30,\n"
            "        'tier3_natural': 0.25, 'tier4_adversarial': 0.05,\n"
            "    }\n"
            "    cfg['generator']['compositing']['color_strategy'] = {\n"
            "        'contrast_aware': 0.60, 'random': 0.30, 'low_contrast': 0.10,\n"
            "    }\n"
            "    cfg['generator']['degradation']['blur'] = {\n"
            "        'probability': 0.30, 'gaussian_sigma': [0.3, 1.0],\n"
            "        'motion_kernel': [3, 7], 'defocus_radius': [1, 3],\n"
            "    }\n"
            "    cfg['generator']['degradation']['noise']['probability'] = 0.40\n"
            "    cfg['generator']['degradation']['noise']['gaussian_sigma'] = [5, 20]\n"
            "    cfg['generator']['degradation']['occlusion']['probability'] = 0.05\n"
            "    cfg['model']['architecture'] = 'unet'\n"
            "    cfg['model']['encoder'] = 'resnet18'\n"
            "    cfg['model']['encoder_weights'] = 'imagenet'\n"
            "    cfg['model']['head_features'] = 64\n"
            "    cfg['model']['decoder_channels'] = [256, 128, 64, 32, 32]\n"
            "    cfg['model']['heads'] = {'semantic': True, 'affinity': True, 'direction': True}\n"
            "    cfg['model']['num_classes'] = 63\n"
            "    cfg['model']['input_size'] = [IMG_SIZE, IMG_SIZE]\n"
            "    cfg['training']['learning_rate'] = 3e-4\n"
            "    cfg['training']['weight_decay'] = 1e-4\n"
            "    cfg['training']['total_iters'] = TOTAL_ITERS\n"
            "    cfg['training']['warmup_iters'] = 1_000\n"
            "    cfg['training']['eval_interval'] = 2_500\n"
            "    cfg['training']['save_interval'] = 2_500\n"
            "    cfg['training']['log_interval'] = 100\n"
            "    cfg['training']['batch_size'] = BATCH_SIZE\n"
            "    cfg['training']['num_workers'] = 2\n"
            "    cfg['training']['mixed_precision'] = True\n"
            "    cfg['training']['output_dir'] = WEIGHTS_DIRS[w]\n"
            "    cfg['training']['ema'] = {'enabled': True, 'decay': 0.999}\n"
            "    cfg['training']['checkpoint_averaging'] = {'enabled': True, 'top_n': 3}\n"
            "    cfg['training']['keep_best_n'] = 5\n"
            "    cfg['training']['wandb'] = {'project': None}\n"
            "    return cfg\n"
            "\n"
            "CONFIG_PATHS = {}\n"
            "for w in WORKERS:\n"
            "    cfg = build_cfg_for_worker(w)\n"
            "    path = f'/kaggle/working/train_config_w{w}.yaml'\n"
            "    Path(path).write_text(yaml.safe_dump(cfg))\n"
            "    CONFIG_PATHS[w] = path\n"
            "    n_images = len(os.listdir(DATA_DIRS[w] + '/images'))\n"
            "    print(f'Worker {w}: {n_images} samples × {TOTAL_ITERS} iters @ {IMG_SIZE}px, batch {BATCH_SIZE}')\n"
            "    print(f'  config: {path}')"
        ),
        md(
            "## 6 / Launch two parallel training processes\n\n"
            "Each subprocess sees only one GPU (via `CUDA_VISIBLE_DEVICES`), "
            "so the training code runs as if on a single-GPU machine. "
            "They share no PyTorch state — fully independent.\n\n"
            "Logs are written to `/kaggle/working/train_w{N}.log`. This cell "
            "polls every 5 min and prints the tail of each log so you can "
            "track progress in **Save & Run All** mode."
        ),
        code(
            "import subprocess, sys, time\n"
            "\n"
            "GPU_OF = {WORKER_A: 0, WORKER_B: 1}\n"
            "procs = []\n"
            "for w in WORKERS:\n"
            "    env = os.environ.copy()\n"
            "    env['CUDA_VISIBLE_DEVICES'] = str(GPU_OF[w])\n"
            "    # Each process sees just one GPU as cuda:0 -- no code changes needed.\n"
            "    log_path = f'/kaggle/working/train_w{w}.log'\n"
            "    log_f = open(log_path, 'w', buffering=1)\n"
            "    seed = w + 1\n"
            "    cmd = [\n"
            "        sys.executable, '-u', '-m', 'scripts.train_model',\n"
            "        '--config', CONFIG_PATHS[w],\n"
            "        '--resume-latest', WEIGHTS_DIRS[w],\n"
            "        '--seed', str(seed),\n"
            "    ]\n"
            "    proc = subprocess.Popen(cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT)\n"
            "    procs.append({'proc': proc, 'log': log_path, 'log_f': log_f, 'worker': w, 'gpu': GPU_OF[w]})\n"
            "    print(f'Started worker {w} on cuda:{GPU_OF[w]} (PID {proc.pid}); log: {log_path}')\n"
            "\n"
            "print(f'\\nPolling every 5 min until both finish. Each worker ~7-8 hr.\\n')\n"
            "t0 = time.time()\n"
            "while any(p['proc'].poll() is None for p in procs):\n"
            "    time.sleep(300)\n"
            "    elapsed_min = (time.time() - t0) / 60\n"
            "    print(f'=== Status after {elapsed_min:.1f} min ===')\n"
            "    for p in procs:\n"
            "        rc = p['proc'].poll()\n"
            "        status = 'running' if rc is None else f'exited rc={rc}'\n"
            "        try:\n"
            "            last = subprocess.check_output(\n"
            "                ['tail', '-n', '3', p['log']], text=True\n"
            "            ).strip()\n"
            "        except Exception as e:\n"
            "            last = f'(could not tail log: {e})'\n"
            "        print(f'  worker {p[\"worker\"]} (cuda:{p[\"gpu\"]}): {status}')\n"
            "        for line in last.splitlines():\n"
            "            print(f'    | {line}')\n"
            "    print()\n"
            "\n"
            "for p in procs:\n"
            "    p['log_f'].close()\n"
            "    print(f\"Worker {p['worker']} final exit code: {p['proc'].returncode}\")\n"
            "    print(f\"  Full log: {p['log']}\")\n"
            "    print(f\"  Weights:  {WEIGHTS_DIRS[p['worker']]}\")"
        ),
        md(
            "## 7 / Output files (download these)\n\n"
            "Both `weights_wA/averaged_best.pth` and `weights_wB/averaged_best.pth` "
            "will appear in the notebook's **Output** tab. Download them along "
            "with the other workers' outputs and run `scripts/average_runs.py` "
            "locally to build the final ensemble."
        ),
        code(
            "import os\n"
            "for w in WORKERS:\n"
            "    wdir = WEIGHTS_DIRS[w]\n"
            "    print(f'\\n/kaggle/working/weights_w{w}/:')\n"
            "    if not os.path.isdir(wdir):\n"
            "        print('  (missing)')\n"
            "        continue\n"
            "    for name in sorted(os.listdir(wdir)):\n"
            "        size_mb = os.path.getsize(os.path.join(wdir, name)) / 1e6\n"
            "        print(f'  {name:40s}  {size_mb:8.1f} MB')\n"
            "\n"
            "print()\n"
            "print('Next: download averaged_best.pth from each weights_wN dir,')\n"
            "print('      collect all 5 worker .pth files, run scripts/average_runs.py locally.')"
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
    for pair in PAIRS:
        nb = build_dual_notebook(pair["worker_a"], pair["worker_b"])
        out_path = KAGGLE_DIR / pair["filename"]
        out_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote {out_path}  ({len(nb['cells'])} cells)")


if __name__ == "__main__":
    main()
