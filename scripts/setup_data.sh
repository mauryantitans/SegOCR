#!/usr/bin/env bash
# SegOCR — data download script (Linux / macOS / Colab).
# Run from repo root:  bash scripts/setup_data.sh
#
# Total disk: ~30 GB. Skips downloads that already exist.

set -euo pipefail

DATA_ROOT="data"
FONTS="${DATA_ROOT}/fonts"
BG="${DATA_ROOT}/backgrounds"
CORPORA="${DATA_ROOT}/corpora"
BENCH="${DATA_ROOT}/benchmarks"

mkdir -p "${FONTS}" "${BG}/coco" "${BG}/dtd" "${BG}/places" "${CORPORA}" "${BENCH}"

# ── Google Fonts (~1500 families, ~500 MB) ──────────────────────────────────
if [ ! -d "${FONTS}/google-fonts" ]; then
    echo "Cloning Google Fonts..."
    git clone --depth 1 https://github.com/google/fonts.git "${FONTS}/google-fonts"
else
    echo "Google Fonts already present, skipping."
fi

# ── COCO train2017 (~18 GB) ─────────────────────────────────────────────────
if [ ! -d "${BG}/coco/train2017" ]; then
    echo "Downloading COCO train2017 (~18 GB)..."
    wget -q --show-progress -O "${BG}/coco/train2017.zip" \
        http://images.cocodataset.org/zips/train2017.zip
    unzip -q "${BG}/coco/train2017.zip" -d "${BG}/coco"
    rm "${BG}/coco/train2017.zip"
else
    echo "COCO train2017 already present, skipping."
fi

# ── DTD textures (~600 MB) ──────────────────────────────────────────────────
if [ ! -d "${BG}/dtd/images" ]; then
    echo "Downloading DTD..."
    wget -q --show-progress -O "${BG}/dtd-r1.0.1.tar.gz" \
        https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz
    tar -xzf "${BG}/dtd-r1.0.1.tar.gz" -C "${BG}"
    rm "${BG}/dtd-r1.0.1.tar.gz"
    # The archive extracts into ${BG}/dtd/, with the images under a subdir
else
    echo "DTD already present, skipping."
fi

# ── Wikitext-103 corpus ─────────────────────────────────────────────────────
if [ ! -d "${CORPORA}/wikitext" ]; then
    echo "Downloading Wikitext-103..."
    if [ -d ".venv" ]; then
        PY=".venv/bin/python"
    else
        PY="python"
    fi
    $PY - <<'EOF'
from datasets import load_dataset
ds = load_dataset('wikitext', 'wikitext-103-v1')
ds.save_to_disk('data/corpora/wikitext')
EOF
else
    echo "Wikitext already present, skipping."
fi

echo
echo "Setup complete. Manual steps remaining:"
echo "  - ICDAR 2013/2015 — register at https://rrc.cvc.uab.es/  →  ${BENCH}/icdar2013, ${BENCH}/icdar2015"
echo "  - COCO-Text       — https://bgshih.github.io/cocotext/   →  ${BENCH}/cocotext"
echo "  - Total-Text      — https://github.com/cs-chan/Total-Text-Dataset  →  ${BENCH}/totaltext"
