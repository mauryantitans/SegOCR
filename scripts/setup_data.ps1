# SegOCR — data download script (PowerShell, Windows-friendly).
# Run from repo root:  .\scripts\setup_data.ps1
#
# Total disk: ~30 GB.
# Skips downloads that already exist on disk.

$ErrorActionPreference = "Stop"

$DataRoot = "data"
$Fonts    = "$DataRoot/fonts"
$Bg       = "$DataRoot/backgrounds"
$Corpora  = "$DataRoot/corpora"
$Bench    = "$DataRoot/benchmarks"

New-Item -ItemType Directory -Force -Path $Fonts, "$Bg/coco", "$Bg/dtd", "$Bg/places", $Corpora, $Bench | Out-Null

# ── Google Fonts (~1500 families, ~500 MB) ───────────────────────────────────
if (-not (Test-Path "$Fonts/google-fonts")) {
    Write-Host "Cloning Google Fonts..."
    git clone --depth 1 https://github.com/google/fonts.git "$Fonts/google-fonts"
} else {
    Write-Host "Google Fonts already present, skipping."
}

# ── COCO train2017 (~18 GB) ──────────────────────────────────────────────────
if (-not (Test-Path "$Bg/coco/train2017")) {
    Write-Host "Downloading COCO train2017 (~18 GB)..."
    $cocoZip = "$Bg/coco/train2017.zip"
    Invoke-WebRequest -Uri "http://images.cocodataset.org/zips/train2017.zip" -OutFile $cocoZip
    Expand-Archive -Path $cocoZip -DestinationPath "$Bg/coco" -Force
    Remove-Item $cocoZip
} else {
    Write-Host "COCO train2017 already present, skipping."
}

# ── DTD textures (~600 MB) ──────────────────────────────────────────────────
if (-not (Test-Path "$Bg/dtd/images")) {
    Write-Host "Downloading DTD..."
    $dtdTar = "$Bg/dtd-r1.0.1.tar.gz"
    Invoke-WebRequest -Uri "https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz" -OutFile $dtdTar
    tar -xzf $dtdTar -C $Bg
    Move-Item "$Bg/dtd/images" "$Bg/dtd-images-tmp" -ErrorAction SilentlyContinue
    Remove-Item $dtdTar
} else {
    Write-Host "DTD already present, skipping."
}

# ── Wikitext-103 corpus ──────────────────────────────────────────────────────
if (-not (Test-Path "$Corpora/wikitext")) {
    Write-Host "Downloading Wikitext-103..."
    .venv/Scripts/python -c "from datasets import load_dataset; ds = load_dataset('wikitext', 'wikitext-103-v1'); ds.save_to_disk('$Corpora/wikitext')"
} else {
    Write-Host "Wikitext already present, skipping."
}

Write-Host ""
Write-Host "Setup complete. Manual steps remaining:"
Write-Host "  - ICDAR 2013/2015 — register at https://rrc.cvc.uab.es/  →  $Bench/icdar2013, $Bench/icdar2015"
Write-Host "  - COCO-Text       — https://bgshih.github.io/cocotext/   →  $Bench/cocotext"
Write-Host "  - Total-Text      — https://github.com/cs-chan/Total-Text-Dataset  →  $Bench/totaltext"
