# SegOCR — Kaggle Parallel Ensemble Workflow

Five Kaggle accounts, each training one worker, ~9 hours per session. Final ensemble averages the 5 trained models.

## Files in this folder

| Notebook | Where it runs | Purpose |
|---|---|---|
| `00_generate_dataset.ipynb` | **CPU** account #1, one time | Generates 50K samples (5 × 10K worker slices, ~18 GB) and publishes as a Kaggle Dataset. **Sized to fit Kaggle's 20 GB `/kaggle/working/` cap.** |
| `01_train_worker0.ipynb` | **GPU T4** account #1 | Trains worker 0 (`WORKER_ID` hardcoded) |
| `02_train_worker1.ipynb` | **GPU T4** account #2 | Trains worker 1 |
| `03_train_worker2.ipynb` | **GPU T4** account #3 | Trains worker 2 |
| `04_train_worker3.ipynb` | **GPU T4** account #4 | Trains worker 3 |
| `05_train_worker4.ipynb` | **GPU T4** account #5 | Trains worker 4 |
| `_build_train_notebooks.py` | local | Regenerates the 5 trainer notebooks if you edit the template |
| `_build_generate_notebook.py` | local | Regenerates the generator notebook |

---

## End-to-end flow

### Phase 1 — Generate the shared dataset (one-time, ~4 hr)

On **account #1**:

1. Open Kaggle → New Notebook → **File → Import Notebook** → upload `00_generate_dataset.ipynb`.
2. Settings panel:
   - Accelerator: **None** (CPU is fine; generator isn't GPU-bound).
   - Persistence: any.
3. Click **Save Version → Save & Run All**. This runs server-side so connection issues don't kill it.
4. Wait ~4 hours.
5. When the run completes, find the **Output** tab. Click **New Dataset**:
   - Slug: `segocr-ensemble-50k`
   - Visibility: **Public** (simplest for cross-account sharing). Or "Shared with specific users" if you want it private.
6. Note the dataset URL (e.g., `https://www.kaggle.com/datasets/USERNAME/segocr-ensemble-50k`).

### Phase 2 — Train 5 workers in parallel (~9 hr each, all running concurrently)

For each account `N` in `0..4`:

1. Sign in to Kaggle account #N.
2. Open Kaggle → New Notebook → **File → Import Notebook** → upload `0{N+1}_train_worker{N}.ipynb` (e.g., `01_train_worker0.ipynb` on account #1).
3. **Add Data** (sidebar) → search for `segocr-ensemble-50k` → attach. It mounts at `/kaggle/input/segocr-ensemble-50k/`.
4. Settings panel:
   - Accelerator: **GPU T4**
   - Persistence: any
5. Click **Save Version → Save & Run All**.
6. Wait ~9 hours. The session runs server-side; you can close the browser tab.

The 5 workers run on 5 independent Kaggle accounts simultaneously — fully parallel.

### Phase 2b — If a session disconnects mid-training (resume)

Kaggle's "Save & Run All" headless mode is robust; rare to hit. If you do:

1. From the failed notebook's page, click **View Output** to see what was saved (typically partial checkpoints at `/kaggle/working/weights/checkpoint_NNNNNN.pth`).
2. Click **Edit** to open the notebook.
3. **Add Data → Notebook Output Files** → attach the previous version's output.
4. Run cells again. Cell 4 (resume) auto-detects the attached previous checkpoints, copies them to `/kaggle/working/weights/`, and `--resume-latest` picks up where you left off.

### Phase 3 — Download checkpoints and ensemble (on your laptop, ~10 min)

After all 5 workers finish, download each one's output `averaged_best.pth`:

**Option A — Kaggle UI (5 manual downloads)**:
- For each worker notebook: open the notebook page → **Output** tab → download `weights/averaged_best.pth`. Rename to `worker{N}_best.pth`.

**Option B — Kaggle CLI (one command per worker, scriptable)**:
```bash
# One-time setup on your laptop:
pip install kaggle
# Get your API token from https://www.kaggle.com/settings/account → "Create New Token"
# Save the downloaded kaggle.json to:
#   Windows: %USERPROFILE%\.kaggle\kaggle.json
#   Linux/macOS: ~/.kaggle/kaggle.json

# Download per worker (replace YOUR_USERNAME and notebook slugs):
kaggle kernels output USERNAME_OF_ACCOUNT_1/notebook-slug-1 -p ./downloads/worker0
kaggle kernels output USERNAME_OF_ACCOUNT_2/notebook-slug-2 -p ./downloads/worker1
# ... etc for workers 2, 3, 4
```

Then on your laptop, in the repo root:

```bash
# Build the 5-way ensemble:
.venv\Scripts\python -m scripts.average_runs \
    --glob "downloads/worker*/weights/averaged_best.pth" \
    --output downloads/ensemble.pth

# Or with explicit paths:
.venv\Scripts\python -m scripts.average_runs \
    --checkpoints downloads/worker0/weights/averaged_best.pth \
                  downloads/worker1/weights/averaged_best.pth \
                  downloads/worker2/weights/averaged_best.pth \
                  downloads/worker3/weights/averaged_best.pth \
                  downloads/worker4/weights/averaged_best.pth \
    --output downloads/ensemble.pth
```

`ensemble.pth` has both `model` and `ema` keys set to the averaged state, so it loads via the same path as any single-run checkpoint.

### Phase 4 — Evaluate the ensemble (optional)

To get the headline numbers (per-class IoU, decoded text on val samples, etc.):

- **Locally** if you have a GPU.
- **On Kaggle** — create a small new notebook on any account, attach the original `segocr-ensemble-50k` dataset AND upload `ensemble.pth` as an additional dataset/file, then copy cells 8–12 from `notebooks/segocr_colab_longrun.ipynb`. Eval is fast (<30 min on T4) so this barely dents your weekly GPU quota.

---

## What each worker trains on (parameter reference)

All 5 worker notebooks share the same training config — only `WORKER_ID` and the derived `TRAIN_SEED = WORKER_ID + 1` differ. This guarantees:

- **Different data slices** — worker N reads from `/kaggle/input/segocr-ensemble-50k/workerN/`, generated with deterministic per-index seeds. Indices are mathematically disjoint across workers (worker 0 gets indices 0–15999, worker 1 gets 16000–31999, etc.).
- **Different initialization** — `--seed N+1` seeds Python random, NumPy, PyTorch (CPU + CUDA) and DataLoader workers. Each model lands in a different basin of the loss landscape.

Locked training parameters (in every train notebook):

| Setting | Value |
|---|---|
| Image size | 512 × 512 |
| Min font size | 40 px |
| Encoder | ResNet-18 (ImageNet-pretrained) |
| Batch size | 16 |
| Total iterations | 30,000 |
| Warmup | 1,000 iters |
| Learning rate | 3e-4 (AdamW, polynomial decay) |
| Mixed precision | on |
| EMA | on (decay 0.999) |
| Top-N checkpoint averaging | on (top 3) |
| Per-worker samples | 10,000 (sized to fit Kaggle's 20 GB working-dir cap) |
| `rare_char_boost` | 4.0 |
| Layout: paragraph mode | disabled (0%) |
| Text length | max 20 chars, max 3 words |

If you need to change any of these, edit `_build_train_notebooks.py` and re-run it — that regenerates all 5 trainer notebooks consistently.

---

## Realistic expectations

| Setup | fg_miou | `j` IoU | Decoded text match |
|---|---|---|---|
| 1 Colab worker (4 hr, 256²) — your previous experiment | ~0.36 | 0.02 | ~30% |
| 1 Kaggle worker (9 hr, 512²) | ~0.45 | likely >0.30 | ~60% |
| **5-Kaggle-worker ensemble** | **~0.55–0.65** | likely >0.40 | **~75%+** |

These are estimates based on extrapolation from the Colab run. Actual numbers depend on convergence behavior and how diverse the 5 workers turn out to be.

---

## TOS reminder

Using multiple Kaggle accounts is in a gray area of their TOS. To minimize the risk of flagging:

- Use distinct browser profiles (or different devices) per account.
- Don't rapidly switch between accounts in the same session.
- Avoid running notebooks that explicitly cross-link accounts.

Each individual account uses ~9 hr of its 30 hr/week GPU quota — well within limits.
