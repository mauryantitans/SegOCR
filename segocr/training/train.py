"""Training loop.

Implementation Guide §3.10. AdamW + polynomial LR decay + linear warmup.
Mixed precision via torch.cuda.amp on GPU. Logs per-class IoU at every
eval step (not just mean — Implementation Guide §6 gotcha #7) so we can
spot rare-character regression early.

Mitigations for the across-epoch instability seen in the pilot:
  - EMA of weights via torch.optim.swa_utils.AveragedModel.
  - Top-3 checkpoint averaging at end of training.
"""
from __future__ import annotations

import contextlib
import logging
import time
from pathlib import Path

import torch
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader
from tqdm import tqdm

from segocr.models.losses import SegOCRLoss
from segocr.models.unet import build_model
from segocr.training.dataset import SegOCRDataset, collate_fn
from segocr.training.evaluator import Evaluator
from segocr.utils.config import load_config

logger = logging.getLogger(__name__)


def train(config_path: str | Path) -> None:
    """Train SegOCR end-to-end.

    Reads the YAML config, builds the dataset / model / optimizer /
    scheduler, runs a fixed-iteration training loop with periodic
    evaluation, and saves checkpoints. EMA weights are tracked
    throughout; top-N checkpoint averaging happens at the end.
    """
    config = load_config(config_path)
    train_cfg = config["training"]
    model_cfg = config["model"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training on device: %s", device)

    # ── Data ────────────────────────────────────────────────────────────────
    data_dir = Path(config["generator"]["output_dir"])
    train_ds = SegOCRDataset(data_dir, split="train", train_aug=True)
    val_ds = SegOCRDataset(data_dir, split="val", train_aug=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(train_cfg.get("num_workers", 0)),
        pin_memory=device.type == "cuda",
        collate_fn=collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(train_cfg.get("num_workers", 0)),
        pin_memory=device.type == "cuda",
        collate_fn=collate_fn,
    )

    # ── Model + EMA ─────────────────────────────────────────────────────────
    model = build_model(model_cfg).to(device)
    ema_cfg = train_cfg.get("ema", {}) or {}
    ema_model: AveragedModel | None = None
    if ema_cfg.get("enabled", False):
        ema_decay = float(ema_cfg.get("decay", 0.999))
        ema_model = AveragedModel(
            model,
            avg_fn=_make_ema_avg_fn(ema_decay),
        ).to(device)

    # ── Optimizer + scheduler + criterion ──────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
    )
    total_iters = int(train_cfg["total_iters"])
    warmup_iters = int(train_cfg.get("warmup_iters", 1500))
    scheduler = _make_scheduler(optimizer, warmup_iters, total_iters)
    criterion = SegOCRLoss(model_cfg["loss"], num_classes=int(model_cfg["num_classes"]))
    criterion = criterion.to(device)

    use_amp = bool(train_cfg.get("mixed_precision", True)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    evaluator = Evaluator(num_classes=int(model_cfg["num_classes"]), device=device)

    # ── wandb (optional) ────────────────────────────────────────────────────
    wandb_run = _maybe_init_wandb(config)

    # ── Output dirs ─────────────────────────────────────────────────────────
    output_dir = Path(config.get("training", {}).get("output_dir", "weights"))
    output_dir.mkdir(parents=True, exist_ok=True)
    keep_best_n = int(train_cfg.get("keep_best_n", 3))
    best_checkpoints: list[tuple[float, Path]] = []  # (val_miou, path)

    # ── Loop ────────────────────────────────────────────────────────────────
    iteration = 0
    eval_interval = int(train_cfg.get("eval_interval", 2500))
    save_interval = int(train_cfg.get("save_interval", 5000))
    log_interval = int(train_cfg.get("log_interval", 100))
    epoch = 0
    t0 = time.perf_counter()

    pbar = tqdm(total=total_iters, desc="train")
    while iteration < total_iters:
        epoch += 1
        for batch in train_loader:
            if iteration >= total_iters:
                break
            iteration += 1
            images = batch["image"].to(device, non_blocking=True)
            targets = {k: v.to(device, non_blocking=True) for k, v in batch["targets"].items()}

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                predictions = model(images)
                loss, loss_terms = criterion(predictions, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            if ema_model is not None:
                ema_model.update_parameters(model)

            if iteration % log_interval == 0:
                lr = optimizer.param_groups[0]["lr"]
                _log_step(wandb_run, iteration, loss_terms, lr)

            if iteration % eval_interval == 0:
                eval_target = ema_model.module if ema_model is not None else model
                metrics = evaluator.evaluate(eval_target, val_loader)
                _log_eval(wandb_run, iteration, metrics)
                miou = float(metrics.get("miou", 0.0))
                ckpt_path = output_dir / f"checkpoint_{iteration:06d}.pth"
                _save_checkpoint(model, ema_model, optimizer, iteration, ckpt_path)
                best_checkpoints = _track_best(best_checkpoints, miou, ckpt_path, keep_best_n)
                pbar.set_postfix(miou=f"{miou:.3f}")

            if iteration % save_interval == 0:
                ckpt_path = output_dir / f"snapshot_{iteration:06d}.pth"
                _save_checkpoint(model, ema_model, optimizer, iteration, ckpt_path)

            pbar.update(1)

        if epoch > total_iters:  # safety
            break

    pbar.close()
    elapsed = time.perf_counter() - t0
    logger.info(
        "Training finished: %d iterations in %.1f min", total_iters, elapsed / 60.0
    )

    # ── Top-N checkpoint averaging ──────────────────────────────────────────
    avg_cfg = train_cfg.get("checkpoint_averaging", {}) or {}
    if avg_cfg.get("enabled", False) and len(best_checkpoints) >= 2:
        avg_path = output_dir / "averaged_best.pth"
        average_checkpoints([p for _, p in best_checkpoints], avg_path)
        logger.info("Averaged %d checkpoints to %s", len(best_checkpoints), avg_path)

    if wandb_run is not None:
        wandb_run.finish()


# ── Helpers ─────────────────────────────────────────────────────────────────


def _make_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_iters: int,
    total_iters: int,
    power: float = 0.9,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup followed by polynomial decay."""

    def lr_lambda(step: int) -> float:
        if step < warmup_iters:
            return float(step) / float(max(1, warmup_iters))
        progress = (step - warmup_iters) / max(1, total_iters - warmup_iters)
        return max(0.0, (1.0 - progress) ** power)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _make_ema_avg_fn(decay: float):
    def avg_fn(
        averaged_param: torch.Tensor,
        param: torch.Tensor,
        _num_averaged: torch.Tensor,
    ) -> torch.Tensor:
        return decay * averaged_param + (1.0 - decay) * param

    return avg_fn


def _save_checkpoint(
    model: torch.nn.Module,
    ema_model: AveragedModel | None,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    path: Path,
) -> None:
    state = {
        "iteration": iteration,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    if ema_model is not None:
        state["ema"] = ema_model.module.state_dict()
    torch.save(state, path)


def _track_best(
    best: list[tuple[float, Path]],
    miou: float,
    path: Path,
    keep_n: int,
) -> list[tuple[float, Path]]:
    best.append((miou, path))
    best.sort(key=lambda t: -t[0])
    if len(best) > keep_n:
        # Delete the dropped checkpoints
        for _, dropped_path in best[keep_n:]:
            if dropped_path.exists() and dropped_path != path:
                with contextlib.suppress(OSError):
                    dropped_path.unlink()
        best = best[:keep_n]
    return best


def average_checkpoints(checkpoint_paths: list[Path], output_path: Path) -> None:
    """Average state_dicts of the given checkpoints, save to ``output_path``."""
    if not checkpoint_paths:
        raise ValueError("No checkpoints to average")
    states = [torch.load(p, map_location="cpu")["model"] for p in checkpoint_paths]
    avg_state: dict[str, torch.Tensor] = {}
    for key in states[0]:
        if not torch.is_tensor(states[0][key]):
            avg_state[key] = states[0][key]
            continue
        if not states[0][key].is_floating_point():
            avg_state[key] = states[0][key]
            continue
        stacked = torch.stack([s[key].float() for s in states])
        avg_state[key] = stacked.mean(dim=0).to(states[0][key].dtype)
    torch.save({"model": avg_state, "iteration": -1}, output_path)


def _maybe_init_wandb(config: dict):
    wandb_cfg = config.get("training", {}).get("wandb", {}) or {}
    if not wandb_cfg.get("project"):
        return None
    try:
        import wandb

        run = wandb.init(
            project=wandb_cfg["project"],
            entity=wandb_cfg.get("entity"),
            config=config,
        )
        return run
    except Exception as exc:  # noqa: BLE001
        logger.warning("wandb init failed (%s); continuing without it.", exc)
        return None


def _log_step(wandb_run, iteration: int, loss_terms: dict, lr: float) -> None:
    if wandb_run is None:
        return
    payload: dict[str, float] = {f"loss/{k}": float(v.item()) for k, v in loss_terms.items()}
    payload["lr"] = float(lr)
    wandb_run.log(payload, step=iteration)


def _log_eval(wandb_run, iteration: int, metrics: dict[str, float]) -> None:
    if wandb_run is None:
        logger.info("eval @ %d: %s", iteration, _format_metrics(metrics))
        return
    payload = {f"val/{k}": float(v) for k, v in metrics.items()}
    wandb_run.log(payload, step=iteration)


def _format_metrics(metrics: dict[str, float]) -> str:
    items = sorted(metrics.items())
    return " ".join(f"{k}={v:.3f}" for k, v in items[:6])


def evaluate(model: torch.nn.Module, val_loader: DataLoader, device) -> dict[str, float]:
    """Convenience wrapper used by external callers."""
    evaluator = Evaluator(num_classes=int(model.num_classes), device=device)
    return evaluator.evaluate(model, val_loader)


def save_checkpoint(*args, **kwargs):
    return _save_checkpoint(*args, **kwargs)
