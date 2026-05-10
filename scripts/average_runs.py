"""Average model weights across multiple training runs.

Approach 1 multi-account workflow: each account trains its own model with
a distinct seed + distinct dataset, then we combine the final weights
into a single ensemble model. The math is the same operation that
``segocr.training.train.average_checkpoints`` already does for the
top-N checkpoints within a single run; this script generalizes it to
N runs.

Usage:
    # Average three checkpoints together
    python -m scripts.average_runs \\
        --checkpoints worker0/averaged_best.pth \\
                      worker1/averaged_best.pth \\
                      worker2/averaged_best.pth \\
        --output ensemble.pth

    # Or use a glob
    python -m scripts.average_runs \\
        --glob "runs/worker*/averaged_best.pth" \\
        --output ensemble.pth

When each input checkpoint has both ``model`` and ``ema`` keys, EMA
weights are preferred (typically lower variance, average better). When
only ``model`` is present (e.g., the ``averaged_best.pth`` files written
at end-of-training), the model state is used. The output file has both
``model`` and ``ema`` set to the same averaged state, so the standard
inference loaders work without modification.
"""
from __future__ import annotations

import argparse
import glob as _glob
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--checkpoints",
        nargs="+",
        type=Path,
        default=[],
        help="Explicit list of checkpoint paths to average.",
    )
    p.add_argument(
        "--glob",
        type=str,
        default=None,
        help="Glob pattern for checkpoints (e.g., 'runs/worker*/averaged_best.pth').",
    )
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Where to write the averaged checkpoint.",
    )
    p.add_argument(
        "--prefer",
        choices=["ema", "model", "auto"],
        default="auto",
        help="Which key to extract from each checkpoint. 'auto' = ema if "
             "present, model otherwise (default).",
    )
    return p.parse_args()


def _resolve_paths(args: argparse.Namespace) -> list[Path]:
    paths = list(args.checkpoints)
    if args.glob:
        paths.extend(Path(p) for p in sorted(_glob.glob(args.glob)))
    if not paths:
        raise SystemExit("No checkpoints provided. Pass --checkpoints or --glob.")
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise SystemExit(f"Missing checkpoint(s): {missing}")
    return paths


def _pick_state(ckpt: dict, prefer: str) -> dict:
    if prefer == "ema":
        return ckpt["ema"]
    if prefer == "model":
        return ckpt["model"]
    # auto
    return ckpt["ema"] if "ema" in ckpt else ckpt["model"]


def average_states(states: list[dict]) -> dict:
    """Average a list of state_dicts. Non-floating-point or non-tensor
    entries are taken from the first state verbatim."""
    avg: dict[str, torch.Tensor] = {}
    keys = list(states[0])
    for key in keys:
        first = states[0][key]
        if not torch.is_tensor(first):
            avg[key] = first
            continue
        if not first.is_floating_point():
            avg[key] = first
            continue
        stacked = torch.stack([s[key].float() for s in states])
        avg[key] = stacked.mean(dim=0).to(first.dtype)
    return avg


def main() -> None:
    args = parse_args()
    paths = _resolve_paths(args)
    print(f"Averaging {len(paths)} checkpoints with prefer='{args.prefer}':")
    for p in paths:
        print(f"  {p}")

    states: list[dict] = []
    for p in paths:
        ckpt = torch.load(p, map_location="cpu")
        state = _pick_state(ckpt, args.prefer)
        states.append(state)

    avg_state = average_states(states)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Save with both model and ema set to the same averaged state, so
    # the standard inference path picks it up regardless of which key
    # the loader looks at first.
    torch.save(
        {"model": avg_state, "ema": avg_state, "iteration": -1, "n_runs": len(paths)},
        args.output,
    )
    print(f"\nWrote {args.output}  ({len(paths)}-run ensemble)")


if __name__ == "__main__":
    main()
