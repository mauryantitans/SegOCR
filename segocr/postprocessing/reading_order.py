"""Reading-order recovery. Implementation Guide §3.11 / Research Proposal §7.3.

Latin-script ordering:
    (1) DBSCAN-style 1D clustering on y-centroids → line clusters.
    (2) Within each line, sort by x.
    (3) Order lines by y.
    (4) Word boundaries: gap > 1.5× median gap on that line → space.

Avoids sklearn so the postprocessing module has no heavy deps. The
1D clustering is just a sort + adjacent-gap split, which is what
DBSCAN with min_samples=1 would degenerate to in 1D.
"""
from __future__ import annotations

import numpy as np

from segocr.postprocessing.instance_extraction import CharacterInstance
from segocr.utils.charset import class_id_to_char


def recover_text(
    instances: list[CharacterInstance],
    affinity_map: np.ndarray | None = None,  # noqa: ARG001 — see below
    tier: int = 1,
) -> str:
    """Convert detected instances into a readable text string.

    Args:
        instances: output of ``extract_instances``.
        affinity_map: reserved for affinity-aware word splitting; the
            current implementation uses gap-based word breaks (1.5×
            median intra-line gap), which is robust in practice and
            doesn't require model output at decode time.
        tier: charset tier matching the model (default 1: A-Z a-z 0-9).

    Returns:
        Recognized text with ``\\n`` between detected lines and a single
        space between detected words.
    """
    if not instances:
        return ""

    id_to_char = class_id_to_char(tier)

    # ── Group into lines by y-centroid (1D adjacency clustering) ────────────
    sorted_by_y = sorted(instances, key=lambda i: i.centroid[1])
    median_h = float(np.median([inst.bbox[3] for inst in instances]))
    # A vertical gap > 0.5× median character height → new line.
    line_break_eps = max(median_h * 0.5, 1.0)

    lines: list[list[CharacterInstance]] = [[sorted_by_y[0]]]
    for inst in sorted_by_y[1:]:
        prev_y = lines[-1][-1].centroid[1]
        if inst.centroid[1] - prev_y > line_break_eps:
            lines.append([inst])
        else:
            lines[-1].append(inst)

    # ── Within each line: sort by x, insert word breaks where gap large ─────
    out_lines: list[str] = []
    for line in lines:
        line_sorted = sorted(line, key=lambda i: i.centroid[0])
        # Compute inter-character gaps (right-edge of prev → left-edge of next)
        gaps: list[float] = []
        for prev, nxt in zip(line_sorted, line_sorted[1:], strict=False):
            prev_right = prev.bbox[0] + prev.bbox[2]
            gap = nxt.bbox[0] - prev_right
            gaps.append(float(gap))
        # Word-break threshold: 1.5× median gap on this line (degenerates
        # to "no breaks" when fewer than 3 chars — too little signal).
        word_break_thresh = (
            1.5 * float(np.median(gaps)) if len(gaps) >= 2 else float("inf")
        )

        chunks: list[str] = []
        for idx, inst in enumerate(line_sorted):
            chunks.append(id_to_char.get(inst.class_id, ""))
            if idx < len(gaps) and gaps[idx] > word_break_thresh:
                chunks.append(" ")
        out_lines.append("".join(chunks))

    return "\n".join(out_lines)


def apply_language_model(
    text: str,
    confidences: list[float],  # noqa: ARG001 — Week 11+, optional
) -> str:
    """Optional spell-correction pass. Implementation Guide §3.11.

    Not implemented yet. Reserved for a Tier-2 contribution where we
    show ablation with/without LM correction.
    """
    return text
