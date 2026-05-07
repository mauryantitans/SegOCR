"""Reading-order recovery. Implementation Guide §3.11 / Research Proposal §7.3.

Latin-script ordering:
    (1) DBSCAN on y-coordinates → line clusters.
    (2) Within each line, sort by x.
    (3) Order lines by y.
    (4) Word boundaries: affinity head if available, else gap analysis
        (gap > 1.5× median spacing = word break).
"""
from __future__ import annotations

import numpy as np

from segocr.postprocessing.instance_extraction import CharacterInstance


def recover_text(
    instances: list[CharacterInstance],
    affinity_map: np.ndarray | None = None,
) -> str:
    """Convert detected instances into a readable text string.

    Returns the recognized text with newlines between detected lines.
    """
    raise NotImplementedError("recover_text — Week 6")


def apply_language_model(text: str, confidences: list[float]) -> str:
    """Optional spell-correction pass. Implementation Guide §3.11.

    Replace low-confidence chars when edit distance ≤ 2 from a dict word.
    Expected gain: +2–5% word-level accuracy.
    """
    raise NotImplementedError("apply_language_model — Week 6")
