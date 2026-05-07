"""Canonical character set + class-id maps.

Tier 1 (default, 63 classes): A-Z, a-z, 0-9 + background.
Tier 2 (78):  Tier 1 + 15 punctuation: . , ! ? : ; ' " - _ ( ) / + &
Tier 3 (95+): Tier 2 + accented Latin (à, é, ñ, ü, …).

Class 0 is always background. Character → class-id is the canonical
mapping used by the renderer (oracle), the dataset loader, and the
post-processing reading-order recovery.
"""
from __future__ import annotations

import string

# Tier 1 — 62 characters, classes 1..62
CHARSET_TIER1: tuple[str, ...] = tuple(
    string.ascii_uppercase + string.ascii_lowercase + string.digits
)

# Tier 2 — Tier 1 + 15 common punctuation
_TIER2_PUNCT = (".", ",", "!", "?", ":", ";", "'", '"', "-", "_", "(", ")", "/", "+", "&")
CHARSET_TIER2: tuple[str, ...] = CHARSET_TIER1 + _TIER2_PUNCT

# Tier 3 — Tier 2 + a basic accented Latin set
_TIER3_ACCENTED = (
    "à", "á", "â", "ä", "ã", "å",
    "è", "é", "ê", "ë",
    "ì", "í", "î", "ï",
    "ò", "ó", "ô", "ö", "õ",
    "ù", "ú", "û", "ü",
    "ñ", "ç",
    "À", "Á", "Â", "Ä", "Ã", "Å",
    "È", "É", "Ê", "Ë",
    "Ì", "Í", "Î", "Ï",
    "Ò", "Ó", "Ô", "Ö", "Õ",
    "Ù", "Ú", "Û", "Ü",
    "Ñ", "Ç",
)
CHARSET_TIER3: tuple[str, ...] = CHARSET_TIER2 + _TIER3_ACCENTED


def _build_map(charset: tuple[str, ...]) -> dict[str, int]:
    """Char → class-id, with background = 0."""
    return {char: i + 1 for i, char in enumerate(charset)}


_TIER_TO_CHARSET = {
    1: CHARSET_TIER1,
    2: CHARSET_TIER2,
    3: CHARSET_TIER3,
}


def char_to_class_id(tier: int = 1) -> dict[str, int]:
    """Return the char → class-id map for the requested tier."""
    if tier not in _TIER_TO_CHARSET:
        raise ValueError(f"Unknown character set tier: {tier}")
    return _build_map(_TIER_TO_CHARSET[tier])


def class_id_to_char(tier: int = 1) -> dict[int, str]:
    """Return class-id → char map. Class 0 → '' (background)."""
    forward = char_to_class_id(tier)
    return {0: "", **{v: k for k, v in forward.items()}}
