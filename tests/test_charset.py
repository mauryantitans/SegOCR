"""Sanity tests for the canonical character-set definitions."""
from __future__ import annotations

from segocr.utils.charset import (
    CHARSET_TIER1,
    CHARSET_TIER2,
    CHARSET_TIER3,
    char_to_class_id,
    class_id_to_char,
)


def test_tier1_is_62_alphanumeric() -> None:
    assert len(CHARSET_TIER1) == 62
    # 26 + 26 + 10
    assert all(c.isalnum() for c in CHARSET_TIER1)


def test_tier2_extends_tier1() -> None:
    assert CHARSET_TIER2[: len(CHARSET_TIER1)] == CHARSET_TIER1
    assert len(CHARSET_TIER2) == 62 + 15


def test_tier3_extends_tier2() -> None:
    assert CHARSET_TIER3[: len(CHARSET_TIER2)] == CHARSET_TIER2


def test_class_ids_are_dense_and_one_indexed() -> None:
    for tier in (1, 2, 3):
        m = char_to_class_id(tier)
        ids = sorted(m.values())
        assert ids == list(range(1, len(ids) + 1))


def test_class_id_round_trip() -> None:
    for tier in (1, 2, 3):
        forward = char_to_class_id(tier)
        backward = class_id_to_char(tier)
        assert backward[0] == ""
        for char, cls_id in forward.items():
            assert backward[cls_id] == char


def test_no_duplicate_class_ids() -> None:
    for tier in (1, 2, 3):
        m = char_to_class_id(tier)
        assert len(set(m.values())) == len(m)
