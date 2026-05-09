"""TextSampler tests."""
from __future__ import annotations

import string

from segocr.generator.text_sampler import TextSampler
from segocr.utils.charset import CHARSET_TIER1


def test_sample_returns_nonempty_string(text_sampler: TextSampler) -> None:
    text = text_sampler.sample_text()
    assert isinstance(text, str)
    assert len(text) > 0


def test_sample_only_returns_chars_in_charset(text_sampler: TextSampler) -> None:
    charset = set(CHARSET_TIER1)
    for _ in range(50):
        text = text_sampler.sample_text()
        assert all(c in charset for c in text), f"Off-charset char in {text!r}"


def test_no_corpus_falls_back_to_random(text_sampler: TextSampler) -> None:
    assert text_sampler.corpus == []  # no corpus_path in fixture config
    # Should still produce valid text
    text = text_sampler.sample_text()
    assert text


def test_apply_case_modes_distinct() -> None:
    config = {
        "corpus_path": None,
        "min_length": 5,
        "max_length": 5,
        "min_words_per_line": 1,
        "max_words_per_line": 1,
        "max_lines": 1,
        "case_distribution": {"lower": 1.0, "upper": 0.0, "mixed": 0.0, "title": 0.0},
        "rare_char_boost": 1.0,
    }
    sampler = TextSampler(config)
    for _ in range(10):
        text = sampler.sample_text()
        # Once case-applied, the alpha portion must be all lowercase.
        for c in text:
            if c.isalpha():
                assert c == c.lower()


def test_apply_case_upper() -> None:
    config = {
        "corpus_path": None,
        "min_length": 5,
        "max_length": 5,
        "min_words_per_line": 1,
        "max_words_per_line": 1,
        "max_lines": 1,
        "case_distribution": {"lower": 0.0, "upper": 1.0, "mixed": 0.0, "title": 0.0},
        "rare_char_boost": 1.0,
    }
    sampler = TextSampler(config)
    for _ in range(10):
        text = sampler.sample_text()
        for c in text:
            if c.isalpha():
                assert c == c.upper()


def test_sample_paragraph_returns_lines(text_sampler: TextSampler) -> None:
    lines = text_sampler.sample_paragraph()
    assert isinstance(lines, list)
    assert len(lines) >= 1
    assert len(lines) <= text_sampler.max_lines
    for line in lines:
        assert isinstance(line, str)


def test_update_counts_tracks_distribution(text_sampler: TextSampler) -> None:
    text_sampler.update_counts("ZZZZZ")
    dist = text_sampler.get_char_distribution()
    assert dist["Z"] == 1.0
    text_sampler.update_counts("AAAAA")
    dist = text_sampler.get_char_distribution()
    assert dist["Z"] == 0.5
    assert dist["A"] == 0.5


def test_rare_char_boost_increases_underrepresented_weights(
    text_sampler_config: dict,
) -> None:
    sampler = TextSampler(text_sampler_config)
    # Pump in a stream of 'A's to drive A above-target and other chars below
    sampler.update_counts("A" * 1000)
    weights = sampler._sampling_weights()
    char_to_weight = dict(zip(CHARSET_TIER1, weights, strict=True))
    # 'A' should be at base weight 1.0, others at boosted weight
    assert char_to_weight["A"] == 1.0
    boosted = [w for c, w in char_to_weight.items() if c != "A"]
    assert all(w == sampler.rare_char_boost for w in boosted)


def test_length_bounds_respected(text_sampler_config: dict) -> None:
    """Random-generated text must fit configured length bounds.

    We force corpus_path = None so all samples go through _generate_random.
    """
    config = {**text_sampler_config, "corpus_path": None}
    sampler = TextSampler(config)
    for _ in range(20):
        text = sampler.sample_text()
        # Allow off-by-some chars from charset filtering, but the upper
        # bound from random generation must hold.
        assert 1 <= len(text) <= sampler.max_length


def test_filter_to_charset_drops_offcharset_chars() -> None:
    config = {
        "corpus_path": None,
        "min_length": 1,
        "max_length": 10,
        "min_words_per_line": 1,
        "max_words_per_line": 1,
        "max_lines": 1,
        "case_distribution": {"lower": 0.0, "upper": 0.0, "mixed": 1.0, "title": 0.0},
        "rare_char_boost": 1.0,
    }
    sampler = TextSampler(config)
    filtered = sampler._filter_to_charset("Hello, World! 123")
    # Comma, exclamation, space all out of Tier 1 → dropped
    assert "," not in filtered
    assert "!" not in filtered
    assert " " not in filtered
    assert all(c in string.ascii_letters + string.digits for c in filtered)
