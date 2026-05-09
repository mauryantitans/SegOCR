"""Text sampler — generate strings with controlled character distribution.

Implementation Guide §3.2 + Research Proposal §4.2.

Strategy:
    - 70% sampled from a text corpus (Wikipedia/Project Gutenberg/etc.) for
      natural character distribution and realistic word patterns.
    - 30% randomly generated, weighted to oversample currently-rare
      characters in the running dataset distribution. This is what closes
      the gap on rare letters like 'z', 'q', 'x'.
    - Case mode (lower / upper / mixed / title) sampled per the configured
      distribution.

Tracks running per-character frequency so the rare-char boost adapts to
the actual distribution as data is generated.
"""
from __future__ import annotations

import logging
import random
from collections import Counter
from pathlib import Path

from segocr.utils.charset import CHARSET_TIER1

logger = logging.getLogger(__name__)

CORPUS_FRACTION = 0.7      # 70% corpus, 30% random
MIN_SENTENCE_LEN = 5
MAX_SENTENCE_LEN = 500
MAX_LOADED_SENTENCES = 200_000


class TextSampler:
    """Generates text content for rendering."""

    def __init__(
        self,
        config: dict,
        charset: tuple[str, ...] = CHARSET_TIER1,
    ) -> None:
        self.config = config
        self.charset = charset
        self.charset_set = set(charset)
        self.corpus_path = (
            Path(config["corpus_path"]).expanduser()
            if config.get("corpus_path")
            else None
        )
        self.min_length = int(config["min_length"])
        self.max_length = int(config["max_length"])
        self.min_words_per_line = int(config["min_words_per_line"])
        self.max_words_per_line = int(config["max_words_per_line"])
        self.max_lines = int(config["max_lines"])
        self.case_distribution: dict[str, float] = dict(config["case_distribution"])
        self.rare_char_boost = float(config["rare_char_boost"])

        self.corpus: list[str] = self._load_corpus()
        self.char_counts: Counter[str] = Counter()
        self.total_chars: int = 0

        if not self.corpus:
            logger.info(
                "No corpus loaded from %s — falling back to 100%% random text.",
                self.corpus_path,
            )

    # ── Public API ──────────────────────────────────────────────────────────

    def sample_text(self) -> str:
        """Sample one text instance.

        70% corpus-sourced, 30% random-generated, then case-transformed
        and filtered to the active charset. Empty strings are retried
        once before returning a single random char as last-resort.
        """
        for _ in range(2):
            use_corpus = self.corpus and random.random() < CORPUS_FRACTION
            text = self._sample_from_corpus() if use_corpus else self._generate_random()
            text = self._apply_case(text)
            text = self._filter_to_charset(text)
            if text:
                return text
        return random.choice(self.charset)

    def sample_paragraph(self) -> list[str]:
        """Sample a multi-line paragraph for layout Mode 6.

        Returns a list of lines (1..max_lines lines, each
        min_words_per_line..max_words_per_line words).
        """
        n_lines = random.randint(1, self.max_lines)
        lines = []
        for _ in range(n_lines):
            n_words = random.randint(self.min_words_per_line, self.max_words_per_line)
            words = []
            for _ in range(n_words):
                word_len = random.randint(2, 10)
                words.append(self._generate_random_word(word_len))
            line = " ".join(words)
            line = self._apply_case(line)
            line = self._filter_to_charset(line + " ")  # keep space; will be re-stripped
            lines.append(line.strip())
        return [line for line in lines if line]

    def update_counts(self, text: str) -> None:
        """Record characters that just got committed to a generated image,
        so the rare-char boost adapts as the dataset grows."""
        self.char_counts.update(text)
        self.total_chars += len(text)

    def get_char_distribution(self) -> dict[str, float]:
        """Return current per-character frequency in the generated dataset."""
        if self.total_chars == 0:
            return {c: 0.0 for c in self.charset}
        return {
            c: self.char_counts.get(c, 0) / self.total_chars for c in self.charset
        }

    # ── Internal ────────────────────────────────────────────────────────────

    def _load_corpus(self) -> list[str]:
        if not self.corpus_path or not self.corpus_path.exists():
            return []

        # Try HuggingFace `datasets` format first; fall back to plain text.
        try:
            from datasets import load_from_disk

            ds = load_from_disk(str(self.corpus_path))
            split = ds["train"] if "train" in ds else ds[next(iter(ds))]
            sentences = []
            for row in split.select(
                range(min(len(split), MAX_LOADED_SENTENCES))
            )["text"]:
                stripped = row.strip()
                if MIN_SENTENCE_LEN <= len(stripped) <= MAX_SENTENCE_LEN:
                    sentences.append(stripped)
            logger.info("Loaded %d sentences from %s", len(sentences), self.corpus_path)
            return sentences
        except Exception as exc:  # noqa: BLE001
            logger.debug("datasets load failed (%s) — trying plain text.", exc)

        try:
            sentences = []
            with open(self.corpus_path, encoding="utf-8") as f:
                for line in f:
                    stripped = line.strip()
                    if MIN_SENTENCE_LEN <= len(stripped) <= MAX_SENTENCE_LEN:
                        sentences.append(stripped)
                    if len(sentences) >= MAX_LOADED_SENTENCES:
                        break
            logger.info("Loaded %d sentences (plain text) from %s",
                        len(sentences), self.corpus_path)
            return sentences
        except OSError as exc:
            logger.warning("Could not load corpus %s: %s", self.corpus_path, exc)
            return []

    def _sample_from_corpus(self) -> str:
        sentence = random.choice(self.corpus)
        upper_bound = min(self.max_length, len(sentence))
        if upper_bound < self.min_length:
            return sentence
        target_len = random.randint(self.min_length, upper_bound)
        if len(sentence) <= target_len:
            return sentence
        start = random.randint(0, len(sentence) - target_len)
        return sentence[start : start + target_len]

    def _generate_random(self) -> str:
        n = random.randint(self.min_length, self.max_length)
        weights = self._sampling_weights()
        return "".join(random.choices(self.charset, weights=weights, k=n))

    def _generate_random_word(self, length: int) -> str:
        weights = self._sampling_weights()
        return "".join(random.choices(self.charset, weights=weights, k=length))

    def _sampling_weights(self) -> list[float]:
        """Boost characters whose running frequency is below uniform."""
        target = 1.0 / len(self.charset)
        denom = max(1, self.total_chars)
        return [
            self.rare_char_boost
            if (self.char_counts.get(c, 0) / denom) < target
            else 1.0
            for c in self.charset
        ]

    def _apply_case(self, text: str) -> str:
        modes = list(self.case_distribution.keys())
        weights = list(self.case_distribution.values())
        mode = random.choices(modes, weights=weights, k=1)[0]
        if mode == "lower":
            return text.lower()
        if mode == "upper":
            return text.upper()
        if mode == "title":
            return text.title()
        return text  # "mixed" leaves text alone

    def _filter_to_charset(self, text: str) -> str:
        """Drop characters not in the active charset."""
        return "".join(c for c in text if c in self.charset_set)
