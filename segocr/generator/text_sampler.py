"""Text sampler — generate strings with controlled character distribution.

Implementation Guide §3.2 + Research Proposal §4.2.

Strategy:
    - Multiple weighted corpora (signs / receipts / names / numbers /
      general) so generated text is *semantically* meaningful instead of
      random ABCDEFG noise. The bundled mini-corpora ship with the
      package; users can add larger external corpora (Wikitext, Project
      Gutenberg) by listing them in ``text.corpus_paths``.
    - 70% sampled from the selected corpus, 30% randomly generated with
      rare-character oversampling so under-represented chars (z, q, x)
      reach a sane frequency.
    - Case mode (lower / upper / mixed / title) sampled per the
      configured distribution.

Tracks running per-character frequency so the rare-char boost adapts to
the actual generated distribution.
"""
from __future__ import annotations

import logging
import random
from collections import Counter
from pathlib import Path

from segocr.assets.corpora import BUNDLED_CORPORA, get_bundled_corpus_path
from segocr.utils.charset import CHARSET_TIER1

logger = logging.getLogger(__name__)

CORPUS_FRACTION = 0.7  # 70% corpus, 30% random
MIN_SENTENCE_LEN = 2
MAX_SENTENCE_LEN = 500
MAX_LOADED_SENTENCES_PER_CORPUS = 200_000


class TextSampler:
    """Generates text content for rendering.

    Multi-corpus sampling: at sample time the sampler picks one corpus
    weighted by its configured weight, then samples a sentence from that
    corpus. Falls back to bundled mini-corpora if no external paths are
    configured.
    """

    def __init__(
        self,
        config: dict,
        charset: tuple[str, ...] = CHARSET_TIER1,
    ) -> None:
        self.config = config
        self.charset = charset
        self.charset_set = set(charset)
        self.min_length = int(config["min_length"])
        self.max_length = int(config["max_length"])
        self.min_words_per_line = int(config["min_words_per_line"])
        self.max_words_per_line = int(config["max_words_per_line"])
        self.max_lines = int(config["max_lines"])
        self.case_distribution: dict[str, float] = dict(config["case_distribution"])
        self.rare_char_boost = float(config["rare_char_boost"])

        self.corpora: list[tuple[str, list[str], float]] = self._load_corpora()
        self.char_counts: Counter[str] = Counter()
        self.total_chars: int = 0

        if not self.corpora:
            logger.info("No corpora loaded — falling back to 100%% random text.")

    # ── Public API ──────────────────────────────────────────────────────────

    def sample_text(self) -> str:
        """Sample one text instance.

        70% corpus-sourced, 30% random-generated, then case-transformed
        and filtered to the active charset. Empty results are retried
        once before falling back to a single random char.
        """
        for _ in range(2):
            use_corpus = self.corpora and random.random() < CORPUS_FRACTION
            text = self._sample_from_corpus() if use_corpus else self._generate_random()
            text = self._apply_case(text)
            text = self._filter_to_charset(text)
            if text:
                return text
        return random.choice(self.charset)

    def sample_paragraph(self) -> list[str]:
        """Sample a multi-line paragraph for layout Mode 6.

        Returns a list of lines (1..max_lines lines, each
        min_words_per_line..max_words_per_line words). Each line is
        case-transformed and filtered to the active charset.
        """
        n_lines = random.randint(1, self.max_lines)
        lines: list[str] = []
        for _ in range(n_lines):
            n_words = random.randint(self.min_words_per_line, self.max_words_per_line)
            words: list[str] = []
            for _ in range(n_words):
                if self.corpora and random.random() < CORPUS_FRACTION:
                    words.append(self._sample_word_from_corpus())
                else:
                    word_len = random.randint(2, 10)
                    words.append(self._generate_random_word(word_len))
            line = " ".join(words)
            line = self._apply_case(line)
            line = self._filter_to_charset(line)
            if line:
                lines.append(line)
        return lines

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

    # ── Internal — corpus loading ──────────────────────────────────────────

    def _load_corpora(self) -> list[tuple[str, list[str], float]]:
        """Load all configured corpora.

        Config schema (preferred):
            text.corpus_paths:
              - { path: "data/corpora/wikitext", tag: "general", weight: 0.4 }
              - { path: "BUNDLED:signs", tag: "signs", weight: 0.2 }

        Backwards compat: ``text.corpus_path`` (single string) → loaded as
        a single "general" corpus with weight 1.0.

        If neither is set, falls back to all four bundled corpora with
        equal weights.
        """
        out: list[tuple[str, list[str], float]] = []

        explicit_paths = self.config.get("corpus_paths")
        single_path = self.config.get("corpus_path")

        if explicit_paths:
            for entry in explicit_paths:
                path_str = str(entry.get("path"))
                tag = str(entry.get("tag", "user"))
                weight = float(entry.get("weight", 1.0))
                sentences = self._resolve_and_load(path_str)
                if sentences:
                    out.append((tag, sentences, weight))
        elif single_path:
            sentences = self._resolve_and_load(str(single_path))
            if sentences:
                out.append(("general", sentences, 1.0))
        else:
            # Default: equal-weight bundled corpora
            for tag in BUNDLED_CORPORA:
                sentences = self._load_plain_text(get_bundled_corpus_path(tag))
                if sentences:
                    out.append((tag, sentences, 1.0))

        for tag, sentences, weight in out:
            logger.info(
                "Loaded corpus %s: %d entries (weight %.2f)", tag, len(sentences), weight
            )
        return out

    def _resolve_and_load(self, path_str: str) -> list[str]:
        """Load a corpus from a path or a ``BUNDLED:<tag>`` reference."""
        if path_str.startswith("BUNDLED:"):
            tag = path_str.split(":", 1)[1]
            return self._load_plain_text(get_bundled_corpus_path(tag))
        path = Path(path_str).expanduser()
        if not path.exists():
            logger.warning("Corpus path does not exist: %s — skipping.", path)
            return []
        # Try HuggingFace datasets format first (a directory)
        if path.is_dir():
            try:
                from datasets import load_from_disk

                ds = load_from_disk(str(path))
                split = ds["train"] if "train" in ds else ds[next(iter(ds))]
                rows = split.select(
                    range(min(len(split), MAX_LOADED_SENTENCES_PER_CORPUS))
                )["text"]
                return [
                    s.strip()
                    for s in rows
                    if MIN_SENTENCE_LEN <= len(s.strip()) <= MAX_SENTENCE_LEN
                ]
            except Exception as exc:  # noqa: BLE001
                logger.debug("datasets load failed for %s (%s) — trying as text.", path, exc)
        return self._load_plain_text(path)

    def _load_plain_text(self, path: Path) -> list[str]:
        if not path.exists():
            return []
        try:
            sentences: list[str] = []
            with open(path, encoding="utf-8") as f:
                for line in f:
                    stripped = line.strip()
                    if MIN_SENTENCE_LEN <= len(stripped) <= MAX_SENTENCE_LEN:
                        sentences.append(stripped)
                    if len(sentences) >= MAX_LOADED_SENTENCES_PER_CORPUS:
                        break
            return sentences
        except OSError as exc:
            logger.warning("Could not load %s: %s", path, exc)
            return []

    # ── Internal — sampling ─────────────────────────────────────────────────

    def _pick_corpus(self) -> tuple[str, list[str]]:
        tags_and_lists = [(tag, sentences) for tag, sentences, _ in self.corpora]
        weights = [w for _, _, w in self.corpora]
        idx = random.choices(range(len(self.corpora)), weights=weights, k=1)[0]
        return tags_and_lists[idx]

    def _sample_from_corpus(self) -> str:
        _tag, sentences = self._pick_corpus()
        sentence = random.choice(sentences)
        upper_bound = min(self.max_length, len(sentence))
        if upper_bound < self.min_length:
            return sentence
        target_len = random.randint(self.min_length, upper_bound)
        if len(sentence) <= target_len:
            return sentence
        start = random.randint(0, len(sentence) - target_len)
        return sentence[start : start + target_len]

    def _sample_word_from_corpus(self) -> str:
        """Sample a single space-separated word from a corpus entry.

        Used by sample_paragraph so paragraph words come from the
        corpus vocabulary instead of random characters.
        """
        _tag, sentences = self._pick_corpus()
        sentence = random.choice(sentences)
        words = sentence.split()
        if not words:
            return self._generate_random_word(random.randint(2, 8))
        return random.choice(words)

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
        return "".join(c for c in text if c in self.charset_set)
