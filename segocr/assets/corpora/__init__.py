"""Bundled mini-corpora for context-aware text sampling.

These ship with the package so the generator produces semantically
plausible text out of the box, before any large external corpora
(Wikitext, Project Gutenberg) are downloaded. Tag conventions:

    signs     — common signage, business types, action phrases
    receipts  — POS / receipt vocabulary
    names     — common first names, last names
    numbers   — years, prices, codes, room numbers, pages
"""
from __future__ import annotations

from importlib.resources import files
from pathlib import Path


def get_bundled_corpus_path(tag: str) -> Path:
    """Return the path to a bundled corpus file by tag."""
    base = files("segocr.assets.corpora")
    return Path(str(base / f"{tag}.txt"))


BUNDLED_CORPORA = ("signs", "receipts", "names", "numbers")
