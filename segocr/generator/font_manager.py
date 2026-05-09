"""Font loading, validation, caching, and weighted sampling.

Implementation Guide §3.1.

Walks the font root directory for .ttf / .otf files, validates each by
attempting to render every character in the active charset at a reference
size, discards fonts that fail, caches per-character bounding boxes to
JSON, and exposes weighted sampling by category.

Categorization is best-effort: we look at the parent-directory name. The
Google Fonts repo organizes fonts under license buckets (``ofl/``,
``apache/``, ``ufl/``) rather than style buckets, so most fonts will land
in the ``sans-serif`` default. Override by subclassing ``_categorize`` if
you have METADATA.pb metadata you want to parse.
"""
from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import ImageFont

from segocr.utils.charset import CHARSET_TIER1

if TYPE_CHECKING:
    from PIL.ImageFont import FreeTypeFont

logger = logging.getLogger(__name__)

REFERENCE_SIZE = 48
FONT_EXTENSIONS = (".ttf", ".otf")


class FontManager:
    """Manages the font library for text rendering."""

    def __init__(
        self,
        config: dict,
        charset: tuple[str, ...] = CHARSET_TIER1,
    ) -> None:
        self.config = config
        self.root_dir = Path(config["root_dir"]).expanduser().resolve()
        self.cache_path = Path(config["cache_path"]).expanduser().resolve()
        self.min_size = int(config["min_size"])
        self.max_size = int(config["max_size"])
        self.category_weights: dict[str, float] = dict(config["categories"])
        self.charset = charset

        # category → list[Path]
        self.fonts_by_category: dict[str, list[Path]] = {
            cat: [] for cat in self.category_weights
        }
        # (font_path_str, char) → bbox tuple at REFERENCE_SIZE
        self.bbox_cache: dict[tuple[str, str], tuple[int, int, int, int]] = {}

        self._load_or_build_cache()

    # ── Public API ──────────────────────────────────────────────────────────

    @property
    def num_fonts(self) -> int:
        return sum(len(fonts) for fonts in self.fonts_by_category.values())

    def sample_font(self, size: int | None = None) -> tuple[FreeTypeFont, str]:
        """Sample one font, weighted by configured category distribution.

        Categories with no validated fonts are dropped (their probability
        weight is *not* redistributed — picking from an empty bucket would
        be ill-defined; the user can rebalance by editing the config).

        Args:
            size: pixel size; if ``None``, sampled uniformly from
                  [min_size, max_size].

        Returns:
            (FreeTypeFont, category_name)
        """
        non_empty = {
            cat: weight
            for cat, weight in self.category_weights.items()
            if self.fonts_by_category.get(cat)
        }
        if not non_empty:
            raise RuntimeError(
                f"No validated fonts available under {self.root_dir}. "
                "Did you run scripts/setup_data.ps1, or pass a valid root_dir?"
            )

        category = random.choices(
            list(non_empty.keys()), weights=list(non_empty.values()), k=1
        )[0]
        font_path = random.choice(self.fonts_by_category[category])

        if size is None:
            size = random.randint(self.min_size, self.max_size)
        return ImageFont.truetype(str(font_path), size), category

    def get_char_bbox(
        self, font_path: Path | str, char: str, size: int
    ) -> tuple[int, int, int, int]:
        """Get the cached bbox for ``char`` rendered at ``size`` px.

        Cached values are at ``REFERENCE_SIZE``; we scale linearly. This
        is exact under proportional rasterization, which Pillow uses.
        """
        key = (str(font_path), char)
        if key not in self.bbox_cache:
            font = ImageFont.truetype(str(font_path), REFERENCE_SIZE)
            self.bbox_cache[key] = tuple(font.getbbox(char))  # type: ignore[assignment]
        bbox = self.bbox_cache[key]
        scale = size / REFERENCE_SIZE
        return tuple(int(round(c * scale)) for c in bbox)  # type: ignore[return-value]

    def get_all_fonts(
        self, size: int = REFERENCE_SIZE
    ) -> list[tuple[FreeTypeFont, Path]]:
        """Load every validated font at ``size`` px. Skips loads that fail."""
        out: list[tuple[FreeTypeFont, Path]] = []
        for fonts in self.fonts_by_category.values():
            for path in fonts:
                try:
                    out.append((ImageFont.truetype(str(path), size), path))
                except OSError:
                    continue
        return out

    # ── Internal — cache + scan + validate ──────────────────────────────────

    def _load_or_build_cache(self) -> None:
        if self.cache_path.exists():
            try:
                with open(self.cache_path, encoding="utf-8") as f:
                    cached = json.load(f)
                if (
                    cached.get("root_dir") == str(self.root_dir)
                    and cached.get("charset") == list(self.charset)
                ):
                    self._restore_from_cache(cached)
                    if self.num_fonts > 0:
                        return
            except (json.JSONDecodeError, OSError, KeyError) as exc:
                logger.warning("Font cache invalid (%s); rebuilding.", exc)

        self._scan_and_validate()
        self._write_cache()

    def _restore_from_cache(self, cached: dict) -> None:
        self.fonts_by_category = {cat: [] for cat in self.category_weights}
        for cat, paths in cached.get("fonts_by_category", {}).items():
            self.fonts_by_category.setdefault(cat, []).extend(Path(p) for p in paths)
        self.bbox_cache = {}
        for key_str, bbox in cached.get("bbox_cache", {}).items():
            font_path, char = key_str.split("|", 1)
            self.bbox_cache[(font_path, char)] = tuple(bbox)  # type: ignore[assignment]

    def _scan_and_validate(self) -> None:
        if not self.root_dir.exists():
            logger.warning(
                "Font root_dir does not exist: %s — FontManager will be empty.",
                self.root_dir,
            )
            return

        font_paths: list[Path] = []
        for ext in FONT_EXTENSIONS:
            font_paths.extend(self.root_dir.rglob(f"*{ext}"))
        font_paths = sorted(set(font_paths))
        logger.info("Scanning %d fonts under %s ...", len(font_paths), self.root_dir)

        accepted = 0
        for path in font_paths:
            bboxes = self._validate_font(path)
            if bboxes is None:
                continue
            cat = self._categorize(path)
            self.fonts_by_category.setdefault(cat, []).append(path)
            for char, bbox in bboxes.items():
                self.bbox_cache[(str(path), char)] = bbox
            accepted += 1

        logger.info(
            "Font validation complete: %d / %d accepted.", accepted, len(font_paths)
        )

    def _validate_font(
        self, font_path: Path
    ) -> dict[str, tuple[int, int, int, int]] | None:
        """Try to load and render the full character set with this font.

        Returns a per-character bbox dict on success, ``None`` on any failure
        (load error, missing glyph, zero-area glyph). The contract is
        all-or-nothing: a font that handles 61/62 chars is rejected so the
        rest of the pipeline can rely on full coverage.
        """
        try:
            font = ImageFont.truetype(str(font_path), REFERENCE_SIZE)
        except (OSError, ValueError):
            return None

        bboxes: dict[str, tuple[int, int, int, int]] = {}
        for char in self.charset:
            try:
                bbox = font.getbbox(char)
            except Exception:  # noqa: BLE001 — Pillow can raise opaque errors
                return None
            if bbox is None:
                return None
            x0, y0, x1, y1 = bbox
            if x1 - x0 <= 0 or y1 - y0 <= 0:
                return None
            bboxes[char] = (int(x0), int(y0), int(x1), int(y1))
        return bboxes

    def _categorize(self, font_path: Path) -> str:
        """Best-effort categorization from the directory hierarchy.

        Looks for any configured category name in the path components;
        falls back to ``sans-serif`` if nothing matches.
        """
        try:
            rel = font_path.relative_to(self.root_dir)
        except ValueError:
            return "sans-serif"
        parts_lower = [p.lower() for p in rel.parts]
        for part in parts_lower:
            for cat in self.category_weights:
                if cat.lower() in part:
                    return cat
        return "sans-serif"

    def _write_cache(self) -> None:
        if self.num_fonts == 0:
            return  # nothing useful to cache
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_data = {
            "root_dir": str(self.root_dir),
            "charset": list(self.charset),
            "fonts_by_category": {
                cat: [str(p) for p in paths]
                for cat, paths in self.fonts_by_category.items()
                if paths
            },
            "bbox_cache": {
                f"{path}|{char}": list(bbox)
                for (path, char), bbox in self.bbox_cache.items()
            },
        }
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(cache_data, f)
