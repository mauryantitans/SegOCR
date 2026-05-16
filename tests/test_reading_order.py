"""Tests for segocr.postprocessing.reading_order."""
from __future__ import annotations

from segocr.postprocessing.instance_extraction import CharacterInstance
from segocr.postprocessing.reading_order import recover_text
from segocr.utils.charset import char_to_class_id


def _inst(char: str, x: int, y: int, w: int = 20, h: int = 20) -> CharacterInstance:
    """Helper: build a CharacterInstance positioned at (x, y) for ``char``."""
    cls = char_to_class_id(tier=1)[char]
    return CharacterInstance(
        class_id=cls,
        bbox=(x, y, w, h),
        centroid=(float(x + w / 2), float(y + h / 2)),
        area=w * h,
    )


def test_empty_instances_returns_empty_string():
    assert recover_text([]) == ""


def test_single_line_left_to_right():
    instances = [
        _inst("H", 10, 10),
        _inst("i", 35, 10),
    ]
    assert recover_text(instances) == "Hi"


def test_x_sort_independent_of_input_order():
    instances = [
        _inst("i", 35, 10),
        _inst("H", 10, 10),
    ]
    assert recover_text(instances) == "Hi"


def test_two_lines_ordered_top_to_bottom():
    line1 = [_inst("A", 10, 10), _inst("B", 35, 10)]
    line2 = [_inst("C", 10, 60), _inst("D", 35, 60)]
    text = recover_text(line1 + line2)
    assert text == "AB\nCD"


def test_word_break_on_large_gap():
    # "Hi yo" — large gap between 'i' and 'y'
    instances = [
        _inst("H", 0, 10),
        _inst("i", 22, 10),
        _inst("y", 200, 10),   # huge gap → word break
        _inst("o", 222, 10),
    ]
    out = recover_text(instances)
    assert out == "Hi yo"


def test_close_chars_no_word_break():
    # Tight spacing — no word break expected
    instances = [
        _inst("a", 0, 10),
        _inst("b", 22, 10),
        _inst("c", 44, 10),
    ]
    assert recover_text(instances) == "abc"
