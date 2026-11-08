from app.grading import compare, normalize


def res(cols, rows):
    return {"columns": cols, "rows": rows, "row_count": len(rows)}


def test_normalize_numbers():
    assert normalize("55.0") == normalize(55) == normalize("55.000000")
    assert normalize(None) is None and normalize("abc") == "abc"


def test_match_ignores_order_and_names():
    g = compare(res(["a", "b"], [[1, "x"], [2, "y"]]), res(["A", "renamed"], [[2, "y"], [1, "x"]]), False, False)
    assert g.passed and g.details  # name note


def test_order_matters():
    g = compare(res(["a"], [[1], [2]]), res(["a"], [[2], [1]]), True, False)
    assert not g.passed and "order" in g.summary


def test_column_count_mismatch():
    assert not compare(res(["a"], [[1]]), res(["a", "b"], [[1, 2]]), False, False).passed


def test_multiset_semantics_and_reveal():
    exp, act = res(["a"], [[1], [1], [2]]), res(["a"], [[1], [2], [3]])
    hidden = compare(exp, act, False, reveal_missing=False)
    shown = compare(exp, act, False, reveal_missing=True)
    assert not hidden.passed and hidden.missing_rows == [] and shown.missing_rows == [["1"]]
    assert shown.unexpected_rows == [["3"]]
