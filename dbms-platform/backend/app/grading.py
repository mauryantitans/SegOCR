"""Server-side exercise grading by result comparison on freshly loaded data.

The learner's SQL and the reference solution run on identical copies of the
exercise dataset (each in its own sandbox lab). For ``query`` exercises the
last result set of each is compared; for ``dml`` exercises the ``check``
query is run after each and those results are compared.
"""
from __future__ import annotations

import threading
from collections import Counter
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation

from .sandbox.manager import LabManager, SandboxError

GRADER_OWNER = "__grader__"


@dataclass
class GradeResult:
    passed: bool
    summary: str
    details: list[str] = field(default_factory=list)
    your_result: dict | None = None
    missing_rows: list[list] = field(default_factory=list)
    unexpected_rows: list[list] = field(default_factory=list)
    expected_shape: dict | None = None
    error: dict | None = None

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def normalize(v):
    """Normalize a cell so that semantically equal values compare equal (55 == '55.0' == 55.000)."""
    if v is None:
        return None
    if isinstance(v, bool):
        return Decimal(int(v))
    if isinstance(v, (int, float, Decimal)) or (isinstance(v, str) and v and v.strip().lstrip("+-").replace(".", "", 1).isdigit()):
        try:
            d = Decimal(str(v).strip())
            return d.quantize(Decimal("0.000001")).normalize() if d == d else str(v)
        except InvalidOperation:
            return str(v)
    return str(v)


def _rows(result: dict) -> list[tuple]:
    return [tuple(normalize(c) for c in r) for r in result["rows"]]


def compare(expected: dict, actual: dict, order_matters: bool, reveal_missing: bool) -> GradeResult:
    ecols, acols = expected["columns"], actual["columns"]
    shape = {"columns": len(ecols), "rows": expected["row_count"]}
    if len(ecols) != len(acols):
        return GradeResult(False, f"Your result has {len(acols)} column(s); the expected result has {len(ecols)}.",
                           ["Re-read the prompt for the exact columns and their order."], expected_shape=shape)
    if actual.get("truncated") or expected.get("truncated"):
        if actual["row_count"] != expected["row_count"]:
            return GradeResult(False, f"Your query returned {actual['row_count']} rows; expected {expected['row_count']}.",
                               expected_shape=shape)
    erows, arows = _rows(expected), _rows(actual)
    ec, ac = Counter(erows), Counter(arows)
    details = []
    names_differ = [a for a, e in zip(acols, ecols) if a.lower() != e.lower()]
    if ec == ac:
        if order_matters and erows != arows:
            return GradeResult(False, "You return the right rows, but not in the required order.",
                               ["Check the ORDER BY clause, including the direction and any tie-breaker."], expected_shape=shape)
        if names_differ:
            details.append(f"Note: your column names ({', '.join(acols)}) differ from the expected names "
                           f"({', '.join(ecols)}). Values are graded, not names, but aliases matter in real reports.")
        return GradeResult(True, "Correct — your result matches the expected result exactly.", details, expected_shape=shape)
    missing = list((ec - ac).elements())
    unexpected = list((ac - ec).elements())
    summary = (f"Your query returned {len(arows)} row(s); the expected result has {len(erows)}. "
               f"{len(missing)} expected row(s) are missing and {len(unexpected)} row(s) should not be there.")
    if len(arows) == len(erows) and missing:
        details.append("The row count matches, so the filtering is probably right — compare the values column by column "
                       "(rounding, NULL handling, aggregation level or joins).")
    if len(arows) > len(erows):
        details.append("You return too many rows: look for a missing filter, a missing DISTINCT/GROUP BY, or a join that "
                       "multiplies rows.")
    if len(arows) < len(erows):
        details.append("You return too few rows: look for an over-restrictive filter, an INNER JOIN where an outer join is "
                       "needed, or NULLs being excluded by a comparison.")
    to_list = lambda rows: [[str(c) if c is not None else None for c in r] for r in rows[:5]]  # noqa: E731
    return GradeResult(False, summary, details, missing_rows=to_list(missing) if reveal_missing else [],
                       unexpected_rows=to_list(unexpected), expected_shape=shape)


class Grader:
    def __init__(self, labs: LabManager):
        self.labs = labs
        self._expected: dict[str, dict] = {}
        self._lock = threading.Lock()

    def _result_of(self, owner: str, lab: str, ex: dict, sql: str) -> tuple[dict | None, dict | None]:
        """Reset the lab, run sql (and the check query for dml). Returns (result, error)."""
        self.labs.reset_lab(owner, lab, ex.get("dataset", "empty"), setup_sql=ex.get("setup"))
        r = self.labs.run(owner, lab, sql)
        for st in r.statements:
            if st.status != "ok":
                return None, {"index": st.index, "sql": st.sql, "status": st.status, **(st.error or {})}
        if ex.get("kind", "query") == "dml":
            rc = self.labs.run(owner, lab, ex["check"])
            last = rc.statements[-1]
            if last.status != "ok":
                return None, {"status": "check-failed", **(last.error or {})}
            return {"columns": last.columns, "rows": last.rows, "row_count": last.row_count, "truncated": last.truncated}, None
        sets = [st for st in r.statements if st.columns]
        if not sets:
            return None, {"status": "no-result", "message": "Your SQL ran, but no statement returned a result set. "
                                                             "This exercise expects a SELECT."}
        last = sets[-1]
        return {"columns": last.columns, "rows": last.rows, "row_count": last.row_count, "truncated": last.truncated}, None

    def expected(self, ex: dict) -> dict:
        with self._lock:
            cached = self._expected.get(ex["id"])
        if cached:
            return cached
        res, err = self._result_of(GRADER_OWNER, f"exercise:{ex['id']}", ex, ex["solution"])
        if err:
            raise SandboxError(f"Reference solution for {ex['id']} failed: {err.get('message')}")
        with self._lock:
            self._expected[ex["id"]] = res
        return res

    def grade(self, owner: str, ex: dict, sql: str, previous_failures: int) -> GradeResult:
        expected = self.expected(ex)
        actual, err = self._result_of(owner, f"exercise:{ex['id']}", ex, sql)
        if err:
            msg = err.get("message") or "Your SQL failed."
            where = f" (statement {err['index'] + 1})" if "index" in err else ""
            return GradeResult(False, f"Your SQL did not run successfully{where}: {msg}",
                               [err["hint"]] if err.get("hint") else [], error=err)
        g = compare(expected, actual, bool(ex.get("orderMatters")), reveal_missing=previous_failures >= 1)
        g.your_result = {"columns": actual["columns"], "rows": actual["rows"][:20], "row_count": actual["row_count"]}
        return g
