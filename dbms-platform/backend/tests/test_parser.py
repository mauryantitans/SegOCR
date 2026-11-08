from app.content.parser import parse_document

DOC = '''---
id: demo
dataset: shop
concepts: [x]
---
## Introduction
Text.

:::warning[Careful]
```sql
SELECT 'inside callout stays markdown';
```
:::

```sql run
SELECT 1;
```

```exercise
id: ex1
title: T
prompt: P
hints: [h]
solution: SELECT 1
```

```quiz
- id: q1
  question: Q?
  options: [a, b]
  answer: 1
  explanation: because
```
'''


def test_segments_and_ids():
    d = parse_document(DOC, "demo")
    assert not d.errors, d.errors
    kinds = [s["type"] for s in d.segments]
    assert kinds == ["heading", "markdown", "sql", "exercise", "quiz"]
    assert "inside callout" in d.segments[1]["content"]
    sql = d.segments[2]
    assert sql["run"] and sql["key"].startswith("s1:")
    assert d.segments[3]["exercise"]["id"] == "demo/ex1" and d.segments[3]["exercise"]["dataset"] == "shop"
    assert d.segments[4]["questions"][0]["id"] == "demo/q1"


def test_errors_reported():
    bad = DOC.replace("answer: 1", "answer: 7").replace("hints: [h]\n", "")
    errs = parse_document(bad, "demo").errors
    assert any("out of range" in e for e in errs) and any("hint" in e for e in errs)


def test_real_content_parses():
    from pathlib import Path
    from app.content.repository import Repository
    repo = Repository.load(Path(__file__).resolve().parents[2] / "content")
    authored = [r for r in repo.lessons.values() if r.doc]
    assert authored and all(not r.load_error and not r.doc.errors for r in authored)
    for ex in repo.exercises.values():
        assert "solution" in ex
