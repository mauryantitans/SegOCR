#!/usr/bin/env python3
"""Validate course content and capture real MySQL output for every runnable block.

    python scripts/capture_content.py --doc m03-aggregate-functions
    python scripts/capture_content.py --module m03
    python scripts/capture_content.py --all
    python scripts/capture_content.py --all --lint-only      # no MySQL needed

For each lesson/project it:
  1. lints structure (front matter, the 16 canonical sections, exercises, quizzes);
  2. loads the lesson's dataset into a fresh sandbox lab and executes every
     ```sql run``` block and walkthrough step in document order, failing if a
     block errors unexpectedly (or succeeds when marked ``error``);
  3. replays every ```concurrency``` scenario with real concurrent sessions
     and checks each step's expectation (ok / blocks / error);
  4. runs every exercise's reference solution (and ``check`` query);
  5. writes content/generated/captures/<doc>.json with the real results.

Outputs shown to learners therefore always come from an actual MySQL server,
never from hand-written text.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.content.parser import LESSON_SECTIONS, ContentError, parse_document  # noqa: E402

OWNER = "capture"
MIN_EXERCISES = 4
MIN_QUESTIONS = 6
DISPLAY_ROWS = 50


def trim_statement(st: dict, measured: bool) -> dict:
    out = {k: st.get(k) for k in ("sql", "status", "columns", "row_count", "affected_rows", "info", "truncated")}
    out["rows"] = (st.get("rows") or [])[:DISPLAY_ROWS]
    if st.get("row_count") and st["row_count"] > DISPLAY_ROWS:
        out["truncated"] = True
    if st.get("warnings"):
        out["warnings"] = st["warnings"]
    if st.get("error"):
        out["error"] = {"code": st["error"].get("code"), "message": st["error"].get("message")}
    if measured:
        out["elapsed_ms"] = st.get("elapsed_ms")
    return {k: v for k, v in out.items() if not (v is None or v is False or (isinstance(v, list) and not v)) or k in ("rows", "columns")}


# ---------------------------------------------------------------------------
# Lint
# ---------------------------------------------------------------------------
def lint(doc, kind: str, datasets: set[str], all_lesson_ids: set[str]) -> list[str]:
    errs = list(doc.errors)
    meta = doc.meta
    if meta.get("dataset", "empty") not in datasets:
        errs.append(f"unknown dataset '{meta.get('dataset')}'")
    if not meta.get("concepts"):
        errs.append("front matter needs a non-empty 'concepts' list")
    if kind == "lesson":
        titles = [s["title"] for s in doc.sections]
        required = [t for t, _ in LESSON_SECTIONS]
        missing = [t for t in required if t not in titles]
        if missing:
            errs.append(f"missing required sections: {missing}")
        else:
            positions = [titles.index(t) for t in required]
            if positions != sorted(positions):
                errs.append("required sections are out of order")
        n_ex = sum(1 for s in doc.segments if s["type"] == "exercise")
        n_q = sum(len(s["questions"]) for s in doc.segments if s["type"] == "quiz")
        n_run = sum(1 for s in doc.segments if s["type"] == "sql" and s["run"])
        if n_ex < MIN_EXERCISES:
            errs.append(f"only {n_ex} exercises (minimum {MIN_EXERCISES})")
        if n_q < MIN_QUESTIONS:
            errs.append(f"only {n_q} quiz questions (minimum {MIN_QUESTIONS})")
        if n_run < 1 and not meta.get("noRunnableExamples"):
            errs.append("no ```sql run``` blocks (set noRunnableExamples: true with a reason if truly not applicable)")
    for seg in doc.segments:
        if seg["type"] == "exercise":
            ds = seg["exercise"].get("dataset")
            if ds and ds not in datasets:
                errs.append(f"exercise {seg['exercise']['id']} uses unknown dataset {ds}")
        if seg["type"] == "concurrency":
            for k, st in enumerate(seg["steps"]):
                if st["expect"] == "blocks":
                    from app.sandbox.splitter import split_sql
                    if len(split_sql(st["sql"])) != 1:
                        errs.append(f"concurrency {seg['id']} step {k}: a step expected to block must be a single statement")
    # Every structured fence must have produced a segment; a nested ``` inside YAML silently swallows blocks.
    raw = doc.meta.get("_raw_body", "")
    for kind in ("exercise", "quiz", "walkthrough", "concurrency", "viz"):
        fences = len(re.findall(rf"^`{{3,}}{kind}\s*$", raw, re.M))
        parsed = sum(1 for sg in doc.segments if sg["type"] == kind)
        if fences != parsed:
            errs.append(f"{fences} ```{kind} fences but {parsed} parsed — a ``` line inside a YAML block probably closed "
                        "it early (use ~~~ inside, or a four-backtick outer fence)")
    for ref in meta.get("related", []) or []:
        if ref not in all_lesson_ids:
            errs.append(f"related lesson '{ref}' does not exist")
    return errs


def word_count(doc) -> int:
    n = 0
    for s in doc.segments:
        if s["type"] == "markdown":
            n += len(s["content"].split())
        elif s["type"] == "exercise":
            n += len(str(s["exercise"].get("prompt", "")).split()) + len(str(s["exercise"].get("explanation", "")).split())
        elif s["type"] == "quiz":
            n += sum(len(str(q.get("question", "")).split()) + len(str(q.get("explanation", "")).split()) for q in s["questions"])
        elif s["type"] == "walkthrough":
            n += sum(len(str(st.get("explain", "")).split()) for st in s["steps"])
    return n


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------
def capture_doc(m, doc_id: str, doc) -> tuple[dict, list[str]]:
    errs: list[str] = []
    blocks: dict[str, dict] = {}
    dataset = doc.meta.get("dataset", "empty")
    lab = f"lesson:{doc_id}"
    m.reset_lab(OWNER, lab, dataset)

    def run_block(sql: str, expect: str, measured: bool, label: str) -> dict:
        r = m.run(OWNER, lab, sql, stop_on_error=(expect != "error"))
        sts = [s.to_dict() for s in r.statements]
        failed = [s for s in sts if s["status"] in ("error", "rejected", "blocked")]
        if expect == "ok" and failed:
            f = failed[0]
            errs.append(f"{label}: unexpected {f['status']}: {(f.get('error') or {}).get('message')} — in: {f['sql'][:120]}")
        if expect == "error" and not failed:
            errs.append(f"{label}: marked 'error' but every statement succeeded")
        if any(s["status"] == "ok" and s["kind"] in ("ddl",) and s["verb"] == "DROP" for s in sts):
            pass
        return {"statements": [trim_statement(s, measured) for s in sts], "measured": measured or None}

    for seg in doc.segments:
        t = seg["type"]
        if t == "sql" and seg["run"]:
            cap = run_block(seg["sql"], seg["expect"], seg["measured"], f"sql block {seg['id']}")
            if seg["volatile"]:
                cap = {"volatile": True, "statements": [{"sql": s["sql"], "status": s["status"]} for s in cap["statements"]]}
            blocks[seg["key"]] = cap
        elif t == "walkthrough":
            for k, st in enumerate(seg["steps"]):
                if st.get("sql"):
                    blocks[st["key"]] = run_block(st["sql"], st.get("expect", "ok"), False, f"walkthrough {seg['id']} step {k}")
        elif t == "concurrency":
            cap, cerrs = capture_concurrency(m, doc_id, seg, dataset)
            blocks[seg["key"]] = cap
            errs.extend(cerrs)
        elif t == "exercise":
            try:
                errs.extend(check_exercise(m, seg["exercise"]))
            except Exception as e:  # noqa: BLE001 - e.g. invalid setup SQL
                errs.append(f"exercise {seg['exercise']['id']}: setup or solution crashed: {type(e).__name__}: {e}")
    return blocks, errs


def check_exercise(m, ex: dict) -> list[str]:
    if ex.get("kind") == "open":
        return []
    lab = f"exercise:{ex['id']}"
    try:
        return _check_exercise(m, ex, lab)
    finally:
        m.drop_lab(m.schema_for(OWNER, lab))


def _check_exercise(m, ex: dict, lab: str) -> list[str]:
    errs = []
    m.reset_lab(OWNER, lab, ex.get("dataset", "empty"), setup_sql=ex.get("setup"))
    r = m.run(OWNER, lab, ex["solution"])
    bad = [s for s in r.statements if s.status != "ok"]
    if bad:
        errs.append(f"exercise {ex['id']}: solution failed: {(bad[0].error or {}).get('message')}")
        return errs
    if ex.get("kind", "query") == "query":
        last = r.statements[-1]
        if not last.columns:
            errs.append(f"exercise {ex['id']}: query exercise solution must end with a statement returning rows")
        elif last.row_count == 0 and not ex.get("allowEmpty"):
            errs.append(f"exercise {ex['id']}: solution returns 0 rows on the dataset — a wrong answer returning nothing would pass "
                        "(set allowEmpty: true if intended)")
    else:
        rc = m.run(OWNER, lab, ex["check"])
        if any(s.status != "ok" for s in rc.statements) or not rc.statements[-1].columns:
            errs.append(f"exercise {ex['id']}: check query failed or returned no result set")
    return errs


def capture_concurrency(m, doc_id: str, seg: dict, default_ds: str) -> tuple[dict, list[str]]:
    errs = []
    lab = f"conc:{doc_id}:{seg['id']}"
    m.reset_lab(OWNER, lab, seg.get("dataset") or default_ds, setup_sql=seg.get("setup") or None)
    steps_out: list[dict] = []
    pending: dict[str, int] = {}
    label = f"concurrency {seg['id']}"
    pos = {}  # step index -> position in steps_out

    def collect_pending(after_step: int):
        for s2, idx in list(pending.items()):
            done = m.poll(OWNER, lab, s2)
            if done:
                final = done[-1].to_dict()
                out = steps_out[pos[idx]]
                out["statements"][-1] = trim_statement(final, False)
                out["completedAfterStep"] = after_step
                out["finalStatus"] = "error" if final["status"] == "error" else "ok"
                del pending[s2]

    try:
        for k, st in enumerate(seg["steps"]):
            if st.get("observe"):
                time.sleep(0.4)
                collect_pending(k - 1)  # a blocked statement may have finished (or failed) since the last step
                steps_out.append({"step": k, "observe": "locks", "status": "ok", "snapshot": m.inspect_locks(OWNER, lab)})
                continue
            sess = st["session"]
            if sess in pending:
                errs.append(f"{label} step {k}: session {sess} is still blocked from step {pending[sess]}")
                break
            r = m.run(OWNER, lab, st["sql"], session_name=sess, wait_s=1.5, timeout_ms=60_000)
            sts = [s.to_dict() for s in r.statements]
            if sts and sts[-1]["status"] == "blocked":
                status = "blocked"
                pending[sess] = k
            elif any(s["status"] in ("error", "rejected") for s in sts):
                status = "error"
            else:
                status = "ok"
            pos[k] = len(steps_out)
            steps_out.append({"step": k, "session": sess, "status": status,
                              "statements": [trim_statement(s, False) for s in sts]})
            time.sleep(0.4)
            collect_pending(k)
        # expectations
        for out in steps_out:
            st = seg["steps"][out["step"]]
            if st.get("observe"):
                continue
            exp = st["expect"]
            final = out.get("finalStatus", out["status"])
            if exp == "blocks" and out["status"] != "blocked":
                errs.append(f"{label} step {out['step']}: expected to block but finished with {out['status']}")
            elif exp == "ok" and (out["status"] != "ok"):
                msg = (out["statements"][-1].get("error") or {}).get("message") if out["statements"] else ""
                errs.append(f"{label} step {out['step']}: expected ok, got {out['status']} {msg or ''}")
            elif exp == "error" and final != "error":
                errs.append(f"{label} step {out['step']}: expected an error, got {final}")
            code = st.get("errorCode")
            if code and final == "error":
                got = (out["statements"][-1].get("error") or {}).get("code")
                if got != code:
                    errs.append(f"{label} step {out['step']}: expected error {code}, got {got}")
            if exp == "blocks" and "completedAfterStep" not in out:
                errs.append(f"{label} step {out['step']}: blocked step never completed — release the lock in a later step")
    finally:
        m.drop_lab(m.schema_for(OWNER, lab))  # closes the sessions and kills anything still waiting
    return {"steps": steps_out}, errs


def validate_assessment(m, path: Path, datasets: set[str]) -> list[str]:
    from app.content.parser import _check_question
    errs: list[str] = []
    try:
        a = yaml.safe_load(path.read_text())
    except yaml.YAMLError as e:
        return [f"invalid YAML: {e}"]
    for key in ("id", "module", "title", "questions"):
        if not a.get(key):
            errs.append(f"missing '{key}'")
    if errs:
        return errs
    if a["id"] != f"{a['module']}-assessment" or path.stem != a["id"]:
        errs.append("id must be '<module>-assessment' and match the file name")
    qs = a.get("questions", [])
    if len(qs) < 15:
        errs.append(f"only {len(qs)} questions (minimum 15)")
    ids = set()
    for q in qs:
        q = dict(q)
        q["id"] = f"{a['id']}/{q.get('id')}"
        if q["id"] in ids:
            errs.append(f"duplicate question id {q['id']}")
        ids.add(q["id"])
        _check_question(q, errs, "assessment")
    ch = a.get("challenges", [])
    if len(ch) < 3:
        errs.append(f"only {len(ch)} challenges (minimum 3)")
    for ex in ch:
        ex = dict(ex)
        ex["id"] = f"{a['id']}/{ex.get('id')}"
        ex.setdefault("kind", "query")
        for req in ("title", "prompt", "solution", "hints"):
            if not ex.get(req):
                errs.append(f"challenge {ex['id']} missing '{req}'")
        if ex.get("dataset", "shop") not in datasets:
            errs.append(f"challenge {ex['id']} unknown dataset")
        ex.setdefault("dataset", "shop")
        if m is not None and ex.get("solution"):
            errs.extend(check_exercise(m, ex))
    iv = a.get("interview", [])
    if len(iv) < 5:
        errs.append(f"only {len(iv)} interview questions (minimum 5)")
    for i in iv:
        if not i.get("question") or not i.get("answer"):
            errs.append(f"interview {i.get('id')} needs question and answer")
    return errs


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc", action="append", default=[], help="lesson or project id (repeatable)")
    ap.add_argument("--module", action="append", default=[], help="module id, e.g. m03 (repeatable)")
    ap.add_argument("--projects", action="store_true", help="all authored projects")
    ap.add_argument("--assessment", action="append", default=[], help="module id whose assessment to validate")
    ap.add_argument("--all", action="store_true", help="every authored lesson and project")
    ap.add_argument("--lint-only", action="store_true")
    ap.add_argument("--show", action="store_true", help="print captured blocks (debugging)")
    args = ap.parse_args()

    settings = get_settings()
    content = settings.content_dir
    cur = yaml.safe_load((content / "curriculum.yaml").read_text())
    datasets = {p.stem for p in (content / "datasets").glob("*.sql")}
    lessons = {}
    for mod in cur["modules"]:
        for ch in mod["chapters"]:
            for l in ch["lessons"]:
                lessons[l["id"]] = (mod["id"], content / "lessons" / mod["id"] / f"{l['id']}.md")
    projects = {p["id"]: content / "projects" / f"{p['id']}.md" for p in cur.get("projects", [])}

    targets: list[tuple[str, str, Path]] = []
    for lid, (mid, path) in lessons.items():
        if path.exists() and (args.all or lid in args.doc or mid in args.module):
            targets.append(("lesson", lid, path))
    for pid, path in projects.items():
        if path.exists() and (args.all or args.projects or pid in args.doc):
            targets.append(("project", pid, path))
    unknown = [d for d in args.doc if d not in lessons and d not in projects]
    if unknown:
        print(f"unknown or unauthored ids: {unknown}")
        return 2
    assessments = []
    for mid in set(args.module) | set(args.assessment) | ({m["id"] for m in cur["modules"]} if args.all else set()):
        path = content / "assessments" / f"{mid}-assessment.yaml"
        if path.exists():
            assessments.append(path)
        elif mid in args.assessment:
            print(f"no assessment file {path}")
            return 2
    if not targets and not assessments:
        print("nothing to do (no authored documents matched)")
        return 1

    manager = None
    if not args.lint_only:
        from app.sandbox.manager import LabManager
        manager = LabManager(settings, content / "datasets")
        version = manager.server_version()
        print(f"capturing against MySQL {version}")
    out_dir = content / "generated" / "captures"
    out_dir.mkdir(parents=True, exist_ok=True)

    failures = 0
    for kind, doc_id, path in targets:
        try:
            text = path.read_text()
            doc = parse_document(text, doc_id)
            doc.meta["_raw_body"] = text
        except (ContentError, yaml.YAMLError) as e:
            print(f"✗ {doc_id}: {e}")
            failures += 1
            continue
        errs = lint(doc, kind, datasets, set(lessons))
        if doc.meta.get("id") != doc_id:
            errs.append(f"front matter id {doc.meta.get('id')!r} != {doc_id!r}")
        blocks = {}
        if manager is not None:
            try:
                blocks, cerrs = capture_doc(manager, doc_id, doc)
                errs.extend(cerrs)
            except Exception as e:  # noqa: BLE001
                errs.append(f"capture crashed: {type(e).__name__}: {e}")
            finally:
                try:
                    manager.drop_lab(manager.schema_for(OWNER, f"lesson:{doc_id}"))
                except Exception:  # noqa: BLE001
                    pass
        if args.show:
            print(json.dumps(blocks, indent=1, ensure_ascii=False, default=str))
        wc = word_count(doc)
        if errs:
            failures += 1
            print(f"✗ {doc_id} ({wc} words)")
            for e in errs:
                print(f"    - {e}")
        else:
            print(f"✓ {doc_id} ({wc} words, {len(blocks)} captured blocks)")
        if manager is not None and not errs:
            payload = {"doc": doc_id, "meta": {"mysqlVersion": manager.server_version(),
                                               "capturedAt": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")},
                       "blocks": blocks}
            (out_dir / f"{doc_id}.json").write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n")
    for path in sorted(assessments):
        errs = validate_assessment(manager, path, datasets)
        if errs:
            failures += 1
            print(f"✗ {path.stem}")
            for e in errs:
                print(f"    - {e}")
        else:
            print(f"✓ {path.stem}")
    if manager is not None:
        manager.shutdown()
    total = len(targets) + len(assessments)
    print(f"\n{total - failures}/{total} documents passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
