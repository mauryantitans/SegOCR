"""Parse lesson/project Markdown into renderable *segments*.

A lesson file is Markdown with YAML front matter. Special fenced blocks become
structured segments (see docs/AUTHORING.md):

    ```sql                 display-only SQL (still openable in the playground)
    ```sql run             executed in order at capture time; real output stored
    ```sql run error       must fail; the real error is stored and shown
    ```sql run volatile    must succeed; output is not shown (non-deterministic)
    ```sql run measured    output shown with a "measured on build machine" notice
    ```exercise            YAML — graded hands-on exercise
    ```quiz                YAML — multiple-choice questions
    ```viz                 YAML — interactive visualization
    ```concurrency         YAML — multi-session transaction scenario
    ```walkthrough         YAML — step-by-step execution with real intermediate results

Structured blocks are only extracted at the top level (not inside ::: callouts).
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

import yaml

FENCE = re.compile(r"^(?P<indent>[ ]{0,3})(?P<fence>`{3,}|~{3,})(?P<info>[^`]*)$")
STRUCTURED = {"exercise", "quiz", "viz", "concurrency", "walkthrough"}
SQL_FLAGS = {"run", "error", "volatile", "measured"}

# Canonical lesson sections (docs/AUTHORING.md). Each maps to one of the 7 learning stages.
LESSON_SECTIONS = [
    ("Introduction", "concept"),
    ("DBMS Theory", "concept"),
    ("Why It Exists", "concept"),
    ("Internal Working", "internals"),
    ("MySQL Implementation", "implementation"),
    ("Syntax Breakdown", "implementation"),
    ("Practical Examples", "implementation"),
    ("Execution Walkthrough", "execution"),
    ("Output", "execution"),
    ("Alternative Approaches", "optimization"),
    ("Performance Analysis", "optimization"),
    ("Best Practices", "optimization"),
    ("Common Mistakes", "optimization"),
    ("Industry Application", "industry"),
    ("Practice Exercises", "assessment"),
    ("Knowledge Assessment", "assessment"),
]
SECTION_STAGE = {name: stage for name, stage in LESSON_SECTIONS}


class ContentError(ValueError):
    pass


@dataclass
class ParsedDoc:
    meta: dict
    segments: list[dict]
    sections: list[dict]
    errors: list[str] = field(default_factory=list)


def sql_hash(sql: str) -> str:
    return hashlib.sha1(sql.strip().encode()).hexdigest()[:12]


def split_front_matter(text: str) -> tuple[dict, str]:
    if not text.startswith("---"):
        raise ContentError("missing YAML front matter (file must start with ---)")
    end = text.find("\n---", 3)
    if end < 0:
        raise ContentError("unterminated YAML front matter")
    meta = yaml.safe_load(text[3:end]) or {}
    body = text[end + 4:].lstrip("\n")
    return meta, body


def slugify(s: str) -> str:
    s = re.sub(r"[`*_]", "", s.lower())
    return re.sub(r"[^a-z0-9]+", "-", s).strip("-")


def parse_document(text: str, doc_id_hint: str = "") -> ParsedDoc:
    meta, body = split_front_matter(text)
    doc_id = meta.get("id", doc_id_hint)
    segments: list[dict] = []
    sections: list[dict] = []
    errors: list[str] = []
    md_buf: list[str] = []
    counters: dict[str, int] = {}

    def next_id(prefix: str) -> str:
        counters[prefix] = counters.get(prefix, 0) + 1
        return f"{prefix}{counters[prefix]}"

    def flush_md():
        content = "\n".join(md_buf).strip("\n")
        if content.strip():
            segments.append({"type": "markdown", "content": content})
        md_buf.clear()

    lines = body.split("\n")
    i = 0
    container_depth = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if container_depth == 0 and line.startswith("## "):
            title = line[3:].strip()
            anchor = slugify(title)
            sections.append({"title": title, "anchor": anchor, "stage": SECTION_STAGE.get(title, "concept")})
            flush_md()
            segments.append({"type": "heading", "title": title, "anchor": anchor,
                             "stage": SECTION_STAGE.get(title, "concept")})
            i += 1
            continue

        if re.match(r"^:{3,}\s*[A-Za-z]", stripped):
            container_depth += 1
        elif re.match(r"^:{3,}\s*$", stripped) and container_depth > 0:
            container_depth -= 1

        m = FENCE.match(line)
        if not m:
            md_buf.append(line)
            i += 1
            continue

        fence, info = m.group("fence"), m.group("info").strip()
        # find closing fence
        j = i + 1
        while j < len(lines):
            mm = FENCE.match(lines[j])
            if mm and mm.group("fence")[0] == fence[0] and len(mm.group("fence")) >= len(fence) and not mm.group("info").strip():
                break
            j += 1
        if j >= len(lines):
            errors.append(f"unterminated code fence starting at body line {i + 1}")
            md_buf.extend(lines[i:])
            break
        code = "\n".join(lines[i + 1:j])
        words = info.split()
        lang = words[0].lower() if words else ""
        flags = {w.lower() for w in words[1:]}

        if container_depth > 0 and (lang in STRUCTURED or (lang == "sql" and "run" in flags)):
            errors.append(f"line {i + 1}: '{info}' blocks must not be nested inside a ::: callout")

        if container_depth == 0 and lang == "sql":
            unknown = flags - SQL_FLAGS
            if unknown:
                errors.append(f"line {i + 1}: unknown sql flags {sorted(unknown)}")
            flush_md()
            seg = {"type": "sql", "id": next_id("s"), "sql": code.strip("\n"), "run": "run" in flags,
                   "expect": "error" if "error" in flags else "ok", "volatile": "volatile" in flags,
                   "measured": "measured" in flags}
            seg["key"] = f"{seg['id']}:{sql_hash(seg['sql'])}"
            segments.append(seg)
        elif container_depth == 0 and lang in STRUCTURED:
            flush_md()
            try:
                data = yaml.safe_load(code)
            except yaml.YAMLError as e:
                errors.append(f"line {i + 1}: invalid YAML in {lang} block: {e}")
                data = None
            if data is not None:
                seg = _structured(lang, data, doc_id, next_id, errors, i + 1)
                if seg:
                    segments.append(seg)
        else:
            md_buf.extend(lines[i:j + 1])
        i = j + 1

    flush_md()
    for seg in segments:
        if seg["type"] == "exercise":
            seg["exercise"].setdefault("dataset", meta.get("dataset", "empty"))
    if container_depth != 0:
        errors.append("unbalanced ::: callout containers")
    return ParsedDoc(meta=meta, segments=segments, sections=sections, errors=errors)


def _structured(kind: str, data, doc_id: str, next_id, errors: list[str], line: int) -> dict | None:
    where = f"line {line} ({kind})"
    if kind == "exercise":
        if not isinstance(data, dict):
            errors.append(f"{where}: exercise must be a mapping")
            return None
        local = str(data.get("id") or next_id("ex"))
        ex = dict(data)
        ex["id"] = local if "/" in local else f"{doc_id}/{local}"
        ex["kind"] = ex.get("kind", "query")
        for req in ("title", "prompt", "solution"):
            if not ex.get(req):
                errors.append(f"{where}: exercise {ex['id']} missing '{req}'")
        if ex["kind"] not in ("query", "dml", "open"):
            errors.append(f"{where}: exercise {ex['id']} has invalid kind {ex['kind']}")
        if ex["kind"] == "dml" and not ex.get("check"):
            errors.append(f"{where}: dml exercise {ex['id']} needs a 'check' query")
        if ex["kind"] != "open" and not ex.get("hints"):
            errors.append(f"{where}: exercise {ex['id']} should have at least one hint")
        return {"type": "exercise", "exercise": ex}
    if kind == "quiz":
        qs = data.get("questions") if isinstance(data, dict) else data
        if not isinstance(qs, list) or not qs:
            errors.append(f"{where}: quiz must be a non-empty list of questions")
            return None
        out = []
        for q in qs:
            q = dict(q)
            local = str(q.get("id") or next_id("q"))
            q["id"] = local if "/" in local else f"{doc_id}/{local}"
            _check_question(q, errors, where)
            out.append(q)
        return {"type": "quiz", "id": next_id("quiz"), "title": data.get("title") if isinstance(data, dict) else None,
                "questions": out}
    if kind == "viz":
        if not isinstance(data, dict) or "type" not in data:
            errors.append(f"{where}: viz block needs a 'type'")
            return None
        props = {k: v for k, v in data.items() if k != "type"}
        return {"type": "viz", "id": next_id("viz"), "viz": data["type"], "props": props}
    if kind == "concurrency":
        if not isinstance(data, dict) or not isinstance(data.get("steps"), list):
            errors.append(f"{where}: concurrency block needs 'steps'")
            return None
        cid = next_id("c")
        steps = []
        for st in data["steps"]:
            st = dict(st)
            if st.get("observe"):
                st["observe"] = "locks"
                st["expect"] = "ok"
                steps.append(st)
                continue
            if st.get("session") not in ("A", "B", "C"):
                errors.append(f"{where}: each step needs session A, B or C")
            if not st.get("sql"):
                errors.append(f"{where}: each step needs sql")
            st["expect"] = st.get("expect", "ok")
            if st["expect"] not in ("ok", "blocks", "error", "any"):
                errors.append(f"{where}: invalid expect '{st['expect']}'")
            steps.append(st)
        body = yaml.safe_dump({"setup": data.get("setup"), "steps": steps}, sort_keys=True)
        return {"type": "concurrency", "id": cid, "key": f"{cid}:{sql_hash(body)}", "title": data.get("title"),
                "description": data.get("description"), "dataset": data.get("dataset"),
                "setup": data.get("setup") or "", "steps": steps}
    if kind == "walkthrough":
        if not isinstance(data, dict) or not isinstance(data.get("steps"), list):
            errors.append(f"{where}: walkthrough block needs 'steps'")
            return None
        wid = next_id("w")
        steps = []
        for k, st in enumerate(data["steps"]):
            st = dict(st)
            if st.get("sql"):
                st["key"] = f"{wid}.{k}:{sql_hash(st['sql'])}"
            steps.append(st)
        return {"type": "walkthrough", "id": wid, "title": data.get("title"), "steps": steps}
    return None


def _check_question(q: dict, errors: list[str], where: str):
    qtype = q.get("type", "single")
    q["type"] = qtype
    if qtype not in ("single", "multi"):
        errors.append(f"{where}: question {q['id']} invalid type {qtype}")
        return
    opts = q.get("options")
    if not q.get("question") or not isinstance(opts, list) or len(opts) < 2:
        errors.append(f"{where}: question {q['id']} needs 'question' and at least 2 options")
        return
    ans = q.get("answer")
    answers = ans if isinstance(ans, list) else [ans]
    if qtype == "single" and isinstance(ans, list):
        errors.append(f"{where}: single-answer question {q['id']} has a list answer")
    if not all(isinstance(a, int) and 0 <= a < len(opts) for a in answers):
        errors.append(f"{where}: question {q['id']} answer index out of range")
    if not q.get("explanation"):
        errors.append(f"{where}: question {q['id']} needs an explanation")
