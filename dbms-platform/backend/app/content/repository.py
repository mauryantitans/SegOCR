"""In-memory course content repository.

Loads curriculum.yaml, lesson/project Markdown, assessments and captured
outputs at startup. Public views are *sanitized*: exercise solutions, quiz
answers and model answers never leave the server until the grading endpoints
decide the learner may see them.
"""
from __future__ import annotations

import copy
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .parser import LESSON_SECTIONS, ContentError, ParsedDoc, parse_document

log = logging.getLogger("content")

SECRET_EXERCISE_FIELDS = {"solution", "explanation", "check"}
SECRET_QUESTION_FIELDS = {"answer", "explanation"}


@dataclass
class LessonRecord:
    id: str
    module_id: str
    chapter_id: str
    title: str
    level: str
    minutes: int
    prerequisites: list[str]
    topics: list[str]
    order: int
    path: Path
    doc: ParsedDoc | None = None
    load_error: str | None = None


@dataclass
class Repository:
    root: Path
    course: dict = field(default_factory=dict)
    modules: list[dict] = field(default_factory=list)
    lessons: dict[str, LessonRecord] = field(default_factory=dict)
    projects: dict[str, dict] = field(default_factory=dict)
    assessments: dict[str, dict] = field(default_factory=dict)
    exams: list[dict] = field(default_factory=list)
    exercises: dict[str, dict] = field(default_factory=dict)
    questions: dict[str, dict] = field(default_factory=dict)
    captures: dict[str, dict] = field(default_factory=dict)
    capture_meta: dict[str, dict] = field(default_factory=dict)
    datasets: dict[str, dict] = field(default_factory=dict)

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(cls, root: Path) -> "Repository":
        repo = cls(root=root)
        repo._load()
        return repo

    def _load(self):
        cur = yaml.safe_load((self.root / "curriculum.yaml").read_text())
        self.course = cur["course"]
        self.modules = cur["modules"]
        self.exams = cur.get("exams", [])
        ds_file = self.root / "datasets" / "datasets.yaml"
        self.datasets = yaml.safe_load(ds_file.read_text()) if ds_file.exists() else {}
        self._load_captures()
        order = 0
        for m in self.modules:
            for c in m["chapters"]:
                for l in c["lessons"]:
                    order += 1
                    rec = LessonRecord(
                        id=l["id"], module_id=m["id"], chapter_id=c["id"], title=l["title"], level=l["level"],
                        minutes=l.get("minutes", 30), prerequisites=l.get("prerequisites", []),
                        topics=l.get("topics", []), order=order,
                        path=self.root / "lessons" / m["id"] / f"{l['id']}.md")
                    if rec.path.exists():
                        self._parse_into(rec)
                    self.lessons[rec.id] = rec
        for p in cur.get("projects", []):
            path = self.root / "projects" / f"{p['id']}.md"
            entry = dict(p, authored=path.exists(), doc=None, load_error=None)
            if path.exists():
                try:
                    entry["doc"] = parse_document(path.read_text(), p["id"])
                    self._index_doc(entry["doc"])
                except (ContentError, yaml.YAMLError) as e:
                    entry["load_error"] = str(e)
                    log.error("project %s failed to load: %s", p["id"], e)
            self.projects[p["id"]] = entry
        adir = self.root / "assessments"
        if adir.exists():
            for f in sorted(adir.glob("*.yaml")):
                try:
                    a = yaml.safe_load(f.read_text())
                    self._index_assessment(a)
                    self.assessments[a["id"]] = a
                except Exception as e:  # noqa: BLE001 - log and skip broken content
                    log.error("assessment %s failed to load: %s", f.name, e)

    def _load_captures(self):
        cdir = self.root / "generated" / "captures"
        if not cdir.exists():
            return
        for f in cdir.glob("*.json"):
            data = json.loads(f.read_text())
            self.capture_meta[data.get("doc", f.stem)] = data.get("meta", {})
            for k, v in data.get("blocks", {}).items():
                self.captures[f"{data.get('doc', f.stem)}|{k}"] = v

    def _parse_into(self, rec: LessonRecord):
        try:
            rec.doc = parse_document(rec.path.read_text(), rec.id)
            if rec.doc.meta.get("id") != rec.id:
                rec.doc.errors.append(f"front matter id '{rec.doc.meta.get('id')}' does not match curriculum id")
            self._index_doc(rec.doc)
        except (ContentError, yaml.YAMLError) as e:
            rec.load_error = str(e)
            log.error("lesson %s failed to load: %s", rec.id, e)

    def _index_doc(self, doc: ParsedDoc):
        default_ds = doc.meta.get("dataset", "empty")
        for seg in doc.segments:
            if seg["type"] == "exercise":
                ex = seg["exercise"]
                ex.setdefault("dataset", default_ds)
                ex.setdefault("concepts", doc.meta.get("concepts", []))
                ex["doc"] = doc.meta.get("id")
                self.exercises[ex["id"]] = ex
            elif seg["type"] == "quiz":
                for q in seg["questions"]:
                    q.setdefault("concepts", doc.meta.get("concepts", []))
                    q["doc"] = doc.meta.get("id")
                    self.questions[q["id"]] = q

    def _index_assessment(self, a: dict):
        for q in a.get("questions", []):
            if "/" not in str(q["id"]):
                q["id"] = f"{a['id']}/{q['id']}"
            q.setdefault("type", "single")
            q["doc"] = a["id"]
            self.questions[q["id"]] = q
        for ex in a.get("challenges", []):
            if "/" not in str(ex["id"]):
                ex["id"] = f"{a['id']}/{ex['id']}"
            ex.setdefault("kind", "query")
            ex.setdefault("dataset", "shop")
            ex["doc"] = a["id"]
            self.exercises[ex["id"]] = ex
        for iq in a.get("interview", []):
            if "/" not in str(iq["id"]):
                iq["id"] = f"{a['id']}/{iq['id']}"

    # ------------------------------------------------------------------ public views
    def lesson_status(self, rec: LessonRecord) -> str:
        if rec.doc is not None and not rec.load_error:
            return "authored"
        return "planned"

    def course_tree(self) -> dict:
        modules = []
        for m in self.modules:
            chapters = []
            for c in m["chapters"]:
                chapters.append({"id": c["id"], "title": c["title"], "lessons": [
                    self.lesson_summary(self.lessons[l["id"]]) for l in c["lessons"]]})
            modules.append({k: m[k] for k in ("id", "number", "title", "level", "description", "outcomes")} |
                           {"chapters": chapters, "hasAssessment": f"{m['id']}-assessment" in self.assessments})
        projects = [{k: p[k] for k in ("id", "title", "level", "prerequisites", "practices", "authored")}
                    for p in self.projects.values()]
        return {"course": self.course, "modules": modules, "projects": projects,
                "exams": self.exams, "sections": [{"title": t, "stage": s} for t, s in LESSON_SECTIONS]}

    def lesson_summary(self, rec: LessonRecord) -> dict:
        return {"id": rec.id, "title": rec.title, "level": rec.level, "minutes": rec.minutes,
                "prerequisites": rec.prerequisites, "topics": rec.topics, "moduleId": rec.module_id,
                "chapterId": rec.chapter_id, "order": rec.order, "status": self.lesson_status(rec)}

    def _public_segments(self, doc: ParsedDoc, doc_id: str) -> list[dict]:
        out = []
        for seg in doc.segments:
            seg = copy.deepcopy(seg)
            t = seg["type"]
            if t == "sql" and seg["run"]:
                seg["capture"] = self.captures.get(f"{doc_id}|{seg['key']}")
            elif t == "exercise":
                ex = seg.pop("exercise")
                seg.update(public_exercise(ex))
            elif t == "quiz":
                seg["questions"] = [public_question(q) for q in seg["questions"]]
            elif t == "concurrency":
                seg["capture"] = self.captures.get(f"{doc_id}|{seg['key']}")
            elif t == "walkthrough":
                for st in seg["steps"]:
                    if st.get("key"):
                        st["capture"] = self.captures.get(f"{doc_id}|{st['key']}")
            out.append(seg)
        return out

    def lesson_detail(self, lesson_id: str) -> dict | None:
        rec = self.lessons.get(lesson_id)
        if rec is None:
            return None
        base = self.lesson_summary(rec)
        mod = next(m for m in self.modules if m["id"] == rec.module_id)
        base["module"] = {"id": mod["id"], "number": mod["number"], "title": mod["title"]}
        ordered = sorted(self.lessons.values(), key=lambda r: r.order)
        idx = ordered.index(rec)
        base["prev"] = self.lesson_summary(ordered[idx - 1]) if idx > 0 else None
        base["next"] = self.lesson_summary(ordered[idx + 1]) if idx + 1 < len(ordered) else None
        base["prerequisiteLessons"] = [self.lesson_summary(self.lessons[p]) for p in rec.prerequisites if p in self.lessons]
        if rec.doc is None:
            base.update(segments=[], sections=[], meta={})
            return base
        meta = rec.doc.meta
        base.update(
            meta={k: meta.get(k) for k in ("dataset", "concepts", "references", "summary", "mysqlVersion")},
            sections=rec.doc.sections,
            segments=self._public_segments(rec.doc, rec.id),
            captureMeta=self.capture_meta.get(rec.id),
        )
        return base

    def project_detail(self, project_id: str) -> dict | None:
        p = self.projects.get(project_id)
        if p is None:
            return None
        out = {k: p[k] for k in ("id", "title", "level", "prerequisites", "practices", "authored")}
        if p["doc"] is not None:
            out.update(meta={k: p["doc"].meta.get(k) for k in ("dataset", "concepts", "summary")},
                       sections=p["doc"].sections, segments=self._public_segments(p["doc"], project_id),
                       captureMeta=self.capture_meta.get(project_id))
        return out

    def assessment_public(self, assessment_id: str) -> dict | None:
        a = self.assessments.get(assessment_id)
        if a is None:
            return None
        return {"id": a["id"], "title": a.get("title"), "module": a.get("module"), "passMark": a.get("passMark", 70),
                "description": a.get("description"),
                "questions": [public_question(q) for q in a.get("questions", [])],
                "challenges": [public_exercise(e) for e in a.get("challenges", [])],
                "interview": [{"id": i["id"], "question": i["question"], "concepts": i.get("concepts", [])}
                              for i in a.get("interview", [])]}

    def interview_answer(self, qid: str) -> dict | None:
        for a in self.assessments.values():
            for i in a.get("interview", []):
                if i["id"] == qid:
                    return i
        return None

    def exam_questions(self, exam_id: str) -> list[dict]:
        exam = next((e for e in self.exams if e["id"] == exam_id), None)
        if exam is None:
            return []
        pool = [q for a in self.assessments.values() if a.get("module") in exam["modules"] for q in a.get("questions", [])]
        return pool

    def concept_index(self) -> dict[str, list[str]]:
        idx: dict[str, list[str]] = {}
        for rec in self.lessons.values():
            if rec.doc:
                for c in rec.doc.meta.get("concepts", []) or []:
                    idx.setdefault(c, []).append(rec.id)
        return idx


def public_exercise(ex: dict) -> dict:
    out = {k: v for k, v in ex.items() if k not in SECRET_EXERCISE_FIELDS}
    out["type"] = "exercise"
    if ex.get("kind") == "open":
        out.pop("hints", None) if not ex.get("hints") else None
    return out


def public_question(q: dict) -> dict:
    return {k: v for k, v in q.items() if k not in SECRET_QUESTION_FIELDS}
