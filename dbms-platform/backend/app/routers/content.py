"""Course content delivery (sanitized: no solutions or answers)."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from ..state import state

router = APIRouter(prefix="/api", tags=["content"])


@router.get("/course")
def course():
    return state.repo.course_tree()


@router.get("/lessons/{lesson_id}")
def lesson(lesson_id: str):
    d = state.repo.lesson_detail(lesson_id)
    if d is None:
        raise HTTPException(404, "Lesson not found")
    return d


@router.get("/projects/{project_id}")
def project(project_id: str):
    d = state.repo.project_detail(project_id)
    if d is None:
        raise HTTPException(404, "Project not found")
    return d


@router.get("/datasets")
def datasets():
    return state.repo.datasets


@router.get("/search")
def search(q: str = Query(min_length=2, max_length=100)):
    terms = q.lower().split()
    hits = []
    for rec in state.repo.lessons.values():
        hay = " ".join([rec.title, " ".join(rec.topics), " ".join(s["title"] for s in (rec.doc.sections if rec.doc else []))]).lower()
        if rec.doc:
            hay += " " + " ".join(rec.doc.meta.get("concepts", []) or [])
        score = sum(hay.count(t) + (5 if t in rec.title.lower() else 0) for t in terms)
        if all(t in hay for t in terms):
            hits.append((score, state.repo.lesson_summary(rec)))
    hits.sort(key=lambda h: (-h[0], h[1]["order"]))
    return [h[1] for h in hits[:25]]
