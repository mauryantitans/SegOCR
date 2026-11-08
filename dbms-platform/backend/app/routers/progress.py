"""Progress tracking, bookmarks, notes, weak-concept analysis and activity."""
from __future__ import annotations

import datetime as dt
from collections import defaultdict

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from ..db import get_db
from ..models import AssessmentResult, Bookmark, ExerciseAttempt, LessonProgress, Note, QuizAttempt, User, _now
from ..security import current_user
from ..state import state

router = APIRouter(prefix="/api/progress", tags=["progress"])


class StatusIn(BaseModel):
    status: str = Field(pattern="^(in_progress|completed)$")


class BookmarkIn(BaseModel):
    target: str = Field(max_length=200)
    label: str = Field(max_length=200)


class NoteIn(BaseModel):
    body: str = Field(max_length=100_000)


def _iso(d: dt.datetime | None) -> str | None:
    return d.isoformat() + "Z" if d else None


@router.get("")
def summary(user: User = Depends(current_user), db: Session = Depends(get_db)):
    rows = db.scalars(select(LessonProgress).where(LessonProgress.user_id == user.id)).all()
    lessons = {r.lesson_id: {"status": r.status, "startedAt": _iso(r.started_at), "completedAt": _iso(r.completed_at),
                             "updatedAt": _iso(r.updated_at)} for r in rows}
    solved = db.scalars(select(ExerciseAttempt.exercise_id).where(ExerciseAttempt.user_id == user.id,
                                                                  ExerciseAttempt.passed.is_(True)).distinct()).all()
    quiz = db.execute(select(QuizAttempt.question_id, QuizAttempt.correct).where(QuizAttempt.user_id == user.id)
                      .order_by(QuizAttempt.id)).all()
    last_answer = {qid: ok for qid, ok in quiz}
    assessments = db.scalars(select(AssessmentResult).where(AssessmentResult.user_id == user.id)
                             .order_by(AssessmentResult.id.desc())).all()
    best: dict[str, dict] = {}
    for a in assessments:
        b = best.get(a.assessment_id)
        if b is None or a.score > b["score"]:
            best[a.assessment_id] = {"score": a.score, "passed": a.passed, "at": _iso(a.created_at)}
    authored = [r for r in state.repo.lessons.values() if state.repo.lesson_status(r) == "authored"]
    completed = sum(1 for r in authored if lessons.get(r.id, {}).get("status") == "completed")
    return {"lessons": lessons, "solvedExercises": solved, "answeredQuestions": last_answer, "assessments": best,
            "stats": {"authoredLessons": len(authored), "completedLessons": completed,
                      "solvedExercises": len(solved), "questionsAnswered": len(last_answer),
                      "questionsCorrect": sum(1 for v in last_answer.values() if v)}}


@router.put("/lessons/{lesson_id}")
def set_status(lesson_id: str, body: StatusIn, user: User = Depends(current_user), db: Session = Depends(get_db)):
    if lesson_id not in state.repo.lessons and lesson_id not in state.repo.projects:
        raise HTTPException(404, "Unknown lesson")
    row = db.scalar(select(LessonProgress).where(LessonProgress.user_id == user.id, LessonProgress.lesson_id == lesson_id))
    if row is None:
        row = LessonProgress(user_id=user.id, lesson_id=lesson_id, status=body.status)
        db.add(row)
    elif row.status == "completed" and body.status == "in_progress":
        return {"status": row.status}  # opening a completed lesson again doesn't un-complete it
    row.status = body.status
    row.completed_at = _now() if body.status == "completed" else None
    db.commit()
    return {"status": row.status}


@router.delete("/lessons/{lesson_id}")
def reset_status(lesson_id: str, user: User = Depends(current_user), db: Session = Depends(get_db)):
    db.execute(delete(LessonProgress).where(LessonProgress.user_id == user.id, LessonProgress.lesson_id == lesson_id))
    db.commit()
    return {"status": None}


@router.get("/bookmarks")
def bookmarks(user: User = Depends(current_user), db: Session = Depends(get_db)):
    rows = db.scalars(select(Bookmark).where(Bookmark.user_id == user.id).order_by(Bookmark.created_at.desc())).all()
    return [{"id": b.id, "target": b.target, "label": b.label, "createdAt": _iso(b.created_at)} for b in rows]


@router.post("/bookmarks", status_code=201)
def add_bookmark(body: BookmarkIn, user: User = Depends(current_user), db: Session = Depends(get_db)):
    existing = db.scalar(select(Bookmark).where(Bookmark.user_id == user.id, Bookmark.target == body.target))
    if existing:
        return {"id": existing.id}
    b = Bookmark(user_id=user.id, target=body.target, label=body.label)
    db.add(b)
    db.commit()
    return {"id": b.id}


@router.delete("/bookmarks/{bid}")
def delete_bookmark(bid: int, user: User = Depends(current_user), db: Session = Depends(get_db)):
    db.execute(delete(Bookmark).where(Bookmark.id == bid, Bookmark.user_id == user.id))
    db.commit()
    return {"ok": True}


@router.get("/notes")
def notes(user: User = Depends(current_user), db: Session = Depends(get_db)):
    rows = db.scalars(select(Note).where(Note.user_id == user.id).order_by(Note.updated_at.desc())).all()
    return [{"lessonId": n.lesson_id, "body": n.body, "updatedAt": _iso(n.updated_at)} for n in rows]


@router.get("/notes/{lesson_id}")
def note(lesson_id: str, user: User = Depends(current_user), db: Session = Depends(get_db)):
    n = db.scalar(select(Note).where(Note.user_id == user.id, Note.lesson_id == lesson_id))
    return {"lessonId": lesson_id, "body": n.body if n else "", "updatedAt": _iso(n.updated_at) if n else None}


@router.put("/notes/{lesson_id}")
def save_note(lesson_id: str, body: NoteIn, user: User = Depends(current_user), db: Session = Depends(get_db)):
    n = db.scalar(select(Note).where(Note.user_id == user.id, Note.lesson_id == lesson_id))
    if not body.body.strip():
        if n:
            db.delete(n)
            db.commit()
        return {"lessonId": lesson_id, "body": ""}
    if n is None:
        n = Note(user_id=user.id, lesson_id=lesson_id, body=body.body)
        db.add(n)
    else:
        n.body = body.body
    db.commit()
    return {"lessonId": lesson_id, "body": n.body, "updatedAt": _iso(n.updated_at)}


@router.get("/weak-concepts")
def weak_concepts(user: User = Depends(current_user), db: Session = Depends(get_db)):
    """Concepts the learner repeatedly gets wrong, weighting recent attempts more heavily."""
    stats: dict[str, dict] = defaultdict(lambda: {"attempts": 0, "failures": 0, "score": 0.0})
    quiz = db.execute(select(QuizAttempt.concepts, QuizAttempt.correct).where(QuizAttempt.user_id == user.id)
                      .order_by(QuizAttempt.id)).all()
    ex = db.execute(select(ExerciseAttempt.concepts, ExerciseAttempt.passed).where(ExerciseAttempt.user_id == user.id)
                    .order_by(ExerciseAttempt.id)).all()
    for concepts, ok in list(quiz) + list(ex):
        for c in filter(None, (concepts or "").split(",")):
            s = stats[c]
            s["attempts"] += 1
            s["failures"] += 0 if ok else 1
            # exponential moving accuracy: recent results count more than old ones
            s["score"] = 0.6 * s["score"] + 0.4 * (1.0 if ok else 0.0) if s["attempts"] > 1 else (1.0 if ok else 0.0)
    index = state.repo.concept_index()
    out = []
    for c, s in stats.items():
        if s["failures"] >= 2 and s["score"] < 0.7:
            out.append({"concept": c, "attempts": s["attempts"], "failures": s["failures"],
                        "recentAccuracy": round(s["score"], 2),
                        "lessons": [state.repo.lesson_summary(state.repo.lessons[l]) for l in index.get(c, [])[:4]]})
    out.sort(key=lambda x: (x["recentAccuracy"], -x["failures"]))
    return out


@router.get("/activity")
def activity(user: User = Depends(current_user), db: Session = Depends(get_db)):
    events = []
    for r in db.scalars(select(LessonProgress).where(LessonProgress.user_id == user.id)
                        .order_by(LessonProgress.updated_at.desc()).limit(20)):
        events.append({"type": "lesson", "id": r.lesson_id, "status": r.status, "at": _iso(r.updated_at)})
    for a in db.scalars(select(ExerciseAttempt).where(ExerciseAttempt.user_id == user.id)
                        .order_by(ExerciseAttempt.id.desc()).limit(20)):
        events.append({"type": "exercise", "id": a.exercise_id, "passed": a.passed, "at": _iso(a.created_at)})
    for a in db.scalars(select(AssessmentResult).where(AssessmentResult.user_id == user.id)
                        .order_by(AssessmentResult.id.desc()).limit(10)):
        events.append({"type": "assessment", "id": a.assessment_id, "score": a.score, "passed": a.passed,
                       "at": _iso(a.created_at)})
    events.sort(key=lambda e: e["at"] or "", reverse=True)
    return events[:30]
