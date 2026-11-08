"""Exercises, quizzes, module assessments and cumulative exams (graded server-side)."""
from __future__ import annotations

import datetime as dt
import random

from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..content.repository import public_question
from ..db import get_db
from ..models import AssessmentResult, ExerciseAttempt, QuizAttempt, User
from ..sandbox.manager import SandboxError
from ..security import current_user
from ..state import state
from .labs import owner_of

router = APIRouter(prefix="/api", tags=["learning"])

SOLUTION_UNLOCK_ATTEMPTS = 2


class AttemptIn(BaseModel):
    submission: str = Field(min_length=1, max_length=50_000)
    hints_used: int = 0


class AnswerIn(BaseModel):
    question_id: str
    answer: int | list[int]


class SubmitIn(BaseModel):
    answers: dict[str, int | list[int]]


def _exercise(eid: str) -> dict:
    ex = state.repo.exercises.get(eid)
    if ex is None:
        raise HTTPException(404, "Exercise not found")
    return ex


def _attempt_stats(db: Session, user: User, eid: str) -> tuple[int, int]:
    total = db.scalar(select(func.count()).select_from(ExerciseAttempt)
                      .where(ExerciseAttempt.user_id == user.id, ExerciseAttempt.exercise_id == eid)) or 0
    passed = db.scalar(select(func.count()).select_from(ExerciseAttempt)
                       .where(ExerciseAttempt.user_id == user.id, ExerciseAttempt.exercise_id == eid,
                              ExerciseAttempt.passed.is_(True))) or 0
    return total, passed


def _solution_unlocked(total: int, passed: int, kind: str) -> bool:
    return passed > 0 or total >= SOLUTION_UNLOCK_ATTEMPTS or (kind == "open" and total >= 1)


@router.get("/exercises/{eid:path}/status")
def exercise_status(eid: str, user: User = Depends(current_user), db: Session = Depends(get_db)):
    ex = _exercise(eid)
    total, passed = _attempt_stats(db, user, eid)
    last = db.scalar(select(ExerciseAttempt).where(ExerciseAttempt.user_id == user.id, ExerciseAttempt.exercise_id == eid)
                     .order_by(ExerciseAttempt.id.desc()).limit(1))
    return {"attempts": total, "passed": passed > 0, "solutionUnlocked": _solution_unlocked(total, passed, ex.get("kind", "query")),
            "lastSubmission": last.submission if last else None}


@router.post("/exercises/{eid:path}/attempt")
async def attempt(eid: str, body: AttemptIn, user: User = Depends(current_user), db: Session = Depends(get_db)):
    ex = _exercise(eid)
    kind = ex.get("kind", "query")
    total, passed_before = _attempt_stats(db, user, eid)
    concepts = ",".join(ex.get("concepts", []))[:500]
    if kind == "open":
        if len(body.submission.strip()) < 40:
            raise HTTPException(422, "Write a fuller answer (at least a few sentences) before comparing with the model answer.")
        db.add(ExerciseAttempt(user_id=user.id, exercise_id=eid, doc_id=ex.get("doc", ""), submission=body.submission,
                               passed=True, feedback="self-assessed", hints_used=body.hints_used, concepts=concepts))
        db.commit()
        return {"passed": True, "summary": "Answer saved. Compare it with the model answer below and note what you missed.",
                "details": [], "solution": ex["solution"], "explanation": ex.get("explanation"), "solutionUnlocked": True,
                "attempts": total + 1}
    state.limiter.check(owner_of(user))
    grader = state.grader
    if grader is None:
        raise HTTPException(503, "Grading requires the SQL sandbox.")
    failures = total - passed_before
    try:
        result = await run_in_threadpool(grader.grade, owner_of(user), ex, body.submission, failures)
    except SandboxError as e:
        raise HTTPException(400, str(e)) from e
    db.add(ExerciseAttempt(user_id=user.id, exercise_id=eid, doc_id=ex.get("doc", ""), submission=body.submission,
                           passed=result.passed, feedback=result.summary[:2000], hints_used=body.hints_used,
                           concepts=concepts))
    db.commit()
    out = result.to_dict()
    unlocked = _solution_unlocked(total + 1, passed_before + int(result.passed), kind)
    out.update(attempts=total + 1, solutionUnlocked=unlocked)
    if result.passed:
        out.update(solution=ex["solution"], explanation=ex.get("explanation"))
    return out


@router.get("/exercises/{eid:path}/solution")
def solution(eid: str, user: User = Depends(current_user), db: Session = Depends(get_db)):
    ex = _exercise(eid)
    total, passed = _attempt_stats(db, user, eid)
    if not _solution_unlocked(total, passed, ex.get("kind", "query")):
        raise HTTPException(403, f"Make at least {SOLUTION_UNLOCK_ATTEMPTS} attempts (or solve it) to unlock the reference solution.")
    return {"solution": ex["solution"], "explanation": ex.get("explanation")}


# ---------------------------------------------------------------- quizzes
def _grade_question(q: dict, answer) -> bool:
    expected = q["answer"]
    if q.get("type", "single") == "multi":
        exp = sorted(expected if isinstance(expected, list) else [expected])
        got = sorted(answer if isinstance(answer, list) else [answer])
        return exp == got
    return (answer[0] if isinstance(answer, list) and len(answer) == 1 else answer) == expected


def _record(db: Session, user: User, q: dict, answer, correct: bool):
    db.add(QuizAttempt(user_id=user.id, question_id=q["id"], doc_id=q.get("doc", ""), answer=str(answer)[:100],
                       correct=correct, concepts=",".join(q.get("concepts", []))[:500]))


@router.post("/quiz/answer")
def answer(body: AnswerIn, user: User = Depends(current_user), db: Session = Depends(get_db)):
    q = state.repo.questions.get(body.question_id)
    if q is None:
        raise HTTPException(404, "Question not found")
    correct = _grade_question(q, body.answer)
    _record(db, user, q, body.answer, correct)
    db.commit()
    return {"correct": correct, "answer": q["answer"], "explanation": q.get("explanation")}


# ---------------------------------------------------------------- assessments & exams
def _submit(db: Session, user: User, assessment_id: str, questions: list[dict], answers: dict, pass_mark: int) -> dict:
    results, correct = [], 0
    for q in questions:
        given = answers.get(q["id"])
        ok = given is not None and _grade_question(q, given)
        correct += ok
        if given is not None:
            _record(db, user, q, given, ok)
        results.append({"questionId": q["id"], "correct": ok, "answer": q["answer"], "explanation": q.get("explanation")})
    total = len(questions)
    score = round(100 * correct / total) if total else 0
    db.add(AssessmentResult(user_id=user.id, assessment_id=assessment_id, score=score, passed=score >= pass_mark,
                            total=total, correct=correct))
    db.commit()
    return {"score": score, "passed": score >= pass_mark, "correct": correct, "total": total, "passMark": pass_mark,
            "results": results}


@router.get("/assessments/{aid}")
def assessment(aid: str):
    a = state.repo.assessment_public(aid)
    if a is None:
        raise HTTPException(404, "Assessment not found")
    return a


@router.post("/assessments/{aid}/submit")
def submit_assessment(aid: str, body: SubmitIn, user: User = Depends(current_user), db: Session = Depends(get_db)):
    a = state.repo.assessments.get(aid)
    if a is None:
        raise HTTPException(404, "Assessment not found")
    return _submit(db, user, aid, a.get("questions", []), body.answers, a.get("passMark", 70))


@router.get("/interview/{qid:path}/answer")
def interview_answer(qid: str, user: User = Depends(current_user)):
    i = state.repo.interview_answer(qid)
    if i is None:
        raise HTTPException(404, "Question not found")
    return {"answer": i["answer"]}


def _exam_questions(exam: dict, user: User) -> list[dict]:
    pool = state.repo.exam_questions(exam["id"])
    rnd = random.Random(f"{user.id}:{exam['id']}:{dt.date.today().isoformat()}")  # stable for a day per learner
    return rnd.sample(pool, min(exam.get("questions", 30), len(pool)))


@router.get("/exams/{exam_id}")
def exam(exam_id: str, user: User = Depends(current_user)):
    e = next((x for x in state.repo.exams if x["id"] == exam_id), None)
    if e is None:
        raise HTTPException(404, "Exam not found")
    qs = _exam_questions(e, user)
    return {"id": e["id"], "title": e["title"], "modules": e["modules"], "passMark": 70,
            "available": len(qs), "questions": [public_question(q) for q in qs]}


@router.post("/exams/{exam_id}/submit")
def submit_exam(exam_id: str, body: SubmitIn, user: User = Depends(current_user), db: Session = Depends(get_db)):
    e = next((x for x in state.repo.exams if x["id"] == exam_id), None)
    if e is None:
        raise HTTPException(404, "Exam not found")
    return _submit(db, user, exam_id, _exam_questions(e, user), body.answers, 70)
