"""SQL sandbox endpoints. All learner SQL goes through LabManager (per-lab MySQL account)."""
from __future__ import annotations

import hashlib
import logging
import re

from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from ..config import get_settings
from ..models import User
from ..sandbox.manager import SandboxError, SessionBusy
from ..sandbox.splitter import split_sql, strip_leading_comments
from ..security import current_user
from ..state import state

router = APIRouter(prefix="/api/labs", tags=["labs"])
log = logging.getLogger("labs")

LAB_KEY = re.compile(r"^(playground|concurrency|lesson:[a-z0-9-]+|exercise:[a-z0-9/-]+|conc:[a-z0-9-]+:c\d+)$")
SESSIONS = {"main", "A", "B", "C"}


def owner_of(user: User) -> str:
    s = get_settings()
    ns = s.lab_namespace or hashlib.sha256(s.app_database_url.encode()).hexdigest()[:8]
    return f"{ns}:user:{user.id}"


def _check_lab(lab: str):
    if not LAB_KEY.match(lab):
        raise HTTPException(422, "Invalid lab identifier.")


class RunIn(BaseModel):
    lab: str = Field(max_length=160)
    sql: str
    dataset: str | None = None
    session: str = "main"
    stop_on_error: bool = True


class ResetIn(BaseModel):
    lab: str = Field(max_length=160)
    dataset: str | None = None


class ExplainIn(BaseModel):
    lab: str = Field(max_length=160)
    sql: str
    dataset: str | None = None


def _lab_setup(lab: str) -> tuple[str | None, str | None]:
    """Datasets and setup SQL for labs tied to content are taken from the server-side content, never the client."""
    repo = state.repo
    if lab.startswith("exercise:"):
        ex = repo.exercises.get(lab.split(":", 1)[1])
        if ex is None:
            raise HTTPException(404, "Unknown exercise")
        return ex.get("dataset", "empty"), ex.get("setup")
    if lab.startswith("conc:"):
        _, doc_id, block_id = lab.split(":")
        rec = repo.lessons.get(doc_id)
        doc = rec.doc if rec else (repo.projects.get(doc_id) or {}).get("doc")
        seg = next((s for s in (doc.segments if doc else []) if s["type"] == "concurrency" and s["id"] == block_id), None)
        if seg is None:
            raise HTTPException(404, "Unknown concurrency scenario")
        return seg.get("dataset") or doc.meta.get("dataset", "empty"), seg.get("setup") or None
    if lab.startswith("lesson:"):
        doc_id = lab.split(":", 1)[1]
        rec = repo.lessons.get(doc_id)
        doc = rec.doc if rec else (repo.projects.get(doc_id) or {}).get("doc")
        if doc is None:
            raise HTTPException(404, "Unknown lesson")
        return doc.meta.get("dataset", "empty"), None
    return None, None


def _ensure(user: User, lab: str, dataset: str | None):
    labs = state.require_labs()
    ds, setup = _lab_setup(lab)
    info = labs.lab_info(owner_of(user), lab)
    if info is None or (lab in ("playground", "concurrency") and dataset and dataset != info["dataset"]):
        labs.reset_lab(owner_of(user), lab, dataset or ds or "shop", setup_sql=setup)


@router.post("/run")
async def run(body: RunIn, user: User = Depends(current_user)):
    _check_lab(body.lab)
    if body.session not in SESSIONS:
        raise HTTPException(422, "session must be main, A, B or C")
    state.limiter.check(owner_of(user))
    labs = state.require_labs()
    concurrent = body.session != "main"

    def work():
        _ensure(user, body.lab, body.dataset)
        return labs.run(owner_of(user), body.lab, body.sql, session_name=body.session,
                        wait_s=1.5 if concurrent else None, stop_on_error=body.stop_on_error,
                        timeout_ms=120_000 if concurrent else None)

    try:
        result = await run_in_threadpool(work)
    except SessionBusy as e:
        raise HTTPException(409, str(e)) from e
    except SandboxError as e:
        raise HTTPException(400, str(e)) from e
    log.info("run user=%s lab=%s session=%s statements=%d ms=%.1f", user.id, body.lab, body.session,
             len(result.statements), result.elapsed_ms)
    return result.to_dict()


@router.get("/poll")
async def poll(lab: str, session: str, user: User = Depends(current_user)):
    _check_lab(lab)
    labs = state.require_labs()
    done = await run_in_threadpool(labs.poll, owner_of(user), lab, session)
    busy = labs.session_busy(owner_of(user), lab, session)
    return {"finished": [d.to_dict() for d in done], "busy": busy}


@router.get("/locks")
async def locks(lab: str, user: User = Depends(current_user)):
    _check_lab(lab)
    labs = state.require_labs()
    return await run_in_threadpool(labs.inspect_locks, owner_of(user), lab)


@router.post("/reset")
async def reset(body: ResetIn, user: User = Depends(current_user)):
    _check_lab(body.lab)
    state.limiter.check(owner_of(user))
    labs = state.require_labs()
    ds, setup = _lab_setup(body.lab)
    dataset = body.dataset if body.lab in ("playground", "concurrency") and body.dataset else (ds or body.dataset or "shop")
    try:
        info = await run_in_threadpool(labs.reset_lab, owner_of(user), body.lab, dataset, setup)
    except SandboxError as e:
        raise HTTPException(400, str(e)) from e
    return {"lab": body.lab, "dataset": info["dataset"]}


@router.get("/schema")
async def schema(lab: str, dataset: str | None = None, user: User = Depends(current_user)):
    _check_lab(lab)
    labs = state.require_labs()

    def work():
        _ensure(user, lab, dataset)
        return labs.schema_overview(owner_of(user), lab)

    return await run_in_threadpool(work)


EXPLAINABLE = {"SELECT", "WITH", "TABLE", "INSERT", "UPDATE", "DELETE", "REPLACE"}


@router.post("/explain")
async def explain(body: ExplainIn, user: User = Depends(current_user)):
    """Visual EXPLAIN: runs EXPLAIN FORMAT=JSON and FORMAT=TREE for a single statement."""
    _check_lab(body.lab)
    state.limiter.check(owner_of(user))
    stmts = split_sql(body.sql)
    if len(stmts) != 1:
        raise HTTPException(422, "Select exactly one statement to explain.")
    stmt = strip_leading_comments(stmts[0].sql).strip()
    if stmt.split(None, 1)[0].upper() == "EXPLAIN":
        raise HTTPException(422, "Pass the statement itself, without EXPLAIN.")
    if stmt.split(None, 1)[0].upper().lstrip("(") not in EXPLAINABLE:
        raise HTTPException(422, "Only SELECT, TABLE, WITH, INSERT, UPDATE, DELETE and REPLACE can be explained.")
    labs = state.require_labs()

    def work():
        _ensure(user, body.lab, body.dataset)
        r = labs.run(owner_of(user), body.lab, f"EXPLAIN FORMAT=JSON {stmt};\nEXPLAIN FORMAT=TREE {stmt}",
                     stop_on_error=False)
        return r

    r = await run_in_threadpool(work)
    js, tree = r.statements[0], r.statements[1] if len(r.statements) > 1 else None
    if js.status != "ok":
        raise HTTPException(400, (js.error or {}).get("message") or "EXPLAIN failed")
    return {"json": js.rows[0][0] if js.rows else None,
            "tree": tree.rows[0][0] if tree and tree.status == "ok" and tree.rows else None,
            "serverVersion": r.server_version}
