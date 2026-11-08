"""FastAPI application entry point: ``uvicorn app.main:app``."""
from __future__ import annotations

import logging
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from . import models  # noqa: F401  (register ORM models)
from .config import REPO_ROOT, get_settings
from .db import Base, get_engine
from .routers import auth, content, labs, learning, progress
from .state import state

log = logging.getLogger("app")


def _configure_logging(level: str):
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def _cleanup_loop(stop: threading.Event):
    while not stop.wait(600):
        try:
            if state.labs:
                log.info("lab cleanup: %s", state.labs.cleanup())
        except Exception:  # noqa: BLE001 - keep the loop alive
            log.exception("lab cleanup failed")


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()
    _configure_logging(s.log_level)
    if s.environment == "production" and s.secret_key.startswith("dev-only"):
        raise RuntimeError("DBMS_SECRET_KEY must be set in production")
    if get_engine().url.get_backend_name() == "sqlite":
        Base.metadata.create_all(get_engine())  # dev convenience; MySQL deployments use Alembic migrations
    state.init(with_sandbox=True)
    log.info("content loaded: %d lessons (%d authored), %d exercises, %d questions",
             len(state.repo.lessons), sum(1 for r in state.repo.lessons.values() if r.doc), len(state.repo.exercises),
             len(state.repo.questions))
    stop = threading.Event()
    threading.Thread(target=_cleanup_loop, args=(stop,), daemon=True, name="lab-cleanup").start()
    yield
    stop.set()
    if state.labs:
        state.labs.shutdown()


app = FastAPI(title="DBMS & MySQL Learning Platform API", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=get_settings().cors_origins, allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])


@app.middleware("http")
async def access_log(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    if request.url.path.startswith("/api"):
        logging.getLogger("access").info("%s %s %s %.1fms", request.method, request.url.path, response.status_code,
                                         (time.perf_counter() - t0) * 1000)
    return response


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):
    log.exception("unhandled error on %s", request.url.path)
    return JSONResponse({"detail": "Internal server error"}, status_code=500)


for r in (auth.router, content.router, labs.router, learning.router, progress.router):
    app.include_router(r)


@app.get("/api/health")
def health():
    sandbox = "not configured"
    if state.labs:
        try:
            sandbox = state.labs.server_version()
        except Exception as e:  # noqa: BLE001
            sandbox = f"unavailable: {type(e).__name__}"
    return {"status": "ok", "sandboxMysql": sandbox}


# Serve the built frontend (single-container deployments). In development Vite serves it instead.
DIST = REPO_ROOT / "frontend" / "dist"
if DIST.exists():
    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        f = (DIST / path).resolve()
        if path and f.is_file() and DIST.resolve() in f.parents:
            return FileResponse(f)
        return FileResponse(DIST / "index.html")
