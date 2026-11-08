"""Process-wide singletons (content repository, lab manager, grader, rate limiter)."""
from __future__ import annotations

import threading
import time
from collections import defaultdict, deque

from fastapi import HTTPException

from .config import get_settings
from .content.repository import Repository
from .grading import Grader
from .sandbox.manager import LabManager


class RateLimiter:
    """Sliding-window limiter per key (in-process; use a shared store when running several workers)."""

    def __init__(self, per_minute: int):
        self.per_minute = per_minute
        self._hits: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str):
        now = time.monotonic()
        with self._lock:
            q = self._hits[key]
            while q and now - q[0] > 60:
                q.popleft()
            if len(q) >= self.per_minute:
                raise HTTPException(429, f"Too many sandbox runs — limit is {self.per_minute} per minute. Please slow down.")
            q.append(now)


class AppState:
    repo: Repository
    labs: LabManager | None = None
    grader: Grader | None = None
    limiter: RateLimiter

    def init(self, with_sandbox: bool = True):
        s = get_settings()
        self.repo = Repository.load(s.content_dir)
        self.limiter = RateLimiter(s.runs_per_minute)
        if with_sandbox:
            self.labs = LabManager(s, s.content_dir / "datasets")
            self.grader = Grader(self.labs)

    def require_labs(self) -> LabManager:
        if self.labs is None:
            raise HTTPException(503, "The SQL sandbox is not configured on this server.")
        return self.labs


state = AppState()
