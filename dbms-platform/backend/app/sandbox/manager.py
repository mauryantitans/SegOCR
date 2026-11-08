"""Isolated, resettable MySQL lab environments for learners.

Isolation model (see docs/ARCHITECTURE.md#sandbox):

* Learner SQL runs on a dedicated *sandbox* MySQL server, never the
  application database server.
* Every lab is a private schema ``lab_<hash>`` plus a MySQL account with the
  same name that has privileges **only on that schema** and per-hour resource
  limits. Datasets are loaded through that account, so every object in the lab
  is owned by it.
* A trusted admin account (``sandbox_admin``) only creates/drops lab schemas
  and accounts, kills runaway statements and reads metadata. Learner SQL never
  runs as the admin.
* Every statement passes through :mod:`guard` first and runs under a
  watchdog (KILL QUERY after ``statement_timeout_ms``) plus
  ``max_execution_time`` for SELECTs. Result sets are truncated to
  ``max_result_rows``.
"""
from __future__ import annotations

import datetime as dt
import decimal
import hashlib
import hmac
import logging
import re
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pymysql
import pymysql.cursors
from pymysql.constants import CLIENT

from ..config import Settings
from . import guard
from .errors import explain_error
from .splitter import SplitError, split_sql

log = logging.getLogger("sandbox")

LAB_PRIVILEGES = (
    "SELECT, INSERT, UPDATE, DELETE, CREATE, DROP, ALTER, INDEX, REFERENCES, "
    "CREATE TEMPORARY TABLES, LOCK TABLES, CREATE VIEW, SHOW VIEW, CREATE ROUTINE, "
    "ALTER ROUTINE, EXECUTE, TRIGGER, EVENT"
)
META_SCHEMA = "sandbox_meta"


class SandboxError(Exception):
    """Errors that should be reported to the learner as-is (not MySQL errors)."""


class SessionBusy(SandboxError):
    pass


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------
@dataclass
class StatementResult:
    index: int
    sql: str
    kind: str
    verb: str
    status: str  # ok | error | rejected | blocked
    columns: list[str] = field(default_factory=list)
    rows: list[list] = field(default_factory=list)
    row_count: int | None = None        # rows returned (for result sets)
    affected_rows: int | None = None    # rows changed (DML)
    truncated: bool = False
    info: str | None = None
    warnings: list[dict] = field(default_factory=list)
    error: dict | None = None
    elapsed_ms: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RunResult:
    lab: str
    schema: str
    dataset: str
    statements: list[StatementResult]
    elapsed_ms: float
    server_version: str
    notice: str | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["statements"] = [s.to_dict() for s in self.statements]
        return d


def to_jsonable(v):
    if v is None or isinstance(v, (int, str, bool)):
        return v
    if isinstance(v, float):
        return v if v == v and v not in (float("inf"), float("-inf")) else str(v)
    if isinstance(v, decimal.Decimal):
        return str(v)
    if isinstance(v, dt.datetime):
        s = v.strftime("%Y-%m-%d %H:%M:%S")
        return s + (f".{v.microsecond:06d}" if v.microsecond else "")
    if isinstance(v, dt.date):
        return v.isoformat()
    if isinstance(v, dt.timedelta):
        total = int(v.total_seconds())
        sign = "-" if total < 0 else ""
        total = abs(total)
        return f"{sign}{total // 3600:02d}:{(total % 3600) // 60:02d}:{total % 60:02d}"
    if isinstance(v, (bytes, bytearray)):
        b = bytes(v)
        try:
            s = b.decode("utf-8")
            if s.isprintable():
                return s
        except UnicodeDecodeError:
            pass
        return "0x" + b.hex().upper()
    if isinstance(v, set):
        return ",".join(sorted(v))
    return str(v)


# ---------------------------------------------------------------------------
# A single MySQL connection owned by one lab session (main, A, B, ...)
# ---------------------------------------------------------------------------
class LabSession:
    def __init__(self, manager: "LabManager", schema: str, name: str, statement_timeout_ms: int):
        self.manager = manager
        self.schema = schema
        self.name = name
        self.statement_timeout_ms = statement_timeout_ms
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"lab-{schema[-6:]}-{name}")
        self.pending: list[tuple[int, Future]] = []
        self.last_used = time.monotonic()
        self.conn: pymysql.connections.Connection | None = None
        self._lock = threading.Lock()

    # -- connection ---------------------------------------------------------
    def _connect(self):
        s = self.manager.settings
        self.conn = pymysql.connect(
            host=s.sandbox_host, port=s.sandbox_port, user=self.schema,
            password=self.manager.lab_password(self.schema), database=self.schema,
            charset="utf8mb4", autocommit=True, local_infile=False,
            cursorclass=pymysql.cursors.SSCursor, connect_timeout=10,
        )
        with self.conn.cursor() as cur:
            cur.execute(f"SET SESSION max_execution_time = {int(self.statement_timeout_ms)}")
        self.thread_id = self.conn.thread_id()

    def ensure_connected(self):
        if self.conn is None or not self.conn.open:
            self._connect()

    def close(self):
        try:
            if self.conn is not None and self.conn.open:
                self.conn.close()
        except Exception:  # pragma: no cover - best effort
            pass
        self.conn = None
        self.executor.shutdown(wait=False, cancel_futures=True)

    @property
    def busy(self) -> bool:
        return any(not f.done() for _, f in self.pending)

    # -- execution ----------------------------------------------------------
    def submit(self, index: int, sql: str, cls: guard.Classification) -> Future:
        self.last_used = time.monotonic()
        fut = self.executor.submit(self._execute, index, sql, cls)
        self.pending.append((index, fut))
        return fut

    def collect_finished(self) -> list[StatementResult]:
        done, still = [], []
        for idx, f in self.pending:
            (done if f.done() else still).append((idx, f))
        self.pending = still
        return [f.result() for _, f in done]

    def _execute(self, index: int, sql: str, cls: guard.Classification) -> StatementResult:
        res = StatementResult(index=index, sql=sql, kind=cls.kind, verb=cls.verb, status="ok")
        timer = None
        t0 = time.perf_counter()
        try:
            self.ensure_connected()
            timer = threading.Timer(self.statement_timeout_ms / 1000 + 0.5, self.manager.kill_query, args=(self.thread_id,))
            timer.daemon = True
            timer.start()
            cur = self.conn.cursor()
            try:
                cur.execute(sql)
                self._read_results(cur, res)
            finally:
                cur.close()
            if cur.warning_count:
                res.warnings = self._warnings()
        except pymysql.err.MySQLError as e:
            code = e.args[0] if e.args else None
            msg = e.args[1] if len(e.args) > 1 else str(e)
            msg = msg.replace(f"{self.schema}.", "")  # lab schema names are noise for learners
            res.status = "error"
            res.error = {"code": code, "message": msg, "hint": explain_error(code, msg, self.statement_timeout_ms)}
            if isinstance(e, (pymysql.err.OperationalError, pymysql.err.InterfaceError)) and code in (2006, 2013, 2014, 0):
                self.close_conn_only()
        finally:
            if timer:
                timer.cancel()
            res.elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
            self.last_used = time.monotonic()
        return res

    def close_conn_only(self):
        try:
            if self.conn:
                self.conn.close()
        except Exception:
            pass
        self.conn = None

    def _read_results(self, cur, res: StatementResult):
        limit = self.manager.settings.max_result_rows
        first = True
        while True:
            if cur.description:
                if first:
                    res.columns = [d[0] for d in cur.description]
                    rows = cur.fetchmany(limit + 1)
                    res.truncated = len(rows) > limit
                    res.rows = [[to_jsonable(v) for v in r] for r in rows[:limit]]
                    count = len(rows[:limit])
                    if res.truncated:  # drain the rest to count it
                        for _ in cur:
                            count += 1
                        count += 1
                    res.row_count = count
                    first = False
                else:
                    for _ in cur:
                        pass
            else:
                if first or res.affected_rows is None:
                    res.affected_rows = max(cur.rowcount, 0) if cur.rowcount is not None else None
                msg = getattr(cur._result, "message", None) if getattr(cur, "_result", None) else None
                if msg:
                    text = msg.decode("utf-8", "replace") if isinstance(msg, bytes) else str(msg)
                    res.info = re.sub(r"^[^A-Za-z]+", "", text).strip() or None
            if not cur.nextset():
                break

    def _warnings(self) -> list[dict]:
        try:
            with self.conn.cursor() as c2:
                c2.execute("SHOW WARNINGS LIMIT 10")
                return [{"level": r[0], "code": r[1], "message": r[2]} for r in c2.fetchall()]
        except pymysql.err.MySQLError:
            return []


# ---------------------------------------------------------------------------
# The lab manager
# ---------------------------------------------------------------------------
class LabManager:
    def __init__(self, settings: Settings, datasets_dir: Path):
        self.settings = settings
        self.datasets_dir = datasets_dir
        self._sessions: dict[tuple[str, str], LabSession] = {}
        self._lab_locks: dict[str, threading.Lock] = {}
        self._lock = threading.Lock()
        self._admin_lock = threading.Lock()
        self._admin: pymysql.connections.Connection | None = None
        self._server_version: str | None = None
        self._meta_ready = False

    # -- naming ---------------------------------------------------------------
    @staticmethod
    def schema_for(owner: str, lab_key: str) -> str:
        return "lab_" + hashlib.sha256(f"{owner}|{lab_key}".encode()).hexdigest()[:20]

    def lab_password(self, schema: str) -> str:
        h = hmac.new(self.settings.secret_key.encode(), schema.encode(), hashlib.sha256).hexdigest()
        return f"Lb{h[:30]}#9"

    def datasets(self) -> list[str]:
        return sorted(p.stem for p in self.datasets_dir.glob("*.sql"))

    # -- admin connection -------------------------------------------------------
    def _admin_conn(self):
        if self._admin is None or not self._admin.open:
            s = self.settings
            self._admin = pymysql.connect(
                host=s.sandbox_host, port=s.sandbox_port, user=s.sandbox_admin_user,
                password=s.sandbox_admin_password, charset="utf8mb4", autocommit=True, connect_timeout=10,
            )
            with self._admin.cursor() as c:
                c.execute("SET SESSION information_schema_stats_expiry = 0")
                c.execute("SET SESSION lock_wait_timeout = 20")  # never let a reset hang on a metadata lock
        else:
            try:
                self._admin.ping(reconnect=False)
            except pymysql.err.MySQLError:
                self._admin = None
                return self._admin_conn()
        return self._admin

    def admin(self, sql: str, args=None, fetch: bool = False):
        with self._admin_lock:
            conn = self._admin_conn()
            with conn.cursor() as c:
                c.execute(sql, args)
                return c.fetchall() if fetch else None

    def server_version(self) -> str:
        if self._server_version is None:
            self._server_version = self.admin("SELECT VERSION()", fetch=True)[0][0]
        return self._server_version

    def kill_query(self, thread_id: int):
        try:
            self.admin(f"KILL QUERY {int(thread_id)}")
            log.warning("killed runaway statement on thread %s", thread_id)
        except pymysql.err.MySQLError as e:  # already finished
            log.debug("kill_query(%s) failed: %s", thread_id, e)

    def _ensure_meta(self):
        if self._meta_ready:
            return
        self.admin(f"CREATE DATABASE IF NOT EXISTS {META_SCHEMA}")
        self.admin(f"""CREATE TABLE IF NOT EXISTS {META_SCHEMA}.labs (
            schema_name VARCHAR(64) PRIMARY KEY, owner VARCHAR(100) NOT NULL, lab_key VARCHAR(200) NOT NULL,
            dataset VARCHAR(40) NOT NULL, created_at DATETIME NOT NULL, last_used_at DATETIME NOT NULL)""")
        self._meta_ready = True

    def _lab_lock(self, schema: str) -> threading.Lock:
        with self._lock:
            return self._lab_locks.setdefault(schema, threading.Lock())

    # -- lifecycle ------------------------------------------------------------------
    def lab_info(self, owner: str, lab_key: str) -> dict | None:
        self._ensure_meta()
        schema = self.schema_for(owner, lab_key)
        rows = self.admin(f"SELECT dataset, created_at, last_used_at FROM {META_SCHEMA}.labs WHERE schema_name=%s",
                          (schema,), fetch=True)
        if not rows:
            return None
        return {"schema": schema, "dataset": rows[0][0], "created_at": rows[0][1], "last_used_at": rows[0][2]}

    def ensure_lab(self, owner: str, lab_key: str, dataset: str | None = None) -> dict:
        """Create the lab if needed. If ``dataset`` differs from the lab's current dataset, reset it."""
        info = self.lab_info(owner, lab_key)
        if info and (dataset is None or dataset == info["dataset"]):
            self.admin(f"UPDATE {META_SCHEMA}.labs SET last_used_at=NOW() WHERE schema_name=%s", (info["schema"],))
            return info
        return self.reset_lab(owner, lab_key, dataset or (info["dataset"] if info else "empty"))

    def reset_lab(self, owner: str, lab_key: str, dataset: str, setup_sql: str | None = None) -> dict:
        if dataset not in self.datasets():
            raise SandboxError(f"Unknown dataset '{dataset}'. Available: {', '.join(self.datasets())}")
        self._ensure_meta()
        schema = self.schema_for(owner, lab_key)
        s = self.settings
        with self._lab_lock(schema):
            self._close_sessions(schema)
            self._kill_stray_connections(schema)
            account = f"'{schema}'@'{s.sandbox_account_host}'"
            self.admin(f"DROP DATABASE IF EXISTS `{schema}`")
            self.admin(f"CREATE DATABASE `{schema}` CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci")
            self.admin(
                f"CREATE USER IF NOT EXISTS {account} IDENTIFIED BY %s WITH MAX_QUERIES_PER_HOUR {s.lab_max_queries_per_hour} "
                f"MAX_USER_CONNECTIONS {s.lab_max_user_connections}", (self.lab_password(schema),))
            # In GRANT, '_' in a database name is a wildcard: escape it so the grant covers exactly this schema.
            self.admin(f"GRANT {LAB_PRIVILEGES} ON `{schema.replace('_', chr(92) + '_')}`.* TO {account}")
            self._load_dataset(schema, dataset, setup_sql)
            self.admin(
                f"INSERT INTO {META_SCHEMA}.labs VALUES (%s,%s,%s,%s,NOW(),NOW()) "
                f"ON DUPLICATE KEY UPDATE dataset=VALUES(dataset), last_used_at=NOW(), created_at=NOW()",
                (schema, owner[:100], lab_key[:200], dataset))
        return {"schema": schema, "dataset": dataset}

    def _load_dataset(self, schema: str, dataset: str, setup_sql: str | None):
        script = (self.datasets_dir / f"{dataset}.sql").read_text()
        if setup_sql:
            script += "\n" + setup_sql
        if not script.strip() or all(l.strip().startswith("--") or not l.strip() for l in script.splitlines()):
            return
        s = self.settings
        conn = pymysql.connect(host=s.sandbox_host, port=s.sandbox_port, user=schema, password=self.lab_password(schema),
                               database=schema, charset="utf8mb4", autocommit=True,
                               client_flag=CLIENT.MULTI_STATEMENTS, connect_timeout=10)
        try:
            with conn.cursor() as c:
                c.execute(script)
                while c.nextset():
                    pass
        finally:
            conn.close()

    def _kill_stray_connections(self, schema: str):
        """Kill connections of the lab account not owned by this process (e.g. left by a crashed or
        restarted backend). An open transaction there would otherwise block DROP DATABASE on a metadata lock."""
        rows = self.admin("SELECT id FROM information_schema.processlist WHERE user = %s", (schema,), fetch=True)
        for (tid,) in rows:
            self.kill_connection(tid)

    def drop_lab(self, schema: str):
        with self._lab_lock(schema):
            self._close_sessions(schema)
            self._kill_stray_connections(schema)
            self.admin(f"DROP DATABASE IF EXISTS `{schema}`")
            self.admin(f"DROP USER IF EXISTS '{schema}'@'{self.settings.sandbox_account_host}'")
            self.admin(f"DELETE FROM {META_SCHEMA}.labs WHERE schema_name=%s", (schema,))

    def _close_sessions(self, schema: str):
        with self._lock:
            keys = [k for k in self._sessions if k[0] == schema]
            sessions = [self._sessions.pop(k) for k in keys]
        for sess in sessions:
            if sess.busy and getattr(sess, "thread_id", None):
                self.kill_connection(sess.thread_id)
            sess.close()

    def kill_connection(self, thread_id: int):
        try:
            self.admin(f"KILL {int(thread_id)}")
        except pymysql.err.MySQLError:
            pass

    def session(self, schema: str, name: str = "main", timeout_ms: int | None = None) -> LabSession:
        with self._lock:
            sess = self._sessions.get((schema, name))
            if sess is None:
                sess = LabSession(self, schema, name, timeout_ms or self.settings.statement_timeout_ms)
                self._sessions[(schema, name)] = sess
            return sess

    # -- running SQL ---------------------------------------------------------------------
    def prepare(self, sql: str, schema: str) -> list[tuple[str, guard.Classification]]:
        s = self.settings
        if len(sql) > s.max_script_chars:
            raise SandboxError(f"Script is too long ({len(sql):,} characters; limit {s.max_script_chars:,}).")
        try:
            stmts = split_sql(sql)
        except SplitError as e:
            raise SandboxError(str(e)) from e
        if not stmts:
            raise SandboxError("Nothing to run: the editor contains no SQL statements.")
        if len(stmts) > s.max_statements_per_run:
            raise SandboxError(f"Too many statements ({len(stmts)}); the limit per run is {s.max_statements_per_run}.")
        return [(st.sql, guard.classify(st.sql, schema)) for st in stmts]

    def run(self, owner: str, lab_key: str, sql: str, dataset: str | None = None,
            session_name: str = "main", wait_s: float | None = None, stop_on_error: bool = True,
            timeout_ms: int | None = None) -> RunResult:
        info = self.ensure_lab(owner, lab_key, dataset)
        schema = info["schema"]
        prepared = self.prepare(sql, schema)
        sess = self.session(schema, session_name, timeout_ms)
        if sess.busy:
            raise SessionBusy(f"Session {session_name} is still waiting for its previous statement "
                              "(probably blocked on a lock held by another session).")
        t0 = time.perf_counter()
        results: list[StatementResult] = []
        for i, (stmt, cls) in enumerate(prepared):
            if not cls.allowed:
                results.append(StatementResult(index=i, sql=stmt, kind=cls.kind, verb=cls.verb, status="rejected",
                                               error={"code": None, "message": cls.reason, "hint": None}))
                if stop_on_error:
                    break
                continue
            fut = sess.submit(i, stmt, cls)
            try:
                res = fut.result(timeout=wait_s if wait_s is not None else sess.statement_timeout_ms / 1000 + 5)
            except FutureTimeout:
                results.append(StatementResult(index=i, sql=stmt, kind=cls.kind, verb=cls.verb, status="blocked"))
                break
            sess.pending = [(j, f) for j, f in sess.pending if f is not fut]
            results.append(res)
            if res.status == "error" and stop_on_error:
                break
        notice = None
        if any(r.kind in (guard.WRITE, guard.DDL, guard.ROUTINE) and r.status == "ok" for r in results):
            notice = self._quota_notice(schema)
        return RunResult(lab=lab_key, schema=schema, dataset=info["dataset"], statements=results,
                         elapsed_ms=round((time.perf_counter() - t0) * 1000, 2),
                         server_version=self.server_version(), notice=notice)

    def poll(self, owner: str, lab_key: str, session_name: str) -> list[StatementResult]:
        schema = self.schema_for(owner, lab_key)
        with self._lock:
            sess = self._sessions.get((schema, session_name))
        return sess.collect_finished() if sess else []

    def session_busy(self, owner: str, lab_key: str, session_name: str) -> bool:
        schema = self.schema_for(owner, lab_key)
        with self._lock:
            sess = self._sessions.get((schema, session_name))
        return bool(sess and sess.busy)

    def inspect_locks(self, owner: str, lab_key: str) -> dict:
        """Snapshot of InnoDB transactions and locks held/awaited by this lab's sessions.

        Runs fixed queries as the admin account (learners cannot read performance_schema),
        filtered to the lab's own connections, and labels each row with the session name.
        """
        schema = self.schema_for(owner, lab_key)
        with self._lock:
            sessions = {s.thread_id: name for (sch, name), s in self._sessions.items()
                        if sch == schema and getattr(s, "thread_id", None)}
        if not sessions:
            return {"transactions": [], "locks": []}
        ids = ",".join(str(int(t)) for t in sessions)
        trx = self.admin(
            "SELECT trx_mysql_thread_id, trx_id, trx_state, trx_isolation_level, trx_rows_locked, trx_rows_modified, "
            "trx_lock_structs, trx_started FROM information_schema.innodb_trx "
            f"WHERE trx_mysql_thread_id IN ({ids}) ORDER BY trx_started", fetch=True)
        locks = self.admin(
            "SELECT t.PROCESSLIST_ID, l.OBJECT_NAME, l.INDEX_NAME, l.LOCK_TYPE, l.LOCK_MODE, l.LOCK_STATUS, l.LOCK_DATA "
            "FROM performance_schema.data_locks l JOIN performance_schema.threads t ON t.THREAD_ID = l.THREAD_ID "
            f"WHERE t.PROCESSLIST_ID IN ({ids}) AND l.OBJECT_SCHEMA = %s "
            "ORDER BY t.PROCESSLIST_ID, l.OBJECT_NAME, l.LOCK_TYPE DESC, l.INDEX_NAME, l.LOCK_DATA",
            (schema,), fetch=True)
        return {
            "transactions": [{"session": sessions.get(r[0], "?"), "trxId": str(r[1]), "state": r[2], "isolation": r[3],
                              "rowsLocked": r[4], "rowsModified": r[5], "lockStructs": r[6]} for r in trx],
            "locks": [{"session": sessions.get(r[0], "?"), "table": r[1], "index": r[2], "lockType": r[3],
                       "lockMode": r[4], "status": r[5], "lockData": r[6]} for r in locks],
        }

    def _quota_notice(self, schema: str) -> str | None:
        rows = self.admin("SELECT COALESCE(SUM(data_length + index_length), 0) FROM information_schema.tables "
                          "WHERE table_schema = %s", (schema,), fetch=True)
        mb = float(rows[0][0]) / 1024 / 1024
        if mb > self.settings.lab_quota_mb:
            return (f"This lab now uses {mb:.0f} MB, above the {self.settings.lab_quota_mb} MB sandbox quota. "
                    "Please reset the lab or drop large tables.")
        return None

    # -- introspection ------------------------------------------------------------------------
    def schema_overview(self, owner: str, lab_key: str, dataset: str | None = None) -> dict:
        info = self.ensure_lab(owner, lab_key, dataset)
        schema = info["schema"]
        tables = self.admin(
            "SELECT table_name, table_type, table_rows FROM information_schema.tables WHERE table_schema=%s ORDER BY table_name",
            (schema,), fetch=True)
        cols = self.admin(
            "SELECT table_name, column_name, column_type, is_nullable, column_key, column_default, extra "
            "FROM information_schema.columns WHERE table_schema=%s ORDER BY table_name, ordinal_position",
            (schema,), fetch=True)
        idx = self.admin(
            "SELECT table_name, index_name, non_unique, GROUP_CONCAT(column_name ORDER BY seq_in_index) "
            "FROM information_schema.statistics WHERE table_schema=%s GROUP BY table_name, index_name, non_unique "
            "ORDER BY table_name, index_name", (schema,), fetch=True)
        fks = self.admin(
            "SELECT table_name, constraint_name, GROUP_CONCAT(column_name ORDER BY ordinal_position), referenced_table_name, "
            "GROUP_CONCAT(referenced_column_name ORDER BY ordinal_position) FROM information_schema.key_column_usage "
            "WHERE table_schema=%s AND referenced_table_name IS NOT NULL GROUP BY table_name, constraint_name, referenced_table_name",
            (schema,), fetch=True)
        routines = self.admin(
            "SELECT routine_name, routine_type FROM information_schema.routines WHERE routine_schema=%s ORDER BY routine_name",
            (schema,), fetch=True)
        out = {}
        for name, ttype, nrows in tables:
            out[name] = {"name": name, "type": "view" if ttype == "VIEW" else "table", "approxRows": nrows,
                         "columns": [], "indexes": [], "foreignKeys": []}
        for t, c, ctype, nullable, key, default, extra in cols:
            if t in out:
                out[t]["columns"].append({"name": c, "type": ctype, "nullable": nullable == "YES", "key": key,
                                          "default": to_jsonable(default), "extra": extra})
        for t, name, non_unique, columns in idx:
            if t in out:
                out[t]["indexes"].append({"name": name, "unique": not non_unique, "columns": columns.split(",")})
        for t, name, columns, rt, rc in fks:
            if t in out:
                out[t]["foreignKeys"].append({"name": name, "columns": columns.split(","), "refTable": rt,
                                              "refColumns": rc.split(",")})
        return {"schema": schema, "dataset": info["dataset"], "tables": list(out.values()),
                "routines": [{"name": n, "type": t} for n, t in routines]}

    # -- housekeeping ----------------------------------------------------------------------------
    def cleanup(self) -> dict:
        """Disconnect idle sessions and drop labs unused for ``lab_ttl_hours``."""
        now = time.monotonic()
        idle = []
        with self._lock:
            for key, sess in list(self._sessions.items()):
                if not sess.busy and now - sess.last_used > self.settings.lab_idle_disconnect_s:
                    idle.append(self._sessions.pop(key))
        for sess in idle:
            sess.close()
        self._ensure_meta()
        stale = self.admin(f"SELECT schema_name FROM {META_SCHEMA}.labs WHERE last_used_at < NOW() - INTERVAL %s HOUR",
                           (self.settings.lab_ttl_hours,), fetch=True)
        for (schema,) in stale:
            self.drop_lab(schema)
        return {"disconnected": len(idle), "dropped": len(stale)}

    def shutdown(self):
        with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
        for s in sessions:
            s.close()
        if self._admin is not None:
            try:
                self._admin.close()
            except Exception:
                pass
