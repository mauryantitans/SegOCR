"""Statement guard: classify learner statements and reject unsafe ones.

This is defence in depth and a UX layer (friendly explanations). The real
security boundary is MySQL's privilege system: every learner statement runs as
a per-lab MySQL account that only has privileges on its own lab schema and has
per-hour resource limits. A statement that slips past this guard (for example
via PREPARE of a string) still cannot touch anything outside the lab schema.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .splitter import strip_leading_comments

READ = "read"
WRITE = "write"
DDL = "ddl"
TCL = "tcl"
SESSION = "session"
ROUTINE = "routine"
ADMIN = "admin"
OTHER = "other"


@dataclass
class Classification:
    kind: str
    verb: str
    allowed: bool
    reason: str | None = None


_DENY_VERBS = {
    "GRANT": "Account privileges cannot be changed from the sandbox. The security lessons show GRANT/REVOKE as read-only examples.",
    "REVOKE": "Account privileges cannot be changed from the sandbox.",
    "SHUTDOWN": "Server administration is disabled in the sandbox.",
    "KILL": "KILL is disabled in the sandbox; use the Reset button to abandon a stuck session.",
    "INSTALL": "Plugins/components cannot be installed from the sandbox.",
    "UNINSTALL": "Plugins/components cannot be removed from the sandbox.",
    "FLUSH": "FLUSH is a server administration statement and is disabled in the sandbox.",
    "RESET": "RESET is a server administration statement and is disabled in the sandbox.",
    "PURGE": "Binary log administration is disabled in the sandbox.",
    "CHANGE": "Replication configuration is disabled in the sandbox.",
    "CLONE": "CLONE is disabled in the sandbox.",
    "LOAD": "LOAD DATA / LOAD XML read server or client files and are disabled in the sandbox. Use INSERT statements instead.",
    "IMPORT": "IMPORT TABLE is disabled in the sandbox.",
    "CACHE": "CACHE INDEX is disabled in the sandbox.",
    "BINLOG": "BINLOG statements are disabled in the sandbox.",
    "RESTART": "Server administration is disabled in the sandbox.",
}

_READ_VERBS = {"SELECT", "WITH", "TABLE", "VALUES", "SHOW", "DESCRIBE", "DESC", "EXPLAIN", "HELP", "(" }
_WRITE_VERBS = {"INSERT", "UPDATE", "DELETE", "REPLACE"}
_DDL_VERBS = {"CREATE", "ALTER", "DROP", "TRUNCATE", "RENAME"}
_TCL_VERBS = {"START", "BEGIN", "COMMIT", "ROLLBACK", "SAVEPOINT", "RELEASE", "XA"}
_MAINT_VERBS = {"ANALYZE", "OPTIMIZE", "CHECK", "CHECKSUM", "REPAIR"}

_FORBIDDEN_PATTERNS = [
    (re.compile(r"\bINTO\s+(OUTFILE|DUMPFILE)\b", re.I), "SELECT ... INTO OUTFILE/DUMPFILE writes files on the server and is disabled."),
    (re.compile(r"\bLOAD_FILE\s*\(", re.I), "LOAD_FILE() reads server files and is disabled."),
    (re.compile(r"\bSLEEP\s*\(\s*([3-9]\d|\d{3,})", re.I), "SLEEP() longer than 29 seconds is not allowed in the sandbox."),
]

_DDL_DENY = re.compile(
    r"^(CREATE|ALTER|DROP)\s+(DATABASE|SCHEMA|USER|ROLE|SERVER|TABLESPACE|LOGFILE|RESOURCE\s+GROUP|SPATIAL\s+REFERENCE|UNDO|INSTANCE)\b",
    re.I,
)
_RENAME_USER = re.compile(r"^RENAME\s+USER\b", re.I)


def _verb(sql: str) -> str:
    s = strip_leading_comments(sql).lstrip()
    if s.startswith("("):
        return "("
    m = re.match(r"[A-Za-z_]+", s)
    return m.group(0).upper() if m else ""


def classify(sql: str, lab_schema: str) -> Classification:
    body = strip_leading_comments(sql).strip()
    verb = _verb(body)

    for pat, reason in _FORBIDDEN_PATTERNS:
        if pat.search(body):
            return Classification(ADMIN, verb, False, reason)

    if verb in _DENY_VERBS:
        return Classification(ADMIN, verb, False, _DENY_VERBS[verb])

    if verb == "USE":
        target = body[3:].strip().strip(";").strip("`").strip()
        if target.lower() != lab_schema.lower():
            return Classification(SESSION, verb, False,
                                  f"Each lab runs in its own private schema (`{lab_schema}`). Switching databases is disabled; "
                                  "your tables are already in the current schema.")
        return Classification(SESSION, verb, True)

    if verb == "SET":
        up = body.upper()
        if re.match(r"SET\s+(PASSWORD|DEFAULT\s+ROLE|ROLE|RESOURCE\s+GROUP)\b", up):
            return Classification(ADMIN, verb, False, "Account settings cannot be changed from the sandbox.")
        # Only reject assignments TO global/persisted variables; reading @@GLOBAL.x on the right-hand side is fine.
        targets = re.split(r",(?![^()]*\))", up[3:])
        if any(re.match(r"\s*(GLOBAL|PERSIST|PERSIST_ONLY)\s", t) or re.match(r"\s*@@(GLOBAL|PERSIST|PERSIST_ONLY)\.", t)
               for t in targets):
            return Classification(ADMIN, verb, False,
                                  "Global/persisted system variables are server-wide and cannot be changed from the sandbox. "
                                  "Use SET SESSION to change a variable for your own session.")
        if re.match(r"SET\s+(SESSION\s+|LOCAL\s+)?TRANSACTION\b", up):
            return Classification(TCL, verb, True)
        return Classification(SESSION, verb, True)

    if verb in _DDL_VERBS:
        if _DDL_DENY.match(body) or _RENAME_USER.match(body):
            what = body.split()[1].upper() if len(body.split()) > 1 else verb
            return Classification(ADMIN, verb, False,
                                  f"{verb} {what} is disabled in the sandbox. Your lab already has its own private schema; "
                                  "use the Reset button for a fresh copy.")
        return Classification(DDL, verb, True)

    if verb in _READ_VERBS:
        return Classification(READ, verb, True)
    if verb in _WRITE_VERBS:
        return Classification(WRITE, verb, True)
    if verb in _TCL_VERBS:
        return Classification(TCL, verb, True)
    if verb in ("CALL", "DO", "PREPARE", "EXECUTE", "DEALLOCATE", "DECLARE", "SIGNAL", "RESIGNAL", "GET"):
        return Classification(ROUTINE, verb, True)
    if verb in ("LOCK", "UNLOCK"):
        return Classification(SESSION, verb, True)
    if verb in _MAINT_VERBS:
        return Classification(OTHER, verb, True)
    if verb == "HANDLER":
        return Classification(READ, verb, True)
    # Unknown verbs (typos such as SELEKT, or rarely used statements) go to MySQL, which reports the real
    # syntax error (1064) or privilege error; the per-lab account remains the security boundary.
    return Classification(OTHER, verb, True)
