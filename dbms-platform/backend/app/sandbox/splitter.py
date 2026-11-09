"""Split a SQL script into individual statements, the way the mysql client does.

Why not send the whole script with CLIENT.MULTI_STATEMENTS? Because the
sandbox needs per-statement guard checks, per-statement timing/results, and
per-statement error attribution ("statement 3 failed").

Handles:
  * single/double/backtick quoted strings (with doubled-quote and backslash escapes)
  * -- , # and /* */ comments (comments are kept inside the statement text)
  * the mysql-client DELIMITER directive (learners paste CLI scripts)
  * stored program bodies (CREATE PROCEDURE/FUNCTION/TRIGGER/EVENT ... BEGIN ... END)
    without needing DELIMITER, by tracking compound-statement depth.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_$]*")
_ROUTINE_HEAD = re.compile(
    r"^\s*CREATE\s+(?:OR\s+REPLACE\s+)?(?:DEFINER\s*=\s*\S+\s+)?(?:AGGREGATE\s+)?"
    r"(PROCEDURE|FUNCTION|TRIGGER|EVENT)\b",
    re.IGNORECASE,
)
_BLOCK_OPENERS = {"BEGIN", "CASE", "LOOP", "REPEAT", "WHILE", "IF"}
# Tokens after which a new statement starts inside a compound statement.
_STATEMENT_START = {";", "BEGIN", "THEN", "ELSE", "DO", "LOOP", "REPEAT"}


@dataclass
class Statement:
    sql: str
    line: int  # 1-based line where the statement starts


class SplitError(ValueError):
    pass


def _strip_comments_and_space(s: str) -> str:
    """True if the statement contains only whitespace/comments."""
    out, i, n = [], 0, len(s)
    while i < n:
        if s.startswith("--", i) and (i + 2 >= n or s[i + 2] in " \t\r\n"):
            j = s.find("\n", i)
            i = n if j < 0 else j + 1
        elif s[i] == "#":
            j = s.find("\n", i)
            i = n if j < 0 else j + 1
        elif s.startswith("/*", i) and not s.startswith("/*!", i):
            j = s.find("*/", i + 2)
            i = n if j < 0 else j + 2
        else:
            out.append(s[i])
            i += 1
    return "".join(out).strip()


def split_sql(script: str) -> list[Statement]:
    stmts: list[Statement] = []
    delimiter = ";"
    buf: list[str] = []
    i, n = 0, len(script)
    line = 1
    stmt_line = 1
    depth = 0           # compound-statement depth inside a stored program
    in_routine = False  # current statement started with CREATE PROCEDURE/...
    prev_word = ""      # previous keyword (to recognise END IF / END LOOP ...)
    skip_next_opener = False

    def flush():
        nonlocal buf, depth, in_routine, prev_word, skip_next_opener, stmt_line
        text = "".join(buf).strip()
        if _strip_comments_and_space(text):
            stmts.append(Statement(text, stmt_line))
        buf = []
        depth = 0
        in_routine = False
        prev_word = ""
        skip_next_opener = False

    at_line_start = True
    while i < n:
        c = script[i]

        # DELIMITER directive: only recognised at the start of a line, outside a statement body.
        if at_line_start:
            m = re.match(r"[ \t]*DELIMITER[ \t]+(\S+)[ \t]*(?:\r?\n|$)", script[i:], re.IGNORECASE)
            if m and not _strip_comments_and_space("".join(buf)):
                buf = []
                delimiter = m.group(1)
                i += m.end()
                line += 1
                stmt_line = line
                continue
        at_line_start = False

        if not _strip_comments_and_space("".join(buf)) and c not in " \t\r\n":
            stmt_line = line

        # quoted strings / identifiers
        if c in ("'", '"', "`"):
            q = c
            j = i + 1
            while j < n:
                if script[j] == "\\" and q != "`":
                    j += 2
                    continue
                if script[j] == q:
                    if j + 1 < n and script[j + 1] == q:
                        j += 2
                        continue
                    break
                j += 1
            if j >= n:
                raise SplitError(f"Unterminated {q} quote starting on line {line}")
            chunk = script[i:j + 1]
            buf.append(chunk)
            line += chunk.count("\n")
            i = j + 1
            prev_word = ""
            continue

        # comments
        if script.startswith("--", i) and (i + 2 >= n or script[i + 2] in " \t\r\n"):
            j = script.find("\n", i)
            j = n if j < 0 else j
            buf.append(script[i:j])
            i = j
            continue
        if c == "#":
            j = script.find("\n", i)
            j = n if j < 0 else j
            buf.append(script[i:j])
            i = j
            continue
        if script.startswith("/*", i):
            j = script.find("*/", i + 2)
            if j < 0:
                raise SplitError(f"Unterminated /* comment starting on line {line}")
            chunk = script[i:j + 2]
            buf.append(chunk)
            line += chunk.count("\n")
            i = j + 2
            continue

        # custom delimiter (DELIMITER mode disables depth tracking, like the mysql client)
        if delimiter != ";" and script.startswith(delimiter, i):
            flush()
            i += len(delimiter)
            continue

        if delimiter == ";" and c == ";":
            if in_routine and depth > 0:
                buf.append(c)
                i += 1
                prev_word = ";"
                skip_next_opener = False  # "END; IF ..." starts a new IF statement, it is not "END IF"
                continue
            flush()
            i += 1
            continue

        m = _WORD.match(script, i)
        if m and (i == 0 or not (script[i - 1].isalnum() or script[i - 1] in "_$@.")):
            word = m.group(0)
            buf.append(word)
            i = m.end()
            if delimiter == ";":
                if (not in_routine and word.upper() in ("PROCEDURE", "FUNCTION", "TRIGGER", "EVENT")
                        and _ROUTINE_HEAD.match("".join(buf))):
                    in_routine = True
                if in_routine:
                    up = word.upper()
                    if up == "END":
                        depth = max(0, depth - 1)
                        skip_next_opener = True
                    elif skip_next_opener and up in ("IF", "CASE", "LOOP", "REPEAT", "WHILE"):
                        skip_next_opener = False
                    elif up in _BLOCK_OPENERS:
                        skip_next_opener = False
                        if up == "BEGIN" or depth > 0:
                            # IF opens a block only as a statement (IF cond THEN ... END IF), i.e. at the start of a
                            # statement. Elsewhere it is the IF() function or part of DROP/CREATE ... IF [NOT] EXISTS.
                            if up == "IF" and prev_word not in _STATEMENT_START:
                                prev_word = up
                                continue
                            if up == "BEGIN" and re.match(r"\s*(WORK\b|;)", script[i:], re.IGNORECASE) and depth == 0:
                                prev_word = up
                                continue
                            depth += 1
                    else:
                        skip_next_opener = False
                    prev_word = up
            continue

        if c == "\n":
            line += 1
            at_line_start = True
        buf.append(c)
        i += 1

    flush()
    return stmts


_LEADING_COMMENTS = re.compile(r"^(\s+|--[^\n]*(\n|$)|#[^\n]*(\n|$)|/\*(?!!).*?\*/)+", re.DOTALL)


def strip_leading_comments(sql: str) -> str:
    return _LEADING_COMMENTS.sub("", sql, count=1)
