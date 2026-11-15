"""Validation and guarded execution of generated SQL (Phase 4, Tasks 3 and 4).

Generated SQL is never executed unless every layer passes:

1. **Text checks** (``check_text``): exactly one statement; no SQL comments;
   it must start with SELECT or WITH; no data/schema-changing or
   database-level keyword outside string literals (DROP, DELETE, UPDATE,
   ALTER, INSERT, CREATE, REPLACE INTO, ATTACH, DETACH, PRAGMA, VACUUM,
   REINDEX, TRUNCATE, GRANT, ...).
2. **SQLite authorizer** (``_authorizer``): while the statement is prepared
   and run, SQLite reports every action. Only SELECT, column reads on user
   tables, and functions are allowed; reading the metadata tables or
   ``sqlite_master`` and any write/DDL/PRAGMA/ATTACH is denied, so a statement
   that slipped past the text checks still fails.
3. **Read-only connection**: opened with ``mode=ro`` and ``PRAGMA
   query_only=ON``.
4. **Limits**: at most ``max_rows`` rows are fetched (``truncated`` flags
   more), and a progress handler aborts queries that run longer than
   ``timeout_s``.

Errors are returned as messages (``QueryResult.error``); nothing is invented.
"""

from __future__ import annotations

import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

from app.structured.ingestion import META_TABLES

FORBIDDEN = (
    "drop", "delete", "update", "alter", "insert", "create", "attach", "detach", "pragma", "vacuum",
    "reindex", "truncate", "grant", "revoke", "commit", "rollback", "savepoint", "release", "begin",
    "analyze", "load_extension", "writefile", "readfile", "edit", "fts3_tokenizer",
)
DEFAULT_MAX_ROWS = 200
DEFAULT_TIMEOUT_S = 5.0

_STRING_RE = re.compile(r"'(?:[^']|'')*'")
_IDENT_RE = re.compile(r'"(?:[^"]|"")*"')


class UnsafeSQL(ValueError):
    """The SQL failed validation and was not executed."""


@dataclass
class QueryResult:
    sql: str
    columns: list[str] = field(default_factory=list)
    rows: list[tuple] = field(default_factory=list)
    truncated: bool = False
    error: str | None = None
    elapsed_s: float = 0.0

    @property
    def ok(self) -> bool:
        return self.error is None


def _strip_literals(sql: str) -> str:
    return _IDENT_RE.sub('""', _STRING_RE.sub("''", sql))


def check_text(sql: str) -> str:
    """Return the normalised single statement or raise ``UnsafeSQL``."""

    if not sql or not sql.strip():
        raise UnsafeSQL("No SQL was generated.")
    statement = sql.strip()
    while statement.endswith(";"):
        statement = statement[:-1].rstrip()
    bare = _strip_literals(statement)
    if ";" in bare:
        raise UnsafeSQL("Only a single SQL statement is allowed.")
    if "--" in bare or "/*" in bare:
        raise UnsafeSQL("SQL comments are not allowed.")
    if not re.match(r"^\s*(select|with)\b", bare, re.I):
        raise UnsafeSQL("Only SELECT queries (optionally starting with WITH) are allowed.")
    lowered = bare.lower()
    for word in FORBIDDEN:
        if re.search(rf"\b{word}\b", lowered):
            raise UnsafeSQL(f"Forbidden keyword '{word.upper()}': only read-only SELECT queries are allowed.")
    if re.search(r"\breplace\s+into\b|\bor\s+replace\b", lowered):
        raise UnsafeSQL("Forbidden keyword 'REPLACE': only read-only SELECT queries are allowed.")
    return statement


def _authorizer(allowed_tables: set[str]):
    def authorize(action, arg1, arg2, db_name, trigger):
        if action == sqlite3.SQLITE_SELECT:
            return sqlite3.SQLITE_OK
        if action == sqlite3.SQLITE_READ:
            table = (arg1 or "").lower()
            return sqlite3.SQLITE_OK if table in allowed_tables else sqlite3.SQLITE_DENY
        if action == sqlite3.SQLITE_FUNCTION:
            return sqlite3.SQLITE_DENY if (arg2 or "").lower() in FORBIDDEN else sqlite3.SQLITE_OK
        if action == getattr(sqlite3, "SQLITE_RECURSIVE", -1):
            return sqlite3.SQLITE_OK
        return sqlite3.SQLITE_DENY

    return authorize


def _connect_readonly(db_path: Path | str) -> sqlite3.Connection:
    path = Path(db_path)
    if not path.is_file():
        raise FileNotFoundError(f"Structured database not found: {path}")
    conn = sqlite3.connect(f"file:{path.resolve().as_posix()}?mode=ro", uri=True)
    conn.execute("PRAGMA query_only = ON")
    return conn


def _user_tables(conn: sqlite3.Connection, statement: str = "") -> set[str]:
    names = {r[0].lower() for r in conn.execute("SELECT name FROM sqlite_master WHERE type IN ('table', 'view')")}
    allowed = {n for n in names if n not in META_TABLES and not n.startswith("sqlite_")}
    # Common table expressions defined by the statement itself are readable too
    # (but never under the name of a protected table).
    ctes = {m.group(1).lower() for m in re.finditer(r"(?:\bwith\b(?:\s+recursive)?|,)\s*([A-Za-z_]\w*)\s*(?:\([^)]*\))?\s+as\s*\(",
                                                     _strip_literals(statement), re.I)}
    return allowed | {c for c in ctes if c not in META_TABLES and not c.startswith("sqlite_")}


def validate(sql: str, db_path: Path | str) -> str:
    """Full validation without fetching data: text checks, then prepare under the authorizer."""

    statement = check_text(sql)
    conn = _connect_readonly(db_path)
    try:
        conn.set_authorizer(_authorizer(_user_tables(conn, statement)))
        try:
            conn.execute(f"EXPLAIN {statement}")
        except sqlite3.DatabaseError as error:
            message = str(error)
            if "not authorized" in message or "prohibited" in message:
                raise UnsafeSQL("The query touches something other than reading the loaded tables.") from error
            raise UnsafeSQL(f"Invalid SQL: {message}") from error
    finally:
        conn.close()
    return statement


def execute(sql: str, db_path: Path | str, max_rows: int = DEFAULT_MAX_ROWS, timeout_s: float = DEFAULT_TIMEOUT_S) -> QueryResult:
    """Validate and run ``sql``; never raises for SQL problems (see ``QueryResult.error``)."""

    try:
        statement = validate(sql, db_path)
    except (UnsafeSQL, FileNotFoundError) as error:
        return QueryResult(sql=sql, error=str(error))
    conn = _connect_readonly(db_path)
    started = time.perf_counter()
    try:
        conn.set_authorizer(_authorizer(_user_tables(conn, statement)))
        conn.set_progress_handler(lambda: int(time.perf_counter() - started > timeout_s), 10_000)
        cursor = conn.execute(statement)
        rows = cursor.fetchmany(max_rows + 1)
        columns = [d[0] for d in cursor.description or []]
        truncated = len(rows) > max_rows
        return QueryResult(statement, columns, [tuple(r) for r in rows[:max_rows]], truncated,
                           elapsed_s=round(time.perf_counter() - started, 4))
    except sqlite3.OperationalError as error:
        message = "Query exceeded the time limit." if "interrupted" in str(error) else f"Query failed: {error}"
        return QueryResult(statement, error=message, elapsed_s=round(time.perf_counter() - started, 4))
    except sqlite3.DatabaseError as error:
        return QueryResult(statement, error=f"Query failed: {error}", elapsed_s=round(time.perf_counter() - started, 4))
    finally:
        conn.close()
