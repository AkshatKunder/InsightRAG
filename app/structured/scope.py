"""Result-scope validation for multi-period comparison questions (Phase 4.2).

A change question such as "How did Europe sales change between 2024 and 2025?"
is answered only from a result that is scoped to what was asked:

* ``requested_filters``: categorical values named in the question (e.g.
  "Europe") are detected from the *data*, i.e. the distinct values of TEXT
  columns of the tables the query reads (read-only, validated lookups). Nothing
  is dataset- or question-specific.
* ``scope_issues``: the executed query and its rows must apply those filters.
  If the filter column is returned, every row must carry a requested value; if
  it is not returned, the SQL must filter it in a WHERE/HAVING/ON condition (a
  ``CASE WHEN`` in the select list is not a filter). Period values outside the
  requested range are also reported.
* ``derived_columns``: columns the model computed itself and named like a
  calculation (``sales_change``, ``growth_pct``) are not authoritative; they are
  removed and the change is calculated by ``analytics`` from the raw values.
  Real table columns with such names are kept.
* ``relevant_metrics``: the numeric columns the question asks about (by column
  name tokens); other metric columns are not presented.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

from app.structured import sql_safety
from app.structured.schema import Schema

MAX_DISTINCT_FOR_FILTERS = 500
DERIVED_NAME_RE = re.compile(
    r"(?:^|_)(change|changes|diff|difference|delta|growth|increase|decrease|variance|yoy|pct_change|"
    r"percent_change|percentage_change|ratio)(?:$|_)", re.I)
_UNIT_TOKENS = {"usd", "eur", "gbp", "m", "mn", "bn", "k", "pct", "fte", "t", "kt", "mt", "num", "total", "sum", "avg"}
_CLAUSE_RE = re.compile(r"\b(select|where|having|when|on|from|group|order|limit|then|else)\b", re.I)


def tables_in(sql: str, schema: Schema) -> list:
    return [t for t in schema.tables if re.search(rf"\b{re.escape(t.name)}\b", sql or "", re.I)]


def requested_filters(question: str, sql: str, schema: Schema, db_path: Path | str,
                      timeout_s: float = sql_safety.DEFAULT_TIMEOUT_S) -> list[dict]:
    """Categorical values named in the question, per TEXT column of the tables the query reads.

    Returns ``[{"value": "Europe", "columns": [("segment_kpis", "segment")]}, ...]``. A value
    contained in a longer matched value of the same column ("America" in "North America") is
    dropped. Values shorter than 2 characters or purely numeric are ignored.
    """

    text = question.lower()
    found: dict[str, dict] = {}
    for table in tables_in(sql, schema):
        for column in table.columns:
            if column.type != "TEXT" or not (0 < column.distinct <= MAX_DISTINCT_FOR_FILTERS):
                continue
            lookup = sql_safety.execute(f'SELECT DISTINCT "{column.name}" FROM "{table.name}" WHERE "{column.name}" IS NOT NULL',
                                        db_path, max_rows=MAX_DISTINCT_FOR_FILTERS, timeout_s=timeout_s)
            if not lookup.ok:
                continue
            values = [str(r[0]).strip() for r in lookup.rows if r[0] is not None]
            hits = [v for v in values if len(v) >= 2 and not re.fullmatch(r"[\d.,\s-]+", v)
                    and re.search(rf"(?<![\w]){re.escape(v.lower())}(?![\w])", text)]
            hits = [v for v in hits if not any(v != w and v.lower() in w.lower() for w in hits)]
            for value in hits:
                entry = found.setdefault(value.lower(), {"value": value, "columns": []})
                entry["columns"].append((table.name, column.name))
    return list(found.values())


_CLAUSE_END_RE = re.compile(r"(group\s+by|order\s+by|having|limit|window|union|except|intersect)\b", re.I)


def _depth0_scan(sql: str, start: int, stop_re: re.Pattern | None = None, split_re: re.Pattern | None = None):
    """Scan from ``start`` at parenthesis depth 0 (string literals skipped).

    Returns ``(end, cuts)``: where the clause ends (a closing parenthesis of the enclosing level or a
    ``stop_re`` keyword) and the positions of ``split_re`` keywords at depth 0.
    """

    depth, i, cuts = 0, start, []
    while i < len(sql):
        ch = sql[i]
        if ch == "'":
            j = sql.find("'", i + 1)
            i = len(sql) if j == -1 else j + 1
            continue
        boundary = i == 0 or not (sql[i - 1].isalnum() or sql[i - 1] == "_")
        if ch == "(":
            depth += 1
        elif ch == ")":
            if depth == 0:
                break
            depth -= 1
        elif depth == 0 and (boundary or not (ch.isalnum() or ch == "_")):
            # Keywords need a word boundary before them; punctuation separators (",") do not.
            if stop_re and stop_re.match(sql, i):
                break
            m = split_re.match(sql, i) if split_re else None
            if m:
                cuts.append((i, m.end()))
        i += 1
    return i, cuts


def _predicate_patterns(column: str, value: str) -> tuple[str, ...]:
    lit = re.escape(value.replace("'", "''"))
    col = rf"(?:\w+\.)?\"?{re.escape(column)}\"?"
    return (rf"{col}\s*(?:==?|like)\s*'{lit}'", rf"'{lit}'\s*==?\s*{col}", rf"{col}\s+in\s*\([^)]*'{lit}'[^)]*\)")


def _predicate_in_filter_clause(sql: str, column: str, value: str) -> bool:
    """``column = 'value'`` / ``IN (..., 'value')`` / ``LIKE`` inside WHERE, HAVING or a join ON.

    In a WHERE clause the predicate must hold in every top-level OR branch: ``segment = 'Europe' AND
    year = 2024 OR year = 2025`` does not restrict the 2025 rows to Europe.
    """

    patterns = _predicate_patterns(column, value)
    for pattern in patterns:
        for match in re.finditer(pattern, sql, re.I):
            clauses = _CLAUSE_RE.findall(sql[:match.start()])
            if not clauses or clauses[-1].lower() not in ("where", "having", "on"):
                continue
            if clauses[-1].lower() != "where":
                return True
            where = [m for m in re.finditer(r"\bwhere\b", sql[:match.start()], re.I)][-1]
            end, cuts = _depth0_scan(sql, where.end(), _CLAUSE_END_RE, re.compile(r"or\b", re.I))
            if match.start() >= end:
                continue  # the WHERE ended before the predicate (e.g. it is in a later subquery)
            starts = [where.end()] + [c[1] for c in cuts]
            ends = [c[0] for c in cuts] + [end]
            if all(any(re.search(p, sql[s:e], re.I) for p in patterns) for s, e in zip(starts, ends)):
                return True
    return False


def scope_issues(sql: str, columns: Sequence[str], rows: Sequence[Sequence], filters: list[dict],
                 years: Sequence[str] = ()) -> list[str]:
    """Reasons why the result is not scoped to the requested filters and periods (empty if it is)."""

    issues = []
    lower = [c.lower() for c in columns]
    wanted: dict[str, set[str]] = {}
    for f in filters:
        for _, column in f["columns"]:
            wanted.setdefault(column.lower(), set()).add(f["value"].lower())
    for f in filters:
        returned = [c for _, c in f["columns"] if c.lower() in lower]
        if returned:
            for column in returned:
                index = lower.index(column.lower())
                other = sorted({str(r[index]) for r in rows if r[index] is not None
                                and str(r[index]).strip().lower() not in wanted[column.lower()]})
                if other:
                    issues.append(f"the result contains rows for {column} values other than '{f['value']}' "
                                  f"({', '.join(other[:5])}); filter {column} = '{f['value']}' in the WHERE clause")
                elif not any(str(r[index]).strip().lower() == f["value"].lower() for r in rows):
                    issues.append(f"the result contains no rows for {column} = '{f['value']}'")
        elif not any(_predicate_in_filter_clause(sql, c, f["value"]) for _, c in f["columns"]):
            names = " or ".join(f"{t}.{c}" for t, c in f["columns"])
            issues.append(f"the query does not filter {names} = '{f['value']}' (add it to the WHERE clause)")
    if years:
        low, high = int(min(years)), int(max(years))
        for index, name in enumerate(columns):
            values = [r[index] for r in rows if r[index] is not None]
            if values and re.search(r"(?:^|_)(year|yr|fy|period)(?:$|_)", name, re.I) and all(
                    isinstance(v, int) and not isinstance(v, bool) for v in values):
                outside = sorted({v for v in values if not low <= v <= high})
                if outside:
                    issues.append(f"the result contains periods outside {low}-{high} ({', '.join(map(str, outside))})")
    return issues


def derived_columns(columns: Sequence[str], schema: Schema) -> list[int]:
    """Indices of model-computed calculation columns (named like a change, not a real table column)."""

    real = {c.name.lower() for t in schema.tables for c in t.columns}
    return [i for i, c in enumerate(columns) if DERIVED_NAME_RE.search(c) and c.lower() not in real]


def _tokens(name: str) -> set[str]:
    return {t[:-1] if t.endswith("s") and len(t) > 3 else t for t in re.split(r"[_\W]+", name.lower())
            if len(t) >= 3 and t not in _UNIT_TOKENS}


def relevant_metrics(question: str, columns: Sequence[str], numeric: Sequence[int], schema: Schema) -> list[int]:
    """Numeric columns whose name (or original name) the question mentions; all of them if none is mentioned."""

    q = _tokens(question)
    originals = {c.name.lower(): c.original_name for t in schema.tables for c in t.columns}
    hit = [i for i in numeric if _tokens(columns[i]) & q or _tokens(originals.get(columns[i].lower(), "")) & q]
    return hit or list(numeric)


def project(columns: Sequence[str], rows: Sequence[Sequence], keep: Sequence[int]) -> tuple[list[str], list[tuple]]:
    return [columns[i] for i in keep], [tuple(r[i] for i in keep) for r in rows]
