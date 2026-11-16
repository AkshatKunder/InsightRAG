"""Natural language to SQL with the local Ollama model (Phase 4, Task 3).

The prompt contains only the discovered schema (``schema.describe``): table
and column identifiers, types, verified joins, value ranges and sample
categorical values. It contains no dataset-specific example queries.

The model's output is reduced to one statement (``extract_sql``), validated
by ``sql_safety.validate`` and, if invalid or failing, sent back once with the
error message for a repair. If the model answers ``NO_SQL`` (the tables
cannot answer the question) or no valid SQL results, no SQL is executed.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from app.structured import periods, sql_safety
from app.structured.schema import Schema, describe

NO_SQL = "NO_SQL"

PROMPT = """You translate business questions into one SQLite query.

DATABASE SCHEMA:
{schema}

RULES:
1. Write exactly one SQLite SELECT statement (WITH ... SELECT is allowed). Never modify data.
2. Use only the tables and columns listed above, with their exact names. Quote nothing unless needed.
3. Join tables only through a VERIFIED JOIN listed above.
4. Dates are TEXT 'YYYY-MM-DD': filter years with strftime('%Y', col) = '2025', months with
   strftime('%Y-%m', col) = '2025-03', ranges with col BETWEEN 'YYYY-MM-DD' AND 'YYYY-MM-DD'.
5. Use aggregate functions (SUM, COUNT, AVG, MIN, MAX) with GROUP BY for totals and breakdowns, and
   give every computed column a short snake_case alias.
6. For "top"/"highest"/"lowest" questions use ORDER BY with LIMIT.
7. Match text values exactly as listed in the schema values.
8. For changes, growth or comparisons between periods, return one row per period with the period column
   and the value (and the category column if several categories); do not compute differences or
   percentage changes in SQL: they are calculated afterwards from your rows.
9. For shares, percentages of a total or breakdowns, return the underlying value per category; do not
   compute percentages in SQL. Use 1.0 * for any other division to avoid integer division.
10. If the tables cannot answer the question, output exactly: {no_sql}
11. Output only the SQL (or {no_sql}), with no explanation.

GENERIC EXAMPLES (for an unrelated table sales(region TEXT, yr INTEGER, sale_date DATE, amount REAL)):
- How did West amount change between 2023 and 2024?
  SELECT yr, SUM(amount) AS amount FROM sales WHERE region = 'West' AND yr IN (2023, 2024) GROUP BY yr ORDER BY yr
- What share of 2024 amount does each region have?
  SELECT region, SUM(amount) AS amount FROM sales WHERE yr = 2024 GROUP BY region
- Monthly amount in 2024:
  SELECT strftime('%Y-%m', sale_date) AS month, SUM(amount) AS amount FROM sales
  WHERE sale_date BETWEEN '2024-01-01' AND '2024-12-31' GROUP BY month ORDER BY month

QUESTION: {question}
SQL:"""

REPAIR = """{prompt}
{previous}

The previous SQL was rejected: {error}
Write a corrected query following the rules, or {no_sql}.
SQL:"""


@dataclass
class SQLGeneration:
    question: str
    sql: str | None
    attempts: list[dict] = field(default_factory=list)
    no_sql: bool = False
    error: str | None = None


def extract_sql(text: str) -> str | None:
    """The first SQL statement in a model response (code fences and prose removed)."""

    if not text:
        return None
    if re.search(rf"\b{NO_SQL}\b", text) and not re.search(r"\bselect\b", text, re.I):
        return NO_SQL
    fenced = re.search(r"```(?:sql|sqlite)?\s*(.+?)```", text, re.S | re.I)
    body = fenced.group(1) if fenced else text
    match = re.search(r"\b(with|select)\b.+", body, re.S | re.I)
    if not match:
        return None
    sql = match.group(0).strip()
    # Stop at the first statement end outside string literals.
    out, quote = [], False
    for ch in sql:
        if ch == "'":
            quote = not quote
        if ch == ";" and not quote:
            break
        out.append(ch)
    sql = "".join(out).strip()
    # Drop trailing prose lines after the query (e.g. "This query returns ...").
    lines = sql.splitlines()
    while len(lines) > 1 and re.match(r"^\s*(this|the|note|explanation)\b", lines[-1], re.I):
        lines.pop()
    return "\n".join(lines).strip() or None


_ENTITY_QUESTION_RE = re.compile(r"\b(which|who|list|name|top)\b", re.I)


def entity_name_hints(question: str, schema: Schema) -> list[str]:
    """Ask for names when the question asks *which* entities and the schema has a verified name column.

    Schema-driven, not a global rule: only tables with a verified primary key and a
    unique, complete name/title/label column qualify, and only when the question
    asks which/who/list/top about that table's entity ("customers" for a table
    ``..._customers``).
    """

    if not _ENTITY_QUESTION_RE.search(question):
        return []
    tokens = {t[:-1] if t.endswith("s") else t for t in re.findall(r"[a-z]+", question.lower())}
    notes = []
    for table in schema.tables:
        entity = table.name.split("_")[-1]
        entity = entity[:-1] if entity.endswith("s") else entity
        name_col = next((c for c in table.columns if c.type == "TEXT" and re.search(r"name|title|label", c.name)
                         and c.missing == 0 and c.distinct == table.row_count and c.name != table.primary_key), None)
        if table.primary_key and name_col and entity in tokens:
            notes.append(f"The question asks for {entity}s: return {table.name}.{name_col.name} (join through a VERIFIED "
                         f"JOIN on {table.primary_key} if needed), not only the ID.")
    return notes


def lint(sql: str, schema: Schema) -> str | None:
    """Schema-aware checks for valid-but-wrong SQL (reported back to the model for a repair).

    * date functions (strftime/date/julianday) applied to a non-DATE column always yield NULL.
    """

    types = {c.name.lower(): c.type for t in schema.tables for c in t.columns}
    for match in re.finditer(r"\b(strftime|date|julianday)\s*\(\s*(?:'[^']*'\s*,\s*)?([A-Za-z_][\w.]*)\s*[,)]", sql, re.I):
        column = match.group(2).split(".")[-1].lower()
        if types.get(column) in ("INTEGER", "REAL"):
            return (f"{match.group(1)}() is applied to the numeric column '{column}', which returns NULL; "
                    f"compare the number directly (e.g. {column} = 2025).")
    return None


def rewrite_year_strftime(sql: str, schema: Schema) -> tuple[str, list[str]] | None:
    """``strftime('%Y', year)`` on an integer *year* column is the column itself (Phase 4.2).

    The lint rejects date functions on numeric columns. When every attempt was rejected for that reason
    alone, this narrow deterministic rewrite is tried: only the ``'%Y'`` format, and only for INTEGER
    columns whose values are all years (1900-2100). The rewritten query is validated again like any
    generated query. Returns ``(sql, rewritten_columns)`` or ``None``.
    """

    years = {c.name.lower() for t in schema.tables for c in t.columns
             if c.type == "INTEGER" and c.min is not None and 1900 <= c.min <= c.max <= 2100}
    done = []

    def repl(match):
        column = match.group(1).split(".")[-1].strip('"').lower()
        if column not in years:
            return match.group(0)
        done.append(column)
        return match.group(1)

    rewritten = re.sub(r"\bstrftime\s*\(\s*'%Y'\s*,\s*([A-Za-z_][\w.\"]*)\s*\)", repl, sql, flags=re.I)
    return (rewritten, done) if done else None


def generate_sql(
    question: str,
    schema: Schema,
    db_path: Path | str,
    complete: Callable[[str], str],
    max_repairs: int = 2,
    extra_notes: list[str] | None = None,
    entity_hints: bool = True,
) -> SQLGeneration:
    """Generate and validate SQL. ``complete(prompt) -> text`` is the LLM call (injected).

    ``extra_notes`` are added to the period hints, e.g. when a previous result missed a requested year.
    ``entity_hints`` adds the schema-driven "return the name column" note (see ``entity_name_hints``).
    """

    names = entity_name_hints(question, schema) if entity_hints else []
    notes = periods.hints(question) + names + list(extra_notes or [])
    asked = question.strip() + ("\nNOTES:\n- " + "\n- ".join(notes) if notes else "")
    prompt = PROMPT.format(schema=describe(schema), question=asked, no_sql=NO_SQL)
    result = SQLGeneration(question, None)
    current = prompt
    for attempt in range(max_repairs + 1):
        raw = complete(current)
        sql = extract_sql(raw)
        record = {"attempt": attempt + 1, "raw": raw, "sql": sql}
        result.attempts.append(record)
        if sql == NO_SQL:
            result.no_sql, result.error = True, "The loaded tables cannot answer this question."
            return result
        if not sql:
            error = "The model did not return a SQL query."
        else:
            try:
                validated = sql_safety.validate(sql, db_path)
                problem = lint(validated, schema) or periods.quarter_filter_issue(question, validated)
                if problem:
                    raise sql_safety.UnsafeSQL(problem)
                result.sql = validated
                record["valid"] = True
                return result
            except sql_safety.UnsafeSQL as exc:
                error = str(exc)
        record["valid"] = False
        record["error"] = error
        result.error = error
        current = REPAIR.format(prompt=prompt, previous=sql or raw, error=error, no_sql=NO_SQL)
    # Every attempt was rejected: try the deterministic year-strftime rewrite on each attempt, in order.
    for record in list(result.attempts):
        if not record.get("sql") or "numeric column" not in (record.get("error") or ""):
            continue
        rewrite = rewrite_year_strftime(record["sql"], schema)
        if not rewrite:
            continue
        try:
            validated = sql_safety.validate(rewrite[0], db_path)
            if lint(validated, schema) or periods.quarter_filter_issue(question, validated):
                continue
        except sql_safety.UnsafeSQL:
            continue
        result.attempts.append({"attempt": "rewrite", "from_attempt": record["attempt"], "sql": validated, "valid": True,
                                "rewrite": f"strftime('%Y', col) -> col for {', '.join(rewrite[1])}"})
        result.sql, result.error = validated, None
        return result
    return result
