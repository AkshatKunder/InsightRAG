"""Structured-data question answering and the RAG/SQL assistant (Phase 4, Tasks 4-7).

``StructuredQA.ask``: refuse write requests -> generate SQL from the schema ->
validate -> execute read-only with limits -> deterministic answer text,
analytics and chart spec. The SQL is always returned. Failures are reported as
failures; no figures are produced without query results.

``Assistant.ask``: route the question (``router.classify``) and answer it with
the unchanged Phase 3.1 RAG pipeline, the structured engine, or both
(independently, combined with explicit source attribution). It abstains when
the required evidence is missing.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from pathlib import Path

from app.structured import analytics, charts, nl2sql, sql_safety
from app.structured.ingestion import DEFAULT_DB
from app.structured.router import classify, split_hybrid
from app.structured.schema import Schema, discover

ABSTENTION = "I could not find this information in the provided document."
_WRITE_INTENT_RE = re.compile(
    r"\b(delete|drop|truncate|update|insert|alter|modify|overwrite|erase)\b[^?.]*\b"
    r"(rows?|records?|tables?|data|columns?|values?|entries|database)\b", re.I)


def _fmt(value) -> str:
    if isinstance(value, bool) or value is None:
        return "—" if value is None else str(value)
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:,.2f}".rstrip("0").rstrip(".") if abs(value) < 1e15 else str(value)
    return str(value)


def _label(column: str) -> str:
    return column.replace("_", " ")


def render_answer(result: sql_safety.QueryResult, derived: dict) -> str:
    """Natural-language answer built only from the returned rows."""

    if not result.ok:
        return f"The structured data could not answer this: {result.error}"
    if not result.rows:
        return "The query returned no matching records, so there is no figure to report."
    cols, rows = result.columns, result.rows
    if len(rows) == 1:
        parts = [f"{_label(c)}: {_fmt(v)}" for c, v in zip(cols, rows[0])]
        text = "Result: " + "; ".join(parts) + "."
    else:
        shown = rows[:10]
        lines = ["; ".join(f"{_label(c)} {_fmt(v)}" for c, v in zip(cols, r)) for r in shown]
        text = f"The query returned {len(rows)}{'+' if result.truncated else ''} rows:\n- " + "\n- ".join(lines)
        if len(rows) > len(shown):
            text += f"\n(first {len(shown)} of {len(rows)} shown)"
    for change in derived.get("period_changes", [])[:10]:
        group = f"{change['group']}: " if change["group"] is not None else ""
        pct = f" ({change['percent_change']:+.2f}%)" if change["percent_change"] is not None else ""
        text += (f"\n{group}{_label(change['metric'])} changed from {_fmt(change['from_value'])} ({change['from_period']}) "
                 f"to {_fmt(change['to_value'])} ({change['to_period']}), {change['absolute_change']:+,g}{pct}.")
    comparison = derived.get("column_comparison")
    if comparison:
        pct = f" ({comparison['percent_change']:+.2f}%)" if comparison["percent_change"] is not None else ""
        text += (f"\n{_label(comparison['to_metric'])} vs {_label(comparison['from_metric'])}: "
                 f"{comparison['absolute_change']:+,g}{pct}.")
    for share in derived.get("shares", [])[:8]:
        text += f"\n{share['label']}: {share['share_percent']:.2f}% of the total."
    return text


class StructuredQA:
    def __init__(self, db_path: Path | str = DEFAULT_DB, complete: Callable[[str], str] | None = None,
                 max_rows: int = sql_safety.DEFAULT_MAX_ROWS, timeout_s: float = sql_safety.DEFAULT_TIMEOUT_S):
        self.db_path = Path(db_path)
        self.complete = complete
        self.max_rows = max_rows
        self.timeout_s = timeout_s
        self._schema: Schema | None = None

    @property
    def schema(self) -> Schema:
        if self._schema is None:
            self._schema = discover(self.db_path)
        return self._schema

    def refresh(self) -> None:
        self._schema = None

    def ask(self, question: str) -> dict:
        out = {"route": "sql", "question": question, "sql": None, "columns": [], "rows": [], "truncated": False,
               "error": None, "answer": None, "analytics": {}, "chart": {"type": None}, "sources": [], "generation": None}
        if _WRITE_INTENT_RE.search(question):
            out["error"] = "Only read-only questions are supported; the data cannot be modified."
            out["answer"] = out["error"]
            return out
        if not self.schema.tables:
            out["error"] = "No structured tables are loaded."
            out["answer"] = out["error"]
            return out
        if self.complete is None:
            raise ValueError("StructuredQA needs a 'complete' LLM callable to generate SQL")
        generation = nl2sql.generate_sql(question, self.schema, self.db_path, self.complete)
        out["generation"] = {"attempts": generation.attempts, "no_sql": generation.no_sql}
        if generation.sql is None:
            out["error"] = generation.error
            out["answer"] = f"I could not answer this from the structured data: {generation.error}"
            return out
        result = sql_safety.execute(generation.sql, self.db_path, self.max_rows, self.timeout_s)
        out.update(sql=result.sql, columns=result.columns, rows=[list(r) for r in result.rows], truncated=result.truncated,
                   error=result.error)
        derived = analytics.analyse(question, result.columns, result.rows) if result.ok else {}
        out["analytics"] = derived
        out["answer"] = render_answer(result, derived)
        missing = self._missing_periods(question, result) if result.ok else []
        if missing:
            # A change between periods cannot be stated when the query did not return every period.
            out["incomplete"] = {"missing_periods": missing}
            out["answer"] += (f"\nThe query did not return values for {', '.join(missing)}, so the change between "
                              "the periods cannot be calculated from this result.")
        if result.ok:
            out["chart"] = charts.choose_chart(question, result.sql, result.columns, result.rows)
            out["sources"] = self._sources(result.sql)
        return out

    @staticmethod
    def _missing_periods(question: str, result: sql_safety.QueryResult) -> list[str]:
        years = sorted(set(re.findall(r"\b(?:19|20)\d{2}\b", question)))
        if len(years) < 2 or not analytics.CHANGE_INTENT_RE.search(question):
            return []
        returned = {str(v)[:4] for row in result.rows for v in row if v is not None}
        returned |= {y for c in result.columns for y in re.findall(r"(?:19|20)\d{2}", c)}
        return [y for y in years if y not in returned]

    def _sources(self, sql: str) -> list[dict]:
        used = []
        for table in self.schema.tables:
            if re.search(rf"\b{re.escape(table.name)}\b", sql, re.I):
                used.append({"table": table.name, "source_file": table.source_file, "sheet": table.sheet,
                             "rows": table.row_count})
        return used


class Assistant:
    """Routes each question to document RAG, SQL, or both."""

    def __init__(self, structured: StructuredQA, rag_answer: Callable[[str], dict] | None,
                 document_vocabulary: Iterable[str] = ()):
        self.structured = structured
        self.rag_answer = rag_answer
        self.document_vocabulary = set(document_vocabulary)

    def ask(self, question: str) -> dict:
        routing = classify(question, self.structured.schema, self.document_vocabulary)
        route = routing["route"]
        out = {"question": question, "route": route, "routing": routing, "answer": None, "sql_result": None,
               "rag_result": None, "sources": []}
        # Hybrid: each side answers only its own part of the question, independently.
        parts = split_hybrid(question) if route == "hybrid" else {"sql": question, "document": question, "split": False}
        out["sub_questions"] = parts
        if route in ("sql", "hybrid"):
            out["sql_result"] = self.structured.ask(parts["sql"])
        if route in ("document", "hybrid"):
            if self.rag_answer is None:
                out["rag_result"] = {"answer": ABSTENTION, "validation": {"overall_status": "ABSTAINED"}, "citations": []}
            else:
                out["rag_result"] = self.rag_answer(parts["document"])
        out["answer"], out["sources"], out["abstained"] = self._combine(route, out["sql_result"], out["rag_result"])
        return out

    @staticmethod
    def _rag_ok(rag: dict | None) -> bool:
        return bool(rag) and (rag.get("validation") or {}).get("overall_status") != "ABSTAINED"

    @staticmethod
    def _sql_ok(sql: dict | None) -> bool:
        return bool(sql) and sql.get("error") is None and bool(sql.get("rows"))

    def _combine(self, route: str, sql: dict | None, rag: dict | None) -> tuple[str, list[dict], bool]:
        sources = []
        if self._sql_ok(sql):
            sources += [{"type": "table", **s, "sql": sql["sql"]} for s in sql["sources"]]
        if self._rag_ok(rag):
            sources += [{"type": "document", "source": c.get("source"), "page": c.get("page")} for c in rag.get("citations", [])]
        if route == "sql":
            return sql["answer"], sources, not self._sql_ok(sql)
        if route == "document":
            return rag["answer"], sources, not self._rag_ok(rag)
        parts = []
        if self._sql_ok(sql):
            parts.append(f"From the structured data ({', '.join(s['table'] for s in sql['sources']) or 'database'}):\n{sql['answer']}")
        else:
            parts.append(f"From the structured data: no result ({(sql or {}).get('error') or 'no matching records'}).")
        if self._rag_ok(rag):
            pages = ", ".join(str(c.get("page")) for c in rag.get("citations", []))
            parts.append(f"From the document (pages {pages or 'n/a'}):\n{rag['answer']}")
        else:
            parts.append("From the document: no supported evidence was found.")
        abstained = not self._sql_ok(sql) and not self._rag_ok(rag)
        if abstained:
            return ABSTENTION, sources, True
        return "\n\n".join(parts), sources, False


def document_vocabulary(chunks: Iterable[dict]) -> set[str]:
    """Lower-cased words of the indexed document (for distinctive-term routing)."""

    vocab = set()
    for chunk in chunks:
        vocab |= set(re.findall(r"[a-z0-9]+", str(chunk.get("text", "")).lower()))
    return vocab
