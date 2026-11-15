"""Question routing: document RAG, structured SQL, or hybrid (Phase 4, Task 5).

Deterministic and conservative, so existing document questions keep using the
RAG pipeline:

* **structured signal**: an explicit reference to tabular data ("table",
  "dataset", "spreadsheet", "csv", "excel", "uploaded", "records", "rows",
  "database", or a loaded table/dataset name), or a schema term that is
  *distinctive*: a table, column or categorical value that does not occur in
  the document vocabulary. Generic words shared with the report (e.g.
  "sales", "Europe") do not count on their own.
* **document signal**: a reference to the report/document ("annual report",
  "report", "document", "disclosed", "according to the report").

Both signals -> hybrid; structured only -> sql; otherwise -> document (the
default, unchanged Phase 3.1 behaviour). With no tables loaded, everything
goes to the document pipeline.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from app.structured.schema import Schema

_DATA_REFERENCE_RE = re.compile(
    r"\b(tables?|datasets?|data ?set|spreadsheets?|sheets?|csv|excel|xlsx|uploaded|records?|rows?|database|sql|"
    r"structured data|in the data|the data)\b", re.I)
_DOCUMENT_REFERENCE_RE = re.compile(
    r"\b(annual report|the report|report says|report's|reported in the|document|disclosed|according to the report|"
    r"management report|filing)\b", re.I)
_STOP = {"the", "a", "an", "of", "in", "on", "for", "and", "or", "to", "by", "with", "what", "which", "how", "is", "are",
         "was", "were", "did", "do", "does", "each", "per", "all", "total", "number", "value", "values", "id", "name",
         "date", "year", "month", "type", "data", "table", "code", "count", "sum", "average"}


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 2 and t not in _STOP}


def schema_terms(schema: Schema) -> dict[str, set[str]]:
    """Vocabulary of the loaded data: table names, column names (both forms) and categorical values."""

    terms = {"tables": set(), "columns": set(), "values": set()}
    for table in schema.tables:
        for name in (table.name, table.original_name, table.dataset):
            terms["tables"] |= _tokens(str(name).replace("_", " "))
        for column in table.columns:
            terms["columns"] |= _tokens(column.name.replace("_", " ")) | _tokens(column.original_name)
            for value in column.sample_values:
                terms["values"] |= _tokens(str(value))
    for key in ("tables", "columns"):
        # Singular forms too: "kpi" matches table "segment_kpis", "order" matches "orders".
        terms[key] |= {t[:-1] for t in terms[key] if t.endswith("s") and len(t) > 3}
    return terms


_SPLIT_RE = re.compile(r",?\s+(?:and|while|whereas)\s+(?=(?:what|which|how|who|when|where|is|was|were|does|did)\b)", re.I)
_COMPARE_SPLIT_RE = re.compile(r"\s+(?:with|against|to)\s+(?=the\b[^?]*\b(?:annual report|report|document)\b)", re.I)
_DATA_PHRASE_RE = re.compile(r"\s*\b(?:in|from|according to)\s+the\s+[\w\s]*?\b(?:table|data|dataset|sheet|spreadsheet)\b", re.I)


def split_hybrid(question: str) -> dict:
    """Data part and document part of a hybrid question (each answered independently).

    Splits "A in the <table>, and what ... according to the report?" or
    "Compare A in the <table> with B in the annual report" at the clause
    boundary and assigns each part by its own signal. If the question cannot be
    split cleanly, both sides get the whole question.
    """

    splitter = _COMPARE_SPLIT_RE if re.match(r"^\s*compare\b", question, re.I) else _SPLIT_RE
    parts = [p.strip(" ,.?") for p in splitter.split(question, maxsplit=1) if p and p.strip(" ,.?")]
    if len(parts) != 2:
        return {"sql": question, "document": question, "split": False}
    data_part = [p for p in parts if _DATA_REFERENCE_RE.search(p) and not _DOCUMENT_REFERENCE_RE.search(p)]
    doc_part = [p for p in parts if _DOCUMENT_REFERENCE_RE.search(p)]
    if len(data_part) != 1 or len(doc_part) != 1:
        return {"sql": question, "document": question, "split": False}
    sql_q, doc_q = data_part[0], doc_part[0]
    sql_q = re.sub(r"^\s*compare\s+", "What is ", sql_q, flags=re.I)
    content = _tokens(_DOCUMENT_REFERENCE_RE.sub(" ", doc_q)) - {"what", "figure", "says", "say"}
    if len(content) < 2:
        # "... with the annual report": ask the document the data part's question.
        doc_q = f"{_DATA_PHRASE_RE.sub('', sql_q).rstrip(' ?')} according to the annual report"
    if not re.match(r"^(what|which|how|who|when|where|is|was|were|does|did)\b", doc_q, re.I):
        doc_q = f"What is {doc_q}"
    # Years stated once for the whole question apply to both parts.
    for year in re.findall(r"\b(?:19|20)\d{2}\b", question):
        if year not in doc_q:
            doc_q = f"{doc_q.rstrip(' ?')} in {year}"
        if year not in sql_q:
            sql_q = f"{sql_q.rstrip(' ?')} in {year}"
    return {"sql": sql_q.rstrip(" ?") + "?", "document": doc_q.rstrip(" ?") + "?", "split": True}


def classify(question: str, schema: Schema | None, document_vocabulary: Iterable[str] = ()) -> dict:
    if schema is None or not schema.tables:
        return {"route": "document", "reason": "No structured tables are loaded.", "signals": {}}
    vocab = {w.lower() for w in document_vocabulary}
    q = _tokens(question)
    terms = schema_terms(schema)
    distinctive = sorted(t for t in q & (terms["tables"] | terms["columns"] | terms["values"]) if t not in vocab)
    data_ref = sorted({m.group(0).lower() for m in _DATA_REFERENCE_RE.finditer(question)})
    # "<table term> data/table/sheet", e.g. "the orders data", "the segment KPI table".
    for match in re.finditer(r"\b([a-z0-9]+)\s+(?:data|table|sheet|dataset|records)\b", question.lower()):
        if match.group(1) in terms["tables"] | terms["columns"]:
            data_ref.append(match.group(0))
    data_ref = sorted(set(data_ref))
    table_named = sorted(q & terms["tables"])
    doc_ref = sorted({m.group(0).lower() for m in _DOCUMENT_REFERENCE_RE.finditer(question)})
    structured = bool(data_ref or distinctive or (table_named and len(table_named) >= 2))
    signals = {"data_reference": data_ref, "distinctive_schema_terms": distinctive, "table_terms": table_named,
               "document_reference": doc_ref}
    if structured and doc_ref:
        return {"route": "hybrid", "reason": "The question refers to both the loaded data and the document.", "signals": signals}
    if structured:
        return {"route": "sql", "reason": "The question refers to the loaded structured data.", "signals": signals}
    return {"route": "document", "reason": "No reference to the loaded data; answered from the document.", "signals": signals}
