"""Turn an ``Assistant.ask`` result into a display model (Phase 5). No Streamlit import: unit-testable.

Nothing here changes an answer. The text, citations, validation status, abstention and calculated
figures are taken from the backend result as they are; this module only arranges and labels them:

* **raw values** are the rows returned by the read-only SQL query;
* **derived values** are figures calculated in Python by ``analytics`` / ``shares`` (period changes,
  shares, ratios, column comparisons). They are never model output;
* model-computed columns the backend ignored, and results it rejected, are listed separately so they
  are not mistaken for answers.
"""

from __future__ import annotations

import re

ROUTE_LABELS = {"document": "Document (RAG)", "sql": "Structured data (SQL)", "hybrid": "Hybrid: data + document"}
STATUS_LABELS = {"SUPPORTED": "Supported by the cited evidence", "PARTIALLY_SUPPORTED": "Partially supported",
                 "UNSUPPORTED": "Not supported by the cited evidence", "ABSTAINED": "Abstained (no reliable evidence)"}

_UNIT_SUFFIXES = [
    (re.compile(r"(?:^|_)usd_bn$", re.I), "USD bn"), (re.compile(r"(?:^|_)usd_m$", re.I), "USD m"),
    (re.compile(r"(?:^|_)usd$", re.I), "USD"), (re.compile(r"(?:^|_)eur_m$", re.I), "EUR m"),
    (re.compile(r"(?:^|_)eur$", re.I), "EUR"), (re.compile(r"(?:^|_)gbp$", re.I), "GBP"),
    (re.compile(r"(?:^|_)(?:pct|percent)$", re.I), "%"), (re.compile(r"(?:^|_)fte$", re.I), "FTE"),
    (re.compile(r"(?:^|_)mt$", re.I), "Mt"), (re.compile(r"(?:^|_)kt$", re.I), "kt"),
    (re.compile(r"(?:^|_)(?:t|tonnes)$", re.I), "tonnes"),
]


def unit_of(column: str | None) -> str | None:
    """Unit read from a column-name suffix (``sales_usd_m`` -> ``USD m``); ``None`` when there is no unit hint."""

    for pattern, unit in _UNIT_SUFFIXES:
        if column and pattern.search(column):
            return unit
    return None


def label(column: str | None) -> str:
    if not column:
        return ""
    unit = unit_of(column)
    text = column.replace("_", " ")
    return f"{text} ({unit})" if unit else text


def _table(columns: list[str], rows: list[list]) -> dict:
    return {"columns": [label(c) for c in columns], "rows": [list(r) for r in rows]}


def _derived_tables(analytics: dict) -> list[dict]:
    tables = []
    changes = analytics.get("period_changes") or []
    if changes:
        has_group = any(c.get("group") is not None for c in changes)
        cols = (["Group"] if has_group else []) + ["Metric", "From period", "From value", "To period", "To value",
                                                     "Absolute change", "Change (%)"]
        rows = [([c.get("group")] if has_group else []) + [label(c["metric"]), c["from_period"], c["from_value"],
                                                           c["to_period"], c["to_value"], c["absolute_change"],
                                                           c.get("percent_change")] for c in changes]
        tables.append({"title": "Period change", "columns": cols, "rows": rows,
                       "note": "Calculated in Python from the raw values (to − from; % relative to the earlier period)."})
    comparison = analytics.get("column_comparison")
    if comparison:
        tables.append({"title": "Column comparison", "columns": ["From", "From value", "To", "To value", "Absolute change", "Change (%)"],
                       "rows": [[label(comparison["from_metric"]), comparison["from_value"], label(comparison["to_metric"]),
                                 comparison["to_value"], comparison["absolute_change"], comparison.get("percent_change")]],
                       "note": "Calculated in Python from the raw values."})
    share_rows = analytics.get("shares") or []
    if share_rows:
        detailed = all("numerator" in s for s in share_rows)
        has_group = any(s.get("group") is not None for s in share_rows)
        if detailed:
            cols = (["Group"] if has_group else []) + ["Category", "Raw value", "Total (denominator)", "Share (%)"]
            rows = [([s.get("group")] if has_group else []) + [s["label"], s["numerator"], s["denominator"], s["share_percent"]]
                    for s in share_rows]
        else:
            cols, rows = ["Category", "Raw value", "Share (%)"], [[s["label"], s.get("value"), s["share_percent"]] for s in share_rows]
        tables.append({"title": "Share of total", "columns": cols, "rows": rows,
                       "note": "Shares = raw value / total, calculated in Python (never taken from the model)."})
    ratios = analytics.get("ratios") or []
    if ratios:
        tables.append({"title": "Ratio", "columns": ["Category", "Numerator", "Denominator", "Ratio", "Ratio (%)"],
                       "rows": [[r.get("label"), r["numerator"], r["denominator"], r["ratio"], r.get("percent")] for r in ratios],
                       "note": "Calculated in Python from the raw columns."})
    return tables


def present_sql(sql: dict) -> dict:
    notes, warnings = [], []
    if sql.get("truncated"):
        warnings.append("The result was truncated at the row limit; only the first rows are shown.")
    presentation = sql.get("presentation") or {}
    ignored = (presentation.get("model_computed_columns_ignored") or []) + (presentation.get("model_derived_columns_ignored") or [])
    if ignored:
        notes.append("Model-computed columns not used (recalculated from raw values instead): " + ", ".join(ignored) + ".")
    if presentation.get("metrics_not_asked"):
        notes.append("Columns not asked about were left out: " + ", ".join(presentation["metrics_not_asked"]) + ".")
    for kind in ("share_validation", "ratio_validation"):
        check = sql.get(kind) or {}
        if check.get("denominator_verified") is True:
            notes.append("The total (denominator) was checked against an independent read-only query.")
        elif check.get("denominator_verified") is False and check.get("denominator_note"):
            notes.append(f"Denominator: {check['denominator_note']}.")
    scope = sql.get("scope") or {}
    if scope.get("filters"):
        notes.append("Scope filters checked: " + ", ".join(f"{f['value']} ({', '.join(f['columns'])})" for f in scope["filters"]) + ".")
    generation = sql.get("generation") or {}
    retries = [k for k in ("retry_for_missing_periods", "retry_for_scope", "retry_for_raw_values") if k in generation]
    if retries:
        notes.append("The first query was rejected by validation and regenerated once ("
                     + ", ".join(r.replace("retry_for_", "").replace("_", " ") for r in retries) + ").")
    for name in sql.get("resolved_names") or []:
        if isinstance(name, dict) and name.get("table"):
            notes.append(f"Names resolved from table {name['table']}.")
            break
    rejected = sql.get("rejected_result")
    abstained = bool(sql.get("abstained"))
    return {
        "answer": sql.get("answer") or "",
        "error": sql.get("error"),
        "abstained": abstained,
        "sql": sql.get("sql"),
        "raw": _table(sql.get("columns") or [], sql.get("rows") or []) if sql.get("columns") else None,
        "raw_title": "Raw values returned by the read-only query",
        "derived": [] if abstained else _derived_tables(sql.get("analytics") or {}),
        "original_query_result": _table(sql["raw_result"]["columns"], sql["raw_result"]["rows"]) if sql.get("raw_result") else None,
        "rejected": _table(rejected["columns"], rejected["rows"]) if rejected and rejected.get("columns") else None,
        "chart": sql.get("chart") if (sql.get("chart") or {}).get("type") and not abstained else None,
        "chart_reason": (sql.get("chart") or {}).get("reason"),
        "tables": [{"table": s["table"], "source_file": s.get("source_file"), "sheet": s.get("sheet")} for s in sql.get("sources") or []],
        "notes": notes,
        "warnings": warnings,
    }


def present_rag(rag: dict) -> dict:
    validation = rag.get("validation") or {}
    status = validation.get("overall_status") or "UNKNOWN"
    citations = []
    for i, c in enumerate(rag.get("citations") or [], start=1):
        citations.append({"n": i, "source": c.get("source"), "page": c.get("page"), "text": c.get("text") or "",
                          "section": c.get("section"), "scope": c.get("scope")})
    claims = [{"status": c.get("status"), "claim": c.get("claim_text"), "source": c.get("source"), "page": c.get("page"),
               "explanation": c.get("explanation")} for c in validation.get("claims") or []]
    calc = rag.get("calculation")
    derived = None
    if calc:
        derived = {"sentence": calc.get("sentence"), "difference": calc.get("difference"), "percent": calc.get("percent"),
                   "unit": calc.get("unit"),
                   "operands": [{"value": o.get("raw"), "unit": o.get("unit"), "label": o.get("label"), "page": o.get("page")}
                                for o in calc.get("operands") or []]}
    completeness = validation.get("completeness") or {}
    return {
        "answer": rag.get("answer") or "",
        "status": status,
        "status_label": STATUS_LABELS.get(status, status.replace("_", " ").title()),
        "reason": validation.get("reason"),
        "abstained": status == "ABSTAINED",
        "unavailable": rag.get("unavailable"),
        "citations": citations,
        "pages": sorted({(c["source"], c["page"]) for c in citations}, key=lambda x: (str(x[0]), x[1] or 0)),
        "claims": claims,
        "calculation": derived,
        "completeness": completeness.get("status") if completeness else None,
        "citation_fallback": bool(rag.get("citation_fallback")),
    }


def present(result: dict) -> dict:
    route = result.get("route") or "document"
    sql = result.get("sql_result")
    rag = result.get("rag_result")
    parts = result.get("sub_questions") or {}
    return {
        "route": route,
        "source_label": ROUTE_LABELS.get(route, route),
        "routing_reason": (result.get("routing") or {}).get("reason"),
        "answer": result.get("answer") or "",
        "abstained": bool(result.get("abstained")),
        "structured": present_sql(sql) if sql else None,
        "document": present_rag(rag) if rag else None,
        "sub_questions": {"data": parts.get("sql"), "document": parts.get("document")} if route == "hybrid" and parts.get("split") else None,
    }


def format_value(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:,.4f}".rstrip("0").rstrip(".")
    return str(value)
