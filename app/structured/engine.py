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

from app.structured import analytics, charts, nl2sql, periods, scope, shares, sql_safety
from app.structured.ingestion import DEFAULT_DB
from app.structured.router import classify, split_hybrid
from app.structured.schema import Schema, discover

ABSTENTION = "I could not find this information in the provided document."
_LEADING_ARTICLE_RE = re.compile(r"(?:(?<=^)|(?<=[.!?]\s))(?:The|This|That|These|Those|A|An)\s+(?=[A-Z])")
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


def _cell(column: str, value) -> str:
    # Years and other period values are shown as written (2024, not "2,024").
    if isinstance(value, int) and re.search(r"(?:^|_)(year|yr|fy|period|month|quarter)(?:$|_)", column, re.I):
        return str(value)
    return _fmt(value)


def render_answer(result: sql_safety.QueryResult, derived: dict) -> str:
    """Natural-language answer built only from the returned rows."""

    if not result.ok:
        return f"The structured data could not answer this: {result.error}"
    if not result.rows:
        return "The query returned no matching records, so there is no figure to report."
    cols, rows = result.columns, result.rows
    if len(rows) == 1:
        parts = [f"{_label(c)}: {_cell(c, v)}" for c, v in zip(cols, rows[0])]
        text = "Result: " + "; ".join(parts) + "."
    else:
        shown = rows[:10]
        lines = ["; ".join(f"{_label(c)} {_cell(c, v)}" for c, v in zip(cols, r)) for r in shown]
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


def render_period_comparison(result: sql_safety.QueryResult, derived: dict, filters: list[dict]) -> str:
    """Multi-period answer: the original value of each period, then the change calculated in Python."""

    scope_text = ", ".join(f"{c} = '{f['value']}'" for f in filters for _, c in f["columns"][:1])
    roles = analytics.column_roles(result.columns, result.rows)
    period = roles["period"][0]
    lines = [f"Values by period{f' ({scope_text})' if scope_text else ''}, as returned by the query:"]
    for row in sorted(result.rows, key=lambda r: str(r[period])):
        labels = [str(row[i]) for i in roles["label"]]
        values = "; ".join(f"{_label(result.columns[i])} {_fmt(row[i])}" for i in roles["numeric"])
        lines.append(f"- {' / '.join(labels) + ', ' if labels else ''}{_cell(result.columns[period], row[period])}: {values}")
    lines.append("Calculated change (computed in Python from the values above, not taken from the SQL):")
    for change in derived["period_changes"]:
        group = f"{change['group']}, " if change["group"] is not None else ""
        pct = f" ({change['percent_change']:+.2f}%)" if change["percent_change"] is not None else ""
        lines.append(f"- {group}{_label(change['metric'])}: {_fmt(change['from_value'])} ({change['from_period']}) to "
                     f"{_fmt(change['to_value'])} ({change['to_period']}), {change['absolute_change']:+,g}{pct}.")
    return "\n".join(lines)


def render_shares(check: dict) -> str:
    """Raw values, the denominator and the share of each category, all labelled (Phase 4.3)."""

    scope_bits = [f"{f['columns'][0].split('.')[-1]} = '{f['value']}'" for f in check.get("filters", [])
                  if f["value"] not in check.get("selected_categories", [])]
    scope_bits += [f"year {', '.join(check['years'])}"] if check.get("years") else []
    scope_text = f" ({'; '.join(scope_bits)})" if scope_bits else ""
    lines = []
    for d in check["denominators"]:
        group = f"{_label(check['group'])} {_cell(check['group'], d['group'])}: " if d["group"] is not None else ""
        lines.append(f"{group}Share of total {_label(d['metric'])}{scope_text} by {_label(check['category'])} "
                     f"(raw value / total of {d['categories']} {_label(check['category'])} values = {_fmt(d['denominator'])}):")
        for s in check["shares"]:
            if s["group"] == d["group"]:
                marker = " (asked)" if str(s["label"]) in check.get("selected_categories", []) else ""
                lines.append(f"- {s['label']}{marker}: {_fmt(s['numerator'])} / {_fmt(s['denominator'])} = "
                             f"{s['share_percent']:.2f}%")
    verified = check.get("denominator_verified")
    lines.append("Shares are calculated in Python from the raw values above"
                 + ("; the total was checked against an independent query of the table." if verified
                    else f"; the total is the sum of the returned rows ({check.get('denominator_note', 'not independently verified')})."))
    if check.get("ignored_columns"):
        lines.append("Model-computed columns not used: " + ", ".join(check["ignored_columns"]) + ".")
    return "\n".join(lines)


def render_ratios(check: dict) -> str:
    lines = [f"Ratio of {_label(check['numerator_metric'])} to {_label(check['denominator_metric'])} (calculated in Python "
             "from the raw values):"]
    for r in check["ratios"]:
        lines.append(f"- {r['label']}: {_fmt(r['numerator'])} / {_fmt(r['denominator'])} = {r['ratio']:.4g} "
                     f"({r['percent']:.2f}%)")
    if check.get("ignored_columns"):
        lines.append("Model-computed columns not used: " + ", ".join(check["ignored_columns"]) + ".")
    return "\n".join(lines)


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
        if generation.sql is None and not generation.no_sql and nl2sql.entity_name_hints(question, self.schema):
            # The name note can lead the model to reference a name column without the join. Fall back to
            # the plain prompt; names are then added deterministically by _resolve_names.
            plain = nl2sql.generate_sql(question, self.schema, self.db_path, self.complete, entity_hints=False)
            out["generation"]["without_name_hint"] = {"attempts": plain.attempts, "no_sql": plain.no_sql}
            if plain.sql is not None:
                generation = plain
        if generation.sql is None:
            out["error"] = generation.error
            out["answer"] = f"I could not answer this from the structured data: {generation.error}"
            return out
        result = sql_safety.execute(generation.sql, self.db_path, self.max_rows, self.timeout_s)
        years = periods.requested_years(question)
        missing = self._missing_periods(question, result) if result.ok else []
        filters, issues = self._scope(question, result, years) if result.ok else ([], [])
        if years:
            out["scope"] = {"filters": [{"value": f["value"], "columns": [f"{t}.{c}" for t, c in f["columns"]]}
                                        for f in filters], "issues_first_query": issues}
        if missing or issues:
            # One retry that names the missing period(s) and/or the missing filter(s) explicitly.
            notes = []
            if missing:
                notes.append(f"The previous query returned no rows for {', '.join(missing)}. Return one row for each of "
                             f"{', '.join(years)}, without filtering to a single year.")
            for f in (filters if issues else []):
                table, column = f["columns"][0]
                notes.append(f"The question asks only about {column} = '{f['value']}' (table {table}): filter "
                             f"{column} = '{f['value']}' in the WHERE clause.")
            if issues:
                notes.append("The previous query was rejected because " + "; ".join(issues) + ". Return only the period "
                             "column and the requested value for the requested rows; do not compute changes in SQL.")
            retry = nl2sql.generate_sql(question, self.schema, self.db_path, self.complete, extra_notes=notes)
            key = "retry_for_missing_periods" if missing else "retry_for_scope"
            out["generation"][key] = {"missing": missing, "scope_issues": issues, "attempts": retry.attempts}
            if retry.sql is not None:
                retried = sql_safety.execute(retry.sql, self.db_path, self.max_rows, self.timeout_s)
                if retried.ok:
                    result, missing = retried, self._missing_periods(question, retried)
                    retry_filters, issues = self._scope(question, retried, years, extra=filters)
                    filters = retry_filters or filters
            if years:
                out["scope"]["issues_final_query"] = issues
        share_kind = shares.intent(question) if result.ok and not years and not (missing or issues) else None
        share_check = None
        if share_kind:
            result, share_check = self._share_or_ratio(question, result, out, share_kind)
            if share_check["issues"]:
                return self._refuse_share(out, result, share_check, share_kind)
        resolved = self._resolve_names(question, result) if result.ok else []
        out.update(sql=result.sql, columns=result.columns, rows=[list(r) for r in result.rows], truncated=result.truncated,
                   error=result.error, resolved_names=resolved)
        if share_check is not None:
            derived = {k: v for k, v in analytics.analyse(question, result.columns, result.rows).items() if k != "shares"}
            if share_kind == "share":
                derived["shares"] = share_check["shares"]
                derived["share_denominators"] = share_check["denominators"]
                out["answer"] = render_shares(share_check)
            else:
                derived["ratios"] = share_check["ratios"]
                out["answer"] = render_ratios(share_check)
            out["analytics"] = derived
            out["chart"] = charts.choose_chart(question, result.sql, result.columns, result.rows,
                                               parts_of_whole=share_kind == "share" and not share_check.get("group"))
            out["sources"] = self._sources(" ".join([result.sql] + [r["table"] for r in resolved]))
            return out
        if missing or issues:
            # Never calculate from a single period or from rows outside the requested scope: the rows are
            # withheld (kept only as ``rejected_result``) and the answer says the change cannot be calculated.
            out["rejected_result"] = {"columns": out["columns"], "rows": out["rows"]}
            out["columns"], out["rows"] = [], []
            out["abstained"] = True
            if missing:
                out["incomplete"] = {"missing_periods": missing}
                reason = f"the structured data query returned no values for {', '.join(missing)}"
            else:
                reason = "the structured data query did not return rows limited to " + ", ".join(
                    f"{c} = '{f['value']}'" for f in filters for _, c in f["columns"][:1]) + f" ({'; '.join(issues)})"
            out["answer"] = (f"I cannot calculate this change: {reason}, so no comparison between the periods is "
                             "reported.")
            out["sources"] = self._sources(result.sql) if result.ok else []
            return out
        if result.ok and analytics.CHANGE_INTENT_RE.search(question):
            self._present_change(question, result, out, scoped=bool(years))
        derived = analytics.analyse(question, result.columns, result.rows) if result.ok else {}
        out["analytics"] = derived
        if result.ok and years and not derived.get("period_changes") and not derived.get("column_comparison"):
            out["rejected_result"] = {"columns": out["columns"], "rows": out["rows"]}
            out["columns"], out["rows"] = [], []
            out["abstained"] = True
            out["answer"] = ("I cannot calculate this change: the query rows do not form one value per requested period "
                             f"({', '.join(years)}), so no comparison is reported.")
            out["sources"] = self._sources(result.sql)
            return out
        out["columns"], out["rows"] = result.columns, [list(r) for r in result.rows]
        out["answer"] = (render_period_comparison(result, derived, filters) if years and derived.get("period_changes")
                         else render_answer(result, derived))
        if result.ok:
            out["chart"] = charts.choose_chart(question, result.sql, result.columns, result.rows)
            # Tables used by the query plus any table used to resolve names.
            out["sources"] = self._sources(" ".join([result.sql] + [r["table"] for r in resolved]))
        return out

    def _check_share(self, question: str, result: sql_safety.QueryResult, kind: str) -> dict:
        years = sorted(set(re.findall(r"\b(?:19|20)\d{2}\b", question)))
        filters = scope.requested_filters(question, result.sql, self.schema, self.db_path, self.timeout_s)
        if kind == "ratio":
            check = shares.check_ratio(question, result.sql, result.columns, result.rows, self.schema, filters, years)
        else:
            check = shares.check_shares(question, result.sql, result.columns, result.rows, self.schema, self.db_path,
                                        filters, years, truncated=result.truncated, timeout_s=self.timeout_s)
        check["filters"] = [{"value": f["value"], "columns": [f"{t}.{c}" for t, c in f["columns"]]} for f in filters]
        check["years"] = years
        return check

    def _share_or_ratio(self, question: str, result: sql_safety.QueryResult, out: dict, kind: str):
        """Phase 4.3: validate a share/ratio result; one retry asking for raw values; never model fractions."""

        check = self._check_share(question, result, kind)
        record = {"kind": kind, "issues_first_query": check["issues"], "ignored_columns_first_query": check["ignored_columns"]}
        if check["issues"]:
            notes = [shares.raw_value_note(question, result.sql, self.schema, kind),
                     "The previous query was rejected because " + "; ".join(check["issues"]) + "."]
            retry = nl2sql.generate_sql(question, self.schema, self.db_path, self.complete, extra_notes=notes)
            out["generation"]["retry_for_raw_values"] = {"issues": check["issues"], "attempts": retry.attempts}
            if retry.sql is not None:
                retried = sql_safety.execute(retry.sql, self.db_path, self.max_rows, self.timeout_s)
                if retried.ok:
                    result, check = retried, self._check_share(question, retried, kind)
            record["issues_final_query"] = check["issues"]
        record.update({k: check.get(k) for k in ("ignored_columns", "filters", "years", "selected_categories",
                                                 "denominator_verified", "denominator_check", "denominator_note")
                       if check.get(k) is not None})
        out[f"{kind}_validation"] = record
        if not check["issues"]:
            if check["ignored_columns"]:
                out["raw_result"] = {"columns": list(result.columns), "rows": [list(r) for r in result.rows]}
                out["presentation"] = {"model_computed_columns_ignored": check["ignored_columns"]}
            result.columns, result.rows = check["columns"], [tuple(r) for r in check["rows"]]
        return result, check

    def _refuse_share(self, out: dict, result: sql_safety.QueryResult, check: dict, kind: str) -> dict:
        out.update(sql=result.sql, error=result.error, truncated=result.truncated, columns=[], rows=[], abstained=True,
                   rejected_result={"columns": list(result.columns), "rows": [list(r) for r in result.rows]})
        what = "these shares" if kind == "share" else "this ratio"
        out["answer"] = (f"I cannot calculate {what} reliably: {'; '.join(check['issues'])}. No share, percentage or ratio "
                         "is reported.")
        out["sources"] = self._sources(result.sql)
        return out

    @staticmethod
    def _missing_periods(question: str, result: sql_safety.QueryResult) -> list[str]:
        return periods.missing_years(question, result.columns, result.rows)

    def _scope(self, question: str, result: sql_safety.QueryResult, years: list[str],
               extra: list[dict] | None = None) -> tuple[list[dict], list[str]]:
        """Requested category filters and the scope issues of a multi-period result (Phase 4.2).

        Only multi-period comparison questions (two or more requested years) are checked. Filters
        found for an earlier query (``extra``) still apply to the retried one.
        """

        if not years:
            return [], []
        filters = scope.requested_filters(question, result.sql, self.schema, self.db_path, self.timeout_s)
        known = {f["value"].lower() for f in filters}
        filters += [f for f in extra or [] if f["value"].lower() not in known]
        return filters, scope.scope_issues(result.sql, result.columns, result.rows, filters, years)

    def _present_change(self, question: str, result: sql_safety.QueryResult, out: dict, scoped: bool) -> None:
        """Drop model-computed change columns; for multi-period questions keep only the asked metrics.

        The original query columns and rows stay available in ``raw_result``; changes are then
        calculated by ``analytics`` from the remaining raw values.
        """

        dropped = scope.derived_columns(result.columns, self.schema)
        keep = [i for i in range(len(result.columns)) if i not in dropped]
        if not keep:
            return
        columns, rows = scope.project(result.columns, result.rows, keep)
        not_asked = []
        if scoped:
            roles = analytics.column_roles(columns, rows)
            asked = scope.relevant_metrics(question, columns, roles["numeric"], self.schema)
            not_asked = [columns[i] for i in roles["numeric"] if i not in asked]
            columns, rows = scope.project(columns, rows, [i for i in range(len(columns)) if columns[i] not in not_asked])
        if dropped or not_asked:
            out["raw_result"] = {"columns": list(result.columns), "rows": [list(r) for r in result.rows]}
            out["presentation"] = {"model_derived_columns_ignored": [result.columns[i] for i in dropped],
                                   "metrics_not_asked": not_asked}
            result.columns, result.rows = columns, rows

    def _resolve_names(self, question: str, result: sql_safety.QueryResult) -> list[dict]:
        """Add entity names next to ID columns when the question asks for that entity.

        A column is resolved only when every non-null value is a value of a verified
        primary key whose table has a unique name column, and the question mentions
        that table's entity (e.g. "customers"). The lookup is a separate read-only,
        validated query; row order (ranking, top-N) is unchanged.
        """

        resolved = []
        q_tokens = {t[:-1] if t.endswith("s") else t for t in re.findall(r"[a-z]+", question.lower())}
        for table in self.schema.tables:
            key = table.primary_key
            name_col = next((c for c in table.columns if c.type == "TEXT" and re.search(r"name|title|label", c.name)
                             and c.missing == 0 and c.distinct == table.row_count and c.name != key), None)
            entity = re.split(r"_", table.name)[-1]
            if not key or not name_col or (entity[:-1] if entity.endswith("s") else entity) not in q_tokens:
                continue
            if name_col.name in [c.lower() for c in result.columns]:
                continue
            lookup = sql_safety.execute(f'SELECT "{key}", "{name_col.name}" FROM "{table.name}"', self.db_path,
                                        max_rows=max(table.row_count, 1), timeout_s=self.timeout_s)
            if not lookup.ok:
                continue
            names = dict(lookup.rows)
            for index in range(len(result.columns)):
                values = [row[index] for row in result.rows if row[index] is not None]
                if values and all(v in names for v in values):
                    result.columns.insert(index + 1, name_col.name)
                    result.rows = [tuple(row[:index + 1]) + (names.get(row[index]),) + tuple(row[index + 1:]) for row in result.rows]
                    resolved.append({"column": result.columns[index], "table": table.name, "key": key,
                                     "name_column": name_col.name})
                    break
        return resolved

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
                if route == "hybrid":
                    out["rag_result"] = ground_citations(parts["document"], out["rag_result"])
        out["answer"], out["sources"], out["abstained"] = self._combine(route, out["sql_result"], out["rag_result"])
        return out

    @staticmethod
    def _rag_ok(rag: dict | None) -> bool:
        return bool(rag) and (rag.get("validation") or {}).get("overall_status") != "ABSTAINED"

    @staticmethod
    def _sql_ok(sql: dict | None) -> bool:
        return bool(sql) and sql.get("error") is None and bool(sql.get("rows")) and not sql.get("abstained")

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


def ground_citations(question: str, rag: dict) -> dict:
    """Citation fallback for hybrid document answers (the RAG pipeline itself is unchanged).

    For every answer claim that the selected citations do not support, the first
    retrieved context chunk (in rank order) that supports that claim on its own is
    added as a citation; the whole answer is then validated again with the
    augmented citations. Nothing is added when no context chunk supports the claim.
    """

    from app.rag.citation_validator import validate_citation

    def normalise(text: str) -> str:
        # "The Europe sales figure ..." is otherwise read as the entity "The Europe" by the
        # validator's entity check; only a leading article before a capitalised word is removed
        # (value, unit, metric, year and scope checks are unchanged).
        return _LEADING_ARTICLE_RE.sub("", text or "")

    validation = rag.get("validation") or {}
    if validation.get("overall_status") in (None, "ABSTAINED", "SUPPORTED") or not rag.get("context"):
        return rag
    citations = list(rag.get("citations") or [])
    cited = {(c.get("source"), c.get("page")) for c in citations}
    added = []
    for claim in validation.get("claims") or []:
        if claim.get("status") == "SUPPORTED":
            continue
        for chunk in rag["context"]:
            key = (chunk.get("source"), chunk.get("page"))
            if key in cited:
                continue
            if validate_citation(normalise(claim.get("claim_text", "")), [chunk], question=question)["overall_status"] == "SUPPORTED":
                citations.append(chunk)
                cited.add(key)
                added.append(chunk.get("page"))
                break
    if not added:
        return rag
    revalidated = validate_citation(normalise(rag["answer"]), citations, question=question)
    return {**rag, "citations": citations, "validation": revalidated, "citation_fallback": {
        "added_pages": added, "status_before": validation.get("overall_status"),
        "status_after": revalidated.get("overall_status"), "leading_articles_normalised": True}}


def document_vocabulary(chunks: Iterable[dict]) -> set[str]:
    """Lower-cased words of the indexed document (for distinctive-term routing)."""

    vocab = set()
    for chunk in chunks:
        vocab |= set(re.findall(r"[a-z0-9]+", str(chunk.get("text", "")).lower()))
    return vocab
