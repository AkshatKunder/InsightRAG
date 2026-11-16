"""Deterministic shares, percentages of a total and ratios (Phase 4.3).

Share questions ("What share of 2025 employees does each segment have?",
"percentage of total revenue by region", "contribution of each product") and
ratio questions ("ratio of operating income to sales", "income as a
percentage of sales") are answered from **raw metric values**, never from
fractions, percentages or ratios computed by the model in SQL:

* ``intent``: share / ratio / None (``per``-questions such as "revenue per
  order" are averages, not ratios, and are not matched).
* ``select_items`` / ``computed_fraction_columns``: result columns whose SQL
  expression divides or uses a window aggregate, or whose alias names a share,
  percentage or ratio (and is not a real table column), are not authoritative.
* ``check_shares``: identifies the category (and optional group) columns and the
  raw metric, then validates the result: one row per category within a group, no
  missing or negative values, a positive total, no LIMIT/truncation, the
  requested scope (``scope.scope_issues``) and, where the query is a simple
  aggregate of one table, an independent read-only denominator query for the
  requested years and categories. Shares are then numerator / denominator * 100.
* ``check_ratio``: two raw metric columns named in the question; each row's
  ratio is numerator / denominator with a non-zero denominator.

Every issue is a reason to retry once and then refuse, never to guess.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

from app.structured import analytics, scope, sql_safety
from app.structured.schema import Schema

SHARE_RE = re.compile(
    r"\b(share|shares|proportion|percentage of|per ?cent of|% of|contribution|contribute[sd]?|distribution|breakdown|split)\b",
    re.I)
_RATIO_PATTERNS = (
    re.compile(r"\bratio (?:of|between) (?P<a>.+?) (?:to|and) (?P<b>.+?)(?=[?.,;]| in \d| for | per | by |$)", re.I),
    re.compile(r"\b(?P<a>[\w ]+?) (?:as an?|as) (?:percentage|percent|share|proportion|fraction) of (?P<b>.+?)(?=[?.,;]| in \d| for | per | by |$)", re.I),
    re.compile(r"\b(?P<a>[\w ]+?)[- ]to[- ](?P<b>[\w]+(?: [\w]+)?) ratio\b", re.I),
)
FRACTION_NAME_RE = re.compile(
    r"(?:^|_)(share|shares|pct|percent|percentage|ratio|fraction|proportion|portion|contribution)(?:$|_)", re.I)
_TOTAL_WORDS_RE = re.compile(r"\b(total|all|whole|overall)\b", re.I)


def ratio_terms(question: str) -> tuple[str, str] | None:
    for pattern in _RATIO_PATTERNS:
        match = pattern.search(question)
        if match:
            a, b = match.group("a").strip(), match.group("b").strip()
            a = re.sub(r"^(?:what|which|show|give|is|was|the|me|their|its)\s+", "", a, flags=re.I)
            a = re.sub(r"^(?:(?:what|which|is|was|the)\s+)+", "", a, flags=re.I)
            if a and b and not _TOTAL_WORDS_RE.search(b):
                return a, b
    return None


def intent(question: str) -> str | None:
    if ratio_terms(question):
        return "ratio"
    if SHARE_RE.search(question):
        return "share"
    return None


# ---------------------------------------------------------------- SQL select list


def _main_select(sql: str) -> int | None:
    """Start of the outermost (last top-level) SELECT list."""

    starts = [m.end() for m in re.finditer(r"\bselect\b", sql, re.I)]
    top = [s for s in starts if sql[:s].count("(") == sql[:s].count(")")]
    return top[-1] if top else None


def select_items(sql: str) -> list[tuple[str, str]]:
    """``[(expression, output_name), ...]`` of the outermost SELECT (best effort; [] if unparsable)."""

    start = _main_select(sql or "")
    if start is None:
        return []
    end, cuts = scope._depth0_scan(sql, start, re.compile(r"from\b", re.I), re.compile(r","))
    bounds = [start] + [c[1] for c in cuts]
    stops = [c[0] for c in cuts] + [end]
    items = []
    for s, e in zip(bounds, stops):
        text = re.sub(r"^\s*distinct\s+", "", sql[s:e].strip(), flags=re.I)
        alias = re.search(r"\s+(?:as\s+)?\"?([A-Za-z_]\w*)\"?\s*$", text, re.I)
        if alias and not re.fullmatch(r"(?:\w+\.)?\"?\w+\"?", text):
            expr, name = text[:alias.start()].strip(), alias.group(1)
        else:
            expr, name = text, text.split(".")[-1].strip('"')
        items.append((expr, name))
    return items


def computed_fraction_columns(sql: str, columns: Sequence[str], schema: Schema) -> list[int]:
    """Result columns that are model-computed shares, percentages or ratios."""

    real = {c.name.lower() for t in schema.tables for c in t.columns}
    items = select_items(sql)
    exprs = [e for e, _ in items] if len(items) == len(columns) else [None] * len(columns)
    out = []
    for i, name in enumerate(columns):
        expr = exprs[i] or ""
        divides = bool(re.search(r"(?<![*/])/(?![*/])", re.sub(r"'[^']*'", "", expr)))
        window = bool(re.search(r"\bover\s*\(", expr, re.I))
        if divides or window or (FRACTION_NAME_RE.search(name) and name.lower() not in real):
            out.append(i)
    return out


# ---------------------------------------------------------------- shares


def _base_column(expr: str, table) -> tuple[str, str] | None:
    """``SUM(col)`` / ``col`` / ``COUNT(*)`` / ``COUNT(col)`` over a real column -> (aggregate, column)."""

    m = re.fullmatch(r"\s*(sum|count)\s*\(\s*(?:distinct\s+)?((?:\w+\.)?\"?(\w+|\*)\"?)\s*\)\s*", expr, re.I)
    if m:
        column = m.group(3)
        if column == "*" or column.lower() in {c.name.lower() for c in table.columns}:
            return m.group(1).upper(), column
        return None
    m = re.fullmatch(r"\s*(?:\w+\.)?\"?(\w+)\"?\s*", expr)
    if m and m.group(1).lower() in {c.name.lower() for c in table.columns}:
        return "SUM", m.group(1)
    return None


def _non_additive(expr: str) -> bool:
    return bool(re.search(r"\b(avg|min|max)\s*\(", expr or "", re.I))


def _category_and_group(question: str, columns: Sequence[str], rows: Sequence[Sequence], roles: dict):
    """(category index, group index or None, issue)."""

    labels = list(roles["label"])
    periods = [p for p in roles["period"] if len({r[p] for r in rows}) > 1]
    if len(labels) == 1:
        if len(periods) > 1:
            return None, None, "the result has several period columns; the grouping for the shares is ambiguous"
        return labels[0], (periods[0] if periods else None), None
    if not labels and len(periods) == 1:
        return periods[0], None, None  # share of each period in the total over the periods
    if len(labels) == 2 and not periods:
        q = question.lower()
        named = [i for i in labels if re.search(rf"\b(?:each|by|per|every)\s+{re.escape(columns[i].split('_')[-1].rstrip('s'))}", q)]
        if len(named) == 1:
            other = [i for i in labels if i != named[0]][0]
            return named[0], other, None
        return None, None, "the result has two category columns and the question does not say which one the shares are of"
    return None, None, "the result has no single category column to compute shares for"


def check_shares(question: str, sql: str, columns: Sequence[str], rows: Sequence[Sequence], schema: Schema,
                 db_path: Path | str, filters: list[dict], years: Sequence[str], truncated: bool = False,
                 timeout_s: float = sql_safety.DEFAULT_TIMEOUT_S) -> dict:
    """Validate a share result and compute the shares; ``{"issues": [...]}`` when it cannot be done reliably."""

    out = {"issues": [], "ignored_columns": [], "shares": [], "denominators": []}
    fraction = computed_fraction_columns(sql, columns, schema)
    out["ignored_columns"] = [columns[i] for i in fraction]
    keep = [i for i in range(len(columns)) if i not in fraction]
    cols, rws = scope.project(columns, rows, keep)
    roles = analytics.column_roles(cols, rws)
    items = select_items(sql)
    exprs = {name.lower(): expr for expr, name in items} if len(items) == len(columns) else {}
    metrics = scope.relevant_metrics(question, cols, roles["numeric"], schema) if roles["numeric"] else []
    if not metrics:
        out["issues"].append("the result has no raw metric values (only model-computed shares, percentages or ratios); "
                             "return the underlying value per category")
        return out
    if len(metrics) > 1:
        out["issues"].append("the question does not identify one metric for the shares (" + ", ".join(cols[i] for i in metrics) + ")")
        return out
    metric = metrics[0]
    if _non_additive(exprs.get(cols[metric].lower(), "")):
        out["issues"].append(f"{cols[metric]} is an average/minimum/maximum, which cannot be split into shares of a total")
        return out
    category, group, issue = _category_and_group(question, cols, rws, roles)
    if issue:
        out["issues"].append(issue)
        return out
    if truncated or re.search(r"\blimit\s+\d+\s*$", sql.strip().rstrip(";"), re.I):
        out["issues"].append("the result is limited (LIMIT or row cap), so its total is not the total for the requested scope")
    groups: dict = {}
    for row in rws:
        groups.setdefault(row[group] if group is not None else None, []).append(row)
    for key, items_ in groups.items():
        labels = [r[category] for r in items_]
        if len(set(labels)) != len(labels):
            out["issues"].append(f"{cols[category]} values repeat{f' within {key}' if key is not None else ''}; the rows "
                                 "are not one value per category")
        missing = [str(r[category]) for r in items_ if r[metric] is None or not analytics.is_number(r[metric])]
        if missing:
            out["issues"].append(f"{cols[metric]} is missing for {', '.join(missing)}, so the numerator is unknown")
        negative = [str(r[category]) for r in items_ if analytics.is_number(r[metric]) and r[metric] < 0]
        if negative:
            out["issues"].append(f"{cols[metric]} is negative for {', '.join(negative)}; shares of a total are not "
                                 "meaningful with negative parts")
    # A value named in the question that belongs to the category column selects the numerator ("What share
    # did Europe have?"); it does not restrict the denominator. Other named values are scope filters.
    selected = [f for f in filters if any(c.lower() == cols[category].lower() for _, c in f["columns"])]
    filters = [f for f in filters if f not in selected]
    out["selected_categories"] = [f["value"] for f in selected]
    if selected and len({str(r[category]).lower() for r in rws}) <= len(selected):
        out["issues"].append(f"the result contains only {', '.join(out['selected_categories'])}; all {cols[category]} "
                             "values of the requested scope are needed for the total")
    out["issues"] += scope.scope_issues(sql, cols, rws, filters, years)
    if not out["issues"]:
        out["issues"] += _verify_denominator(sql, cols, rws, cols[metric], exprs.get(cols[metric].lower(), cols[metric]),
                                             schema, db_path, filters, years, timeout_s, out)
    if out["issues"]:
        return out
    for key, items_ in groups.items():
        total = sum(r[metric] for r in items_)
        if not total:
            out["issues"].append(f"the total {cols[metric]}{f' for {key}' if key is not None else ''} is zero, so no "
                                 "share can be calculated")
            continue
        out["denominators"].append({"group": key, "metric": cols[metric], "denominator": total, "categories": len(items_)})
        for r in items_:
            out["shares"].append({"label": r[category], "group": key, "category_column": cols[category],
                                  "group_column": cols[group] if group is not None else None, "metric": cols[metric],
                                  "value": r[metric], "numerator": r[metric], "denominator": total,
                                  "share_percent": round(r[metric] / total * 100, 2)})
    if out["issues"]:
        out["shares"], out["denominators"] = [], []
    out["columns"], out["rows"] = cols, [list(r) for r in rws]
    out["metric"], out["category"] = cols[metric], cols[category]
    out["group"] = cols[group] if group is not None else None
    return out


def _verify_denominator(sql, cols, rows, metric, expr, schema, db_path, filters, years, timeout_s, out) -> list[str]:
    """Independent read-only total for the requested scope (single-table aggregates only)."""

    tables = scope.tables_in(sql, schema)
    out["denominator_verified"] = False
    if len(tables) != 1:
        out["denominator_note"] = "the query reads several tables; the denominator is the total of the returned rows"
        return []
    table = tables[0]
    base = _base_column(expr, table)
    if base is None:
        out["denominator_note"] = "the metric expression is not a plain SUM/COUNT of a column; the denominator is the total of the returned rows"
        return []
    conds = []
    if years:
        year_cols = [c for c in table.columns if c.type == "INTEGER" and c.min is not None and 1900 <= c.min <= c.max <= 2100]
        date_cols = [c for c in table.columns if c.type == "DATE"]
        if len(year_cols) == 1:
            conds.append(f'"{year_cols[0].name}" IN ({", ".join(years)})')
        elif not year_cols and len(date_cols) == 1:
            conds.append(f'substr("{date_cols[0].name}", 1, 4) IN ({", ".join(repr(y) for y in years)})')
        else:
            out["denominator_note"] = "the table has no single year or date column for the requested year"
            return []
    for f in filters:
        cs = [c for t, c in f["columns"] if t == table.name]
        if cs:
            conds.append(f'"{cs[0]}" = \'{f["value"].replace(chr(39), chr(39) * 2)}\'')
    agg, column = base
    target = "*" if column == "*" else f'"{column}"'
    check = f'SELECT {agg}({target}) FROM "{table.name}"' + (" WHERE " + " AND ".join(conds) if conds else "")
    result = sql_safety.execute(check, db_path, max_rows=1, timeout_s=timeout_s)
    if not result.ok or not result.rows:
        out["denominator_note"] = "the independent total query failed"
        return []
    expected = result.rows[0][0] or 0
    got = sum(r[cols.index(metric)] for r in rows if analytics.is_number(r[cols.index(metric)]))
    out["denominator_check"] = {"sql": check, "expected_total": expected, "rows_total": got}
    if abs(expected - got) > 1e-6 * max(1, abs(expected)):
        scope_text = " and ".join(conds) or "the whole table"
        return [f"the returned {metric} values add up to {got:,g}, but the total for the requested scope ({scope_text}) is "
                f"{expected:,g}; rows are missing or out of scope"]
    out["denominator_verified"] = True
    return []


# ---------------------------------------------------------------- ratios


def _match_column(term: str, columns: Sequence[str], candidates: Sequence[int], schema: Schema) -> list[int]:
    tokens = scope._tokens(term)
    originals = {c.name.lower(): c.original_name for t in schema.tables for c in t.columns}
    return [i for i in candidates if tokens and (scope._tokens(columns[i]) & tokens
                                                or scope._tokens(originals.get(columns[i].lower(), "")) & tokens)]


def check_ratio(question: str, sql: str, columns: Sequence[str], rows: Sequence[Sequence], schema: Schema,
                filters: list[dict], years: Sequence[str]) -> dict:
    out = {"issues": [], "ignored_columns": [], "ratios": []}
    a_term, b_term = ratio_terms(question)
    fraction = computed_fraction_columns(sql, columns, schema)
    out["ignored_columns"] = [columns[i] for i in fraction]
    keep = [i for i in range(len(columns)) if i not in fraction]
    cols, rws = scope.project(columns, rows, keep)
    roles = analytics.column_roles(cols, rws)
    a = _match_column(a_term, cols, roles["numeric"], schema)
    b = [i for i in _match_column(b_term, cols, roles["numeric"], schema) if i not in a]
    if len(a) != 1 or len(b) != 1:
        out["issues"].append(f"the result does not contain exactly one raw column for '{a_term}' and one for '{b_term}' "
                             "(model-computed ratios are not used)")
        return out
    a, b = a[0], b[0]
    out["issues"] += scope.scope_issues(sql, cols, rws, filters, years)
    labels = roles["label"] + roles["period"]
    for row in rws:
        name = " / ".join(str(row[i]) for i in labels) or "result"
        if not analytics.is_number(row[a]) or not analytics.is_number(row[b]):
            out["issues"].append(f"a value is missing for {name}")
        elif row[b] == 0:
            out["issues"].append(f"{cols[b]} is zero for {name}, so the ratio is undefined")
        else:
            out["ratios"].append({"label": name, "numerator_metric": cols[a], "denominator_metric": cols[b],
                                  "numerator": row[a], "denominator": row[b], "ratio": round(row[a] / row[b], 6),
                                  "percent": round(row[a] / row[b] * 100, 2)})
    if out["issues"]:
        out["ratios"] = []
    out["columns"], out["rows"] = cols, [list(r) for r in rws]
    out["numerator_metric"], out["denominator_metric"] = cols[a], cols[b]
    return out


def raw_value_note(question: str, sql: str, schema: Schema, kind: str) -> str:
    """Retry note: which raw columns to return (from the schema of the queried tables)."""

    numeric = [(t.name, c.name, c.original_name) for t in scope.tables_in(sql, schema) or schema.tables
               for c in t.columns if c.type in ("INTEGER", "REAL")]
    names = [n for _, n, _ in numeric]
    if kind == "ratio":
        a_term, b_term = ratio_terms(question)
        a = _match_column(a_term, names, range(len(names)), schema)
        b = _match_column(b_term, names, range(len(names)), schema)
        wanted = ", ".join(dict.fromkeys(names[i] for i in a + b)) or "the two metrics"
        return (f"Return the raw values of {wanted} as separate columns; do not divide them and do not compute a ratio "
                "or percentage in SQL (it is calculated afterwards).")
    hit = scope.relevant_metrics(question, names, range(len(names)), schema) if names else []
    wanted = ", ".join(dict.fromkeys(names[i] for i in hit)) if hit and len(hit) < len(names) else "the metric"
    return (f"Return the raw value of {wanted} for each category (SUM per category if there are several rows), "
            "one row per category for the requested scope; do not divide, do not use window functions and do not "
            "compute shares, percentages or ratios in SQL (they are calculated afterwards).")
