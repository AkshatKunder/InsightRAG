"""Deterministic business calculations on query results (Phase 4, Task 6).

The SQL query does the aggregation (SUM/COUNT/AVG/MIN/MAX). Derived figures
the question asks for (percentage change, year-over-year/period
comparisons, shares of a total) are computed here in Python from the
returned rows, never by the LLM. Inputs are checked first: values must be
numeric, periods must be distinct and ordered, a percentage change needs a
non-zero base, and two columns are only compared when their names carry the
same unit hint. Every figure keeps the row(s) it came from (traceability).
"""

from __future__ import annotations

import re
from collections.abc import Sequence

_PERIOD_NAME_RE = re.compile(r"(?:^|_)(year|yr|month|quarter|period|date|fy)(?:$|_)", re.I)
_UNIT_HINTS = {
    "usd": re.compile(r"usd|dollar|revenue|sales|income|cost|price|amount", re.I),
    "tonnes": re.compile(r"tonne|_t$|_kt$|_mt$|tons", re.I),
    "percent": re.compile(r"pct|percent|rate|share|margin", re.I),
    "count": re.compile(r"count|number|num_|employees|fte|headcount|qty|quantity", re.I),
}
CHANGE_INTENT_RE = re.compile(
    r"\b(change|changed|growth|grow|grew|increase|decrease|declin\w*|year[- ]over[- ]year|yoy|compared|versus|vs\.?|trend|period)\b", re.I)
SHARE_INTENT_RE = re.compile(r"\b(share|proportion|percentage of|% of|distribution|breakdown|split)\b", re.I)


def is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def unit_of(column: str) -> str | None:
    for unit, pattern in _UNIT_HINTS.items():
        if pattern.search(column):
            return unit
    return None


def same_unit(a: str, b: str) -> bool:
    return unit_of(a) == unit_of(b)


def percent_change(old: float, new: float) -> float | None:
    if not is_number(old) or not is_number(new) or old == 0:
        return None
    return round((new - old) / abs(old) * 100, 2)


def column_roles(columns: Sequence[str], rows: Sequence[Sequence]) -> dict:
    """Split result columns into period, label (categorical) and numeric columns."""

    numeric, period, label = [], [], []
    for index, name in enumerate(columns):
        values = [r[index] for r in rows if r[index] is not None]
        if not values:
            continue
        if _PERIOD_NAME_RE.search(name) or all(isinstance(v, str) and re.fullmatch(r"\d{4}(-\d{2}(-\d{2})?)?", v) for v in values):
            period.append(index)
        elif all(is_number(v) for v in values):
            if all(isinstance(v, int) and 1900 <= v <= 2100 for v in values) and len(set(values)) == len(values) and len(values) > 1:
                period.append(index)  # a column of distinct years
            else:
                numeric.append(index)
        else:
            label.append(index)
    return {"period": period, "numeric": numeric, "label": label}


def period_changes(columns: Sequence[str], rows: Sequence[Sequence]) -> list[dict]:
    """Change between consecutive periods for every numeric column (per label group if present)."""

    roles = column_roles(columns, rows)
    if len(roles["period"]) != 1 or not roles["numeric"]:
        return []
    p = roles["period"][0]
    group_index = roles["label"][0] if len(roles["label"]) == 1 else None
    groups: dict = {}
    for row in rows:
        groups.setdefault(row[group_index] if group_index is not None else None, []).append(row)
    out = []
    for group, items in groups.items():
        periods = [r[p] for r in items]
        if len(set(periods)) != len(periods) or len(items) < 2:
            continue  # duplicated periods: the rows are not a single time series
        items = sorted(items, key=lambda r: str(r[p]))
        for n in roles["numeric"]:
            for before, after in zip(items, items[1:]):
                if not is_number(before[n]) or not is_number(after[n]):
                    continue
                out.append({
                    "group": group, "metric": columns[n], "from_period": before[p], "to_period": after[p],
                    "from_value": before[n], "to_value": after[n],
                    "absolute_change": round(after[n] - before[n], 6),
                    "percent_change": percent_change(before[n], after[n]),
                    "source_rows": [list(before), list(after)],
                })
    return out


def shares(columns: Sequence[str], rows: Sequence[Sequence]) -> list[dict]:
    """Each label's share of the total of a single non-negative numeric column."""

    roles = column_roles(columns, rows)
    if len(roles["numeric"]) != 1 or len(roles["label"]) != 1:
        return []
    n, label = roles["numeric"][0], roles["label"][0]
    values = [r[n] for r in rows]
    if any(not is_number(v) or v < 0 for v in values) or not sum(values):
        return []
    total = sum(values)
    return [{"label": r[label], "value": r[n], "share_percent": round(r[n] / total * 100, 2)} for r in rows]


def summary_stats(columns: Sequence[str], rows: Sequence[Sequence]) -> dict:
    """SUM/COUNT/AVG/MIN/MAX of each numeric result column (deterministic, for traceability)."""

    roles = column_roles(columns, rows)
    stats = {}
    for n in roles["numeric"]:
        values = [r[n] for r in rows if is_number(r[n])]
        if values:
            stats[columns[n]] = {"count": len(values), "sum": round(sum(values), 6), "avg": round(sum(values) / len(values), 6),
                                 "min": min(values), "max": max(values)}
    return stats


def _same_metric_two_periods(a: str, b: str) -> bool:
    """``sales_2024`` vs ``sales_2025``: the same metric name with different period tokens."""

    period = re.compile(r"(?:19|20)\d{2}|q[1-4]|h[12]|prev(?:ious)?|prior|current|last|this")
    ta, tb = re.split(r"[_\W]+", a.lower()), re.split(r"[_\W]+", b.lower())
    pa, pb = [t for t in ta if period.fullmatch(t)], [t for t in tb if period.fullmatch(t)]
    rest_a, rest_b = [t for t in ta if not period.fullmatch(t)], [t for t in tb if not period.fullmatch(t)]
    return bool(pa) and bool(pb) and pa != pb and rest_a == rest_b and same_unit(a, b)


def analyse(question: str, columns: Sequence[str], rows: Sequence[Sequence]) -> dict:
    """Derived figures the question asks for, computed from the rows."""

    result = {}
    if CHANGE_INTENT_RE.search(question):
        changes = period_changes(columns, rows)
        if changes:
            result["period_changes"] = changes
        elif len(rows) == 1:
            # "Compare A with B" in one row: two same-unit numeric columns.
            numeric = [i for i, v in enumerate(rows[0]) if is_number(v)]
            if len(numeric) == 2 and _same_metric_two_periods(columns[numeric[0]], columns[numeric[1]]):
                a, b = numeric
                result["column_comparison"] = {
                    "from_metric": columns[a], "to_metric": columns[b], "from_value": rows[0][a], "to_value": rows[0][b],
                    "absolute_change": round(rows[0][b] - rows[0][a], 6), "percent_change": percent_change(rows[0][a], rows[0][b])}
    if SHARE_INTENT_RE.search(question):
        share = shares(columns, rows)
        if share:
            result["shares"] = share
    return result
