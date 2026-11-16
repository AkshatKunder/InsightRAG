"""Chart-ready data for query results (Phase 4, Task 7).

``choose_chart`` returns a spec ``{"type", "x", "y", "data", "reason"}`` or
``{"type": None, "reason": ...}`` when a chart would mislead:

* **line**: one period column (dates, months or a column of distinct years)
  with at least 3 ordered points and numeric series;
* **pie**: only when the question asks for a share/breakdown, there is one
  non-negative numeric column whose values are parts of a whole (the SQL
  uses SUM or COUNT), and 2-8 categories;
* **bar**: one categorical column (2-30 categories) and numeric series.

No chart for single values, more than 30 categories, mixed units across
series, or rows without a label/period column.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from app.structured.analytics import SHARE_INTENT_RE, column_roles, same_unit

MAX_BAR_CATEGORIES = 30
MAX_PIE_SLICES = 8


def choose_chart(question: str, sql: str, columns: Sequence[str], rows: Sequence[Sequence],
                 parts_of_whole: bool = False) -> dict:
    """``parts_of_whole``: the rows were validated as the complete raw parts of a total (Phase 4.3 shares)."""

    if len(rows) < 2:
        return {"type": None, "reason": "A single row is shown as a value, not a chart."}
    roles = column_roles(columns, rows)
    numeric = roles["numeric"]
    if not numeric:
        return {"type": None, "reason": "No numeric column to plot."}
    if any(not same_unit(columns[numeric[0]], columns[i]) for i in numeric[1:]):
        return {"type": None, "reason": "Numeric columns have different units; one chart would mislead."}
    y = [columns[i] for i in numeric]

    if len(roles["period"]) == 1 and not roles["label"]:
        p = roles["period"][0]
        ordered = sorted(rows, key=lambda r: str(r[p]))
        if len(ordered) >= 3 and len({r[p] for r in ordered}) == len(ordered):
            return {"type": "line", "x": columns[p], "y": y, "reason": "Values over ordered periods.",
                    "data": [{columns[p]: r[p], **{columns[i]: r[i] for i in numeric}} for r in ordered]}

    if len(roles["label"]) == 1 and not roles["period"]:
        x = roles["label"][0]
        categories = len({r[x] for r in rows})
        if categories != len(rows):
            return {"type": None, "reason": "Category labels repeat; the rows are not one series per category."}
        data = [{columns[x]: r[x], **{columns[i]: r[i] for i in numeric}} for r in rows]
        additive = parts_of_whole or bool(re.search(r"\b(sum|count)\s*\(", sql or "", re.I))
        if (SHARE_INTENT_RE.search(question) and len(numeric) == 1 and additive
                and 2 <= categories <= MAX_PIE_SLICES and all(r[numeric[0]] is not None and r[numeric[0]] >= 0 for r in rows)
                and sum(r[numeric[0]] for r in rows) > 0):
            return {"type": "pie", "x": columns[x], "y": y, "data": data, "reason": "Parts of a whole across few categories."}
        if 2 <= categories <= MAX_BAR_CATEGORIES:
            return {"type": "bar", "x": columns[x], "y": y, "data": data, "reason": "Comparison across categories."}
        return {"type": None, "reason": f"{categories} categories are too many for a readable chart."}

    return {"type": None, "reason": "The result has no single category or period axis."}
