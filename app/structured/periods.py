"""Calendar periods named in questions (Phase 4.1, Tasks 2 and 3).

* ``quarters``: "first/second/third/fourth quarter of 2025", "Q1 2025",
  "Q3 of 2024", "2025 Q4", "1st quarter 2025" -> explicit ISO date ranges
  computed with ``calendar.monthrange`` (Q1 2025 = 2025-01-01 .. 2025-03-31;
  Q4 ends on 31 December, so ranges never cross a year boundary).
* ``requested_years``: the distinct years of a period-change question
  ("between 2024 and 2025") that the result must contain.
* ``hints``: notes appended to the NL-to-SQL prompt so the model filters on the
  explicit range and returns one row per requested period.
* ``quarter_filter_issue``: a generated query must contain both bounds of each
  named quarter as literals (otherwise it is sent back for repair).
"""

from __future__ import annotations

import calendar
import re
from datetime import date

_ORDINALS = {"first": 1, "1st": 1, "second": 2, "2nd": 2, "third": 3, "3rd": 3, "fourth": 4, "4th": 4, "last": 4}
_QUARTER_PATTERNS = (
    re.compile(r"\b(?P<n>first|second|third|fourth|last|1st|2nd|3rd|4th)\s+quarter\s+(?:of\s+)?(?:fy\s*)?(?P<y>(?:19|20)\d{2})\b", re.I),
    re.compile(r"\bq(?P<n>[1-4])\s*(?:of\s+)?(?:fy\s*)?(?P<y>(?:19|20)\d{2})\b", re.I),
    re.compile(r"\b(?P<y>(?:19|20)\d{2})\s*[-/ ]?\s*q(?P<n>[1-4])\b", re.I),
)
_CHANGE_RE = re.compile(r"\b(change|changed|growth|grew|increase|decrease|declin\w*|compare|compared|versus|vs\.?|"
                        r"year[- ]over[- ]year|yoy|difference)\b", re.I)


def quarter_range(year: int, quarter: int) -> tuple[str, str]:
    if quarter not in (1, 2, 3, 4):
        raise ValueError("quarter must be 1-4")
    first_month = 3 * (quarter - 1) + 1
    last_month = first_month + 2
    end_day = calendar.monthrange(year, last_month)[1]
    return date(year, first_month, 1).isoformat(), date(year, last_month, end_day).isoformat()


def quarters(question: str) -> list[dict]:
    found, seen = [], set()
    for pattern in _QUARTER_PATTERNS:
        for match in pattern.finditer(question):
            n = match.group("n").lower()
            quarter = int(n) if n.isdigit() else _ORDINALS[n]
            year = int(match.group("y"))
            if (year, quarter) in seen:
                continue
            seen.add((year, quarter))
            start, end = quarter_range(year, quarter)
            found.append({"label": f"Q{quarter} {year}", "start": start, "end": end})
    return found


def requested_years(question: str) -> list[str]:
    """Years a period-change question asks to compare (empty unless it names two or more)."""

    years = sorted(set(re.findall(r"\b(?:19|20)\d{2}\b", question)))
    return years if len(years) >= 2 and _CHANGE_RE.search(question) else []


def hints(question: str) -> list[str]:
    notes = []
    for q in quarters(question):
        notes.append(f"{q['label']} means dates from {q['start']} to {q['end']} inclusive: "
                     f"filter the date column with BETWEEN '{q['start']}' AND '{q['end']}'.")
    years = requested_years(question)
    if years:
        notes.append(f"Return one row per period for each of {', '.join(years)} (the period column and the value); "
                     "do not filter to a single year and do not compute the difference.")
    return notes


def quarter_filter_issue(question: str, sql: str) -> str | None:
    for q in quarters(question):
        if q["start"] not in sql or q["end"] not in sql:
            return (f"{q['label']} must be filtered with the explicit range BETWEEN '{q['start']}' AND '{q['end']}' "
                    "on the date column.")
    return None


def missing_years(question: str, columns, rows) -> list[str]:
    """Requested years not present in the result (values or column names)."""

    years = requested_years(question)
    if not years:
        return []
    returned = set()
    for row in rows:
        for v in row:
            if isinstance(v, int) and not isinstance(v, bool) and 1900 <= v <= 2100:
                returned.add(str(v))
            elif isinstance(v, str) and re.match(r"^(?:19|20)\d{2}(?:-\d{2}(?:-\d{2})?)?$", v.strip()):
                returned.add(v.strip()[:4])
    returned |= {y for c in columns for y in re.findall(r"(?:19|20)\d{2}", str(c))}
    return [y for y in years if y not in returned]
