"""Targeted query rewrites (Phase 3.1, Tasks 1 and 3).

Adds at most one extra retrieval query for two generic question patterns:

* **change**: "How did <owner>'s <metric> change between <Y1> and <Y2>?" (also
  "from <Y1> to <Y2>"). Annual reports put year-over-year changes in tables
  headed "<Y2> <Y1> change % change", so the rewrite is
  "<metric> <Y2> <Y1> change % change". Its results are ordered
  group-level first (``consolidated_first``) because the same table layout
  is repeated per segment. Not used when the question names a segment, region
  or business unit, so segment questions keep their own evidence.
* **definition**: "What is <Entity>'s primary/main/core business?" or "What
  does <Entity> do?" -> "<Entity> company overview", which finds the section
  where a report defines the company.

Everything else returns no rewrite.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

_CHANGE_RE = re.compile(
    r"\bhow\s+(?:did|has|have|was|were)\s+(?:(?P<owner>[A-Z][\w&.-]*(?:\s+[A-Z][\w&.-]*)*)'s\s+)?(?:the\s+)?"
    r"(?P<metric>[a-z][\w\s&()-]*?)\s+(?:change|changed|develop|evolve)\w*\s+"
    r"(?:between|from)\s+(?P<y1>(?:19|20)\d{2})\s+(?:and|to)\s+(?P<y2>(?:19|20)\d{2})",
    re.I,
)
_DEFINITION_RE = re.compile(
    r"^\s*(?:what\s+is\s+(?P<a>[A-Z][\w&.-]*(?:\s+[A-Z][\w&.-]*)*)'s\s+(?:primary|main|core|principal)\s+"
    r"(?:business|activity|activities|line of business)"
    r"|what\s+does\s+(?P<b>[A-Z][\w&.-]*(?:\s+[A-Z][\w&.-]*)*)\s+do)\b",
    re.I,
)
_SEGMENT_WORDS_RE = re.compile(r"\b(?:segment|segments|region|regional|subsidiary|joint venture|division|unit)\b", re.I)


def _names_scope(question: str, scope_names: Sequence[str]) -> bool:
    if _SEGMENT_WORDS_RE.search(question):
        return True
    lower = question.lower()
    return any(re.search(rf"\b{re.escape(name.lower())}\b", lower) for name in scope_names if name)


def rewrite_queries(question: str, scope_names: Sequence[str] = ()) -> list[dict]:
    """Extra queries: ``[{"query", "kind", "consolidated_first"}]`` (possibly empty)."""

    change = _CHANGE_RE.search(question)
    if change and not _names_scope(question, scope_names):
        metric = " ".join(change.group("metric").split())
        return [{
            "query": f"{metric} {change.group('y2')} {change.group('y1')} change % change",
            "kind": "change",
            "consolidated_first": True,
        }]
    definition = _DEFINITION_RE.search(question)
    if definition:
        entity = definition.group("a") or definition.group("b")
        return [{"query": f"{entity} company overview", "kind": "definition", "consolidated_first": False}]
    return []


def consolidated_first(results: Sequence[dict]) -> list[dict]:
    """Stable reorder: chunks whose reporting scope is group-level come before segment/unit chunks."""

    def segment_scoped(item: dict) -> bool:
        return bool(item.get("scope")) and item.get("scope") != "consolidated"

    return sorted(results, key=segment_scoped)
