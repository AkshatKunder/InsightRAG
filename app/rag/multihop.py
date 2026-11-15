"""Multi-fact question decomposition and evidence merging (Phase 3, Task 4).

``decompose`` splits questions that ask for two facts into sub-queries, using
only generic sentence patterns:

1. "<question A>, and what/which/how/who/when <question B>?"
2. "was/were/is <A> higher|lower|greater|... or ... than <B>?" -> A, B
   (the question's years are added to each part).

Anything else is a single query, so ordinary questions keep the existing
single-query behaviour. ``merge_rankings`` interleaves the per-sub-query
rankings (first result of each sub-query first), de-duplicates pages and keeps
the existing metadata, so every sub-query contributes evidence before the
context limit is reached.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

_SECOND_QUESTION_RE = re.compile(r",?\s+and\s+(?=(?:what|which|how|who|when|where)\b)", re.I)
_COMPARISON_RE = re.compile(
    r"\b(?:was|were|is|are)\s+(?P<a>.+?)\s+(?:higher|lower|greater|larger|smaller|more|less)\s+or\s+"
    r"(?:higher|lower|greater|larger|smaller|more|less)\s+than\s+(?P<b>.+?)(?:,|\?|$)",
    re.I,
)
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_POSSESSIVE_RE = re.compile(r"^(?P<owner>[A-Z][\w-]*(?:\s+[A-Z][\w-]*)*)'s\s+", re.U)


def _clean(text: str) -> str:
    return " ".join(text.strip(" ,?.").split())


def decompose(question: str, max_parts: int = 3) -> list[str]:
    """Sub-queries for a multi-fact question, or ``[question]``."""

    text = " ".join(question.split())
    years = " ".join(dict.fromkeys(_YEAR_RE.findall(text)))

    comparison = _COMPARISON_RE.search(text)
    if comparison:
        a, b = _clean(comparison.group("a")), _clean(comparison.group("b"))
        owner = _POSSESSIVE_RE.match(a)
        if owner:
            b = re.sub(r"^(?:its|their)\s+", f"{owner.group('owner')}'s ", b, flags=re.I)
        parts = [p if _YEAR_RE.search(p) or not years else f"{p} {years}" for p in (a, b)]
        if all(len(p.split()) >= 2 for p in parts):
            return parts[:max_parts]

    pieces = [_clean(p) for p in _SECOND_QUESTION_RE.split(text) if _clean(p)]
    # A follow-up with too little content ("what was the main reason") depends on
    # the first part and would retrieve noise on its own: keep the question whole.
    if len(pieces) >= 2 and all(_content_words(p) >= 3 for p in pieces):
        return pieces[:max_parts]
    return [question]


_FUNCTION_WORDS = {
    "what", "which", "how", "who", "when", "where", "was", "were", "is", "are", "did", "do", "does",
    "the", "a", "an", "of", "in", "to", "and", "that", "this", "its", "their", "had", "has", "have", "main",
}


def _content_words(text: str) -> int:
    return sum(1 for word in re.findall(r"[A-Za-z0-9'-]+", text) if word.lower() not in _FUNCTION_WORDS)


def merge_rankings(rankings: Sequence[Sequence[Mapping]], limit: int, sub_queries: Sequence[str] | None = None) -> list[dict]:
    """Round-robin merge of ranked result lists, one result per (source, page)."""

    merged, seen = [], set()
    depth = max((len(r) for r in rankings), default=0)
    for position in range(depth):
        for index, ranking in enumerate(rankings):
            if position >= len(ranking):
                continue
            item = ranking[position]
            key = (item.get("source"), item.get("page"))
            if key in seen:
                continue
            seen.add(key)
            result = dict(item)
            if sub_queries:
                result["sub_query"] = sub_queries[index]
            merged.append(result)
            if len(merged) >= limit:
                return merged
    return merged
