"""Scope-aware context for chunks (Phase 3, Task 1).

Chunk ``text`` is left unchanged (BM25, the Qdrant index and citations keep
working on it). Instead, retrieved chunks get extra metadata that is only
attached where it can be extracted reliably:

``sections``  headed sub-sections inside or before the chunk on its page:
              ``[{"heading", "scope", "word_offset"}]``. A heading is a short
              proper-noun line directly above a "Performance for the year
              ended ..." table, e.g. "North America" or "AMNS India".
``scope``     reporting scope at the chunk start: ``consolidated`` (default),
              ``segment:<name>`` (a reportable segment named in the report's own
              segment list) or ``unit:<name>`` (another headed business unit:
              a joint venture, subsidiary or region).
``chart``     labelled values of a vector bar chart on the page, recovered from
              the PDF drawing (``app.rag.chart_extractor``), when the chunk
              contains those numbers.

``render_chunk`` turns this into excerpt text for the generator, with scope
markers inserted at the heading positions, so a model sees "[Scope: segment
North America]" right above a segment table instead of a bare number grid.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence

_TABLE_START_RE = re.compile(r"^\s*Performance for the year ended\b", re.I)
_SEGMENT_LIST_RE = re.compile(r"reportable segments[^:.]{0,80}:\s*(?P<list>[^.]{10,300})", re.I)
_CONNECTORS = {"and", "of", "&", "the", "de"}
_MAX_HEADING_WORDS = 6
CONSOLIDATED = "consolidated"


def find_segment_names(pages: Iterable[Mapping]) -> list[str]:
    """Reportable segment names as listed by the document itself ("... reportable segments: A, B and C")."""

    for page in pages:
        match = _SEGMENT_LIST_RE.search(" ".join(str(page.get("text", "")).split()))
        if not match:
            continue
        parts = [part.strip() for part in match.group("list").split(",") if part.strip()]
        if len(parts) < 2:
            continue
        last = re.split(r"\s+and\s+", parts[-1])
        if len(last) == 2:
            parts = parts[:-1] + last
        names = [re.sub(r"\s*\(.*?\)\s*", " ", p).strip(" .") for p in parts]
        names = [n for n in names if 0 < len(n.split()) <= _MAX_HEADING_WORDS]
        if len(names) >= 2:
            return names
    return []


def _is_heading(line: str) -> bool:
    words = line.split()
    if not 0 < len(words) <= _MAX_HEADING_WORDS or line.rstrip().endswith((".", ",", ":", ";")):
        return False
    if any(ch.isdigit() for ch in line):
        return False
    return all(word.lower() in _CONNECTORS or word[0].isupper() for word in words)


def _norm_name(name: str) -> str:
    name = re.sub(r"(?<![a-z])jvs?(?![a-z])", "joint ventures", name.lower().replace("&", "and"))
    return " ".join(name.split())


def classify_scope(heading: str | None, segment_names: Sequence[str]) -> str:
    if not heading:
        return CONSOLIDATED
    norm = _norm_name(heading)
    if any(norm == _norm_name(name) for name in segment_names):
        return f"segment:{heading}"
    return f"unit:{heading}"


def page_sections(text: str, segment_names: Sequence[str]) -> list[dict]:
    """Headed sections of a page: ``[{"heading", "scope", "word_index"}]`` in page order."""

    sections, words_before = [], 0
    lines = text.split("\n")
    for index, line in enumerate(lines):
        if _TABLE_START_RE.match(line) and index > 0:
            heading = lines[index - 1].strip()
            if _is_heading(heading):
                heading_words = len(heading.split())
                sections.append({
                    "heading": heading,
                    "scope": classify_scope(heading, segment_names),
                    "word_index": words_before - heading_words,
                })
        words_before += len(line.split())
    return sections


def _locate(chunk_words: list[str], page_words: list[str], start_from: int) -> int | None:
    if not chunk_words:
        return None
    n = len(chunk_words)
    for i in range(start_from, len(page_words) - n + 1):
        if page_words[i] == chunk_words[0] and page_words[i:i + n] == chunk_words:
            return i
    return None


def _chart_lines(values) -> list[str]:
    by_category: dict[str, list[str]] = {}
    for value in values:
        by_category.setdefault(value.category, []).append(f"{value.series} = {value.value}")
    return [f"{category}: {', '.join(items)}" for category, items in by_category.items()]


class ScopeIndex:
    """Scope and chart metadata for every chunk, keyed by (source, page, text)."""

    def __init__(self, chunks: Sequence[Mapping], pages: Sequence[Mapping], charts: Mapping[int, list] | None = None):
        self.segment_names = find_segment_names(pages)
        self._meta: dict[tuple, dict] = {}
        texts = {(p.get("source"), p.get("page")): str(p.get("text", "")) for p in pages}
        cursor: dict[tuple, int] = {}
        sections_cache: dict[tuple, list[dict]] = {}
        for (source, number), page_text in sorted(texts.items(), key=lambda kv: (str(kv[0][0]), kv[0][1] or 0)):
            sections = page_sections(page_text, self.segment_names)
            previous = sections_cache.get((source, (number or 0) - 1))
            # A segment review often runs across pages: the top of a page that has
            # headed tables itself continues the previous page's last section.
            if sections and previous and sections[0]["word_index"] > 0:
                sections.insert(0, {**previous[-1], "word_index": 0, "continued": True})
            sections_cache[(source, number)] = sections
        for chunk in chunks:
            key = (chunk.get("source"), chunk.get("page"))
            page_text = texts.get(key)
            if page_text is None:
                continue
            page_words = page_text.split()
            start = _locate(chunk["text"].split(), page_words, cursor.get(key, 0))
            if start is None:
                start = _locate(chunk["text"].split(), page_words, 0)
            if start is None:
                continue
            cursor[key] = start
            end = start + len(chunk["text"].split())
            sections = sections_cache[key]
            active = [s for s in sections if s["word_index"] <= start]
            inside = [
                {"heading": s["heading"], "scope": s["scope"], "word_offset": s["word_index"] - start}
                for s in sections if start < s["word_index"] < end and not s.get("continued")
            ]
            meta = {"scope": active[-1]["scope"] if active else CONSOLIDATED, "sections": inside}
            if active:
                meta["section"] = active[-1]["heading"]
            chart = (charts or {}).get(chunk.get("page"))
            if chart and sum(1 for v in chart if v.value in chunk["text"].split()) >= 2:
                meta["chart"] = _chart_lines(chart)
            self._meta[(chunk.get("source"), chunk.get("page"), chunk["text"])] = meta

    def metadata(self, chunk: Mapping) -> dict:
        return dict(self._meta.get((chunk.get("source"), chunk.get("page"), chunk.get("text")), {}))

    def enrich(self, results: Iterable[Mapping]) -> list[dict]:
        """Copies of ``results`` with scope metadata added (existing keys are kept)."""

        enriched = []
        for result in results:
            item = dict(result)
            for key, value in self.metadata(result).items():
                item.setdefault(key, value)
            enriched.append(item)
        return enriched


def scope_label(scope: str | None) -> str:
    if not scope or scope == CONSOLIDATED:
        return "consolidated (group) figures unless a row says otherwise"
    kind, _, name = scope.partition(":")
    if kind == "segment":
        return f"{name} segment only (not group total)"
    return f"{name} only (business unit, not group total)"


def render_chunk(chunk: Mapping) -> str:
    """Chunk text with scope markers and recovered chart labels, for the generator prompt."""

    words = str(chunk.get("text", "")).split()
    sections = sorted(chunk.get("sections") or [], key=lambda s: s["word_offset"])
    parts, previous = [], 0
    for section in sections:
        offset = max(0, min(section["word_offset"], len(words)))
        parts.append(" ".join(words[previous:offset]))
        parts.append(f"\n[Scope: {scope_label(section['scope'])}]\n")
        previous = offset
    parts.append(" ".join(words[previous:]))
    body = " ".join(p for p in parts if p).replace(" \n", "\n").replace("\n ", "\n").strip()
    header = []
    if chunk.get("scope") and chunk.get("scope") != CONSOLIDATED:
        header.append(f"[Scope at start: {scope_label(chunk.get('scope'))}]")
    if chunk.get("chart"):
        header.append("[Chart on this page, values by year: " + "; ".join(chunk["chart"]) + "]")
    return ("\n".join(header) + "\n" + body) if header else body
