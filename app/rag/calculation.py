"""Programmatic arithmetic from cited source values (Phase 3.1, Task 2).

For two question types the difference is computed in code, never taken from
the LLM:

* **comparison**: "was <A> higher or lower than <B> (in <year>)?" (the
  ``multihop`` comparison pattern): difference = A - B for the same year;
* **change**: "how did <metric> change between <Y1> and <Y2>?": absolute and
  percentage change from the <Y1> and <Y2> values of the same metric.

Operands come only from evidence and must agree on unit and period:

* recovered chart values (``chart`` metadata, label + series year), or
* table rows whose label contains every metric word and no other measure
  qualifier (production/shipments/capacity/...), mapped to years by the
  nearest "<YYYY> <YYYY>" header, outside segment/unit sections unless the
  question names that scope (``scope_validation``).

If an operand cannot be found unambiguously, nothing is computed (the
answer is left to validation/abstention). Otherwise the answer's
contradicted sentences are dropped, wrong derived numbers and direction words
are corrected, the deterministic result sentence is added, and the operand
passages are added to the citations.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from app.rag.multihop import _COMPARISON_RE, _POSSESSIVE_RE
from app.rag.query_rewriter import _CHANGE_RE
from app.rag.scope_validation import _chart_entry, _mentions, _qualifiers, _scope_name_at, is_contradiction

_STOP = {"the", "its", "their", "of", "in", "a", "an", "total", "group", "company", "arcelormittal", "and"}
_TOKEN_RE = re.compile(r"\(?-?\d[\d,]*(?:\.\d+)?\)?%?|[A-Za-z][\w&'*/-]*|\([^)]{1,40}\)")
_NUM_RE = re.compile(r"^\(?(-?\d[\d,]*(?:\.\d+)?)\)?%?$")
_YEAR_RE = re.compile(r"^(?:19|20)\d{2}$")
_UNIT_RE = re.compile(r"\((?P<scale>thousand|million|billion)s?\s+(?P<unit>tonnes|tons|USD|\$|dollars)\)", re.I)
_SCALE = {"thousand": 1e3, "million": 1e6, "billion": 1e9}


def _words(text: str) -> set[str]:
    out = set()
    for word in re.findall(r"[a-z]+", text.lower()):
        if word in _STOP:
            continue
        out.add(word[:-1] if word.endswith("s") and len(word) > 3 else word)
    return out


def _metric_words(text: str) -> set[str]:
    text = _POSSESSIVE_RE.sub("", text.strip())
    text = re.sub(r"\b(?:19|20)\d{2}\b", " ", text)
    return _words(text)


def _decimals(raw: str) -> int:
    match = re.search(r"\.(\d+)", raw)
    return len(match.group(1)) if match else 0


def _unit_of(label: str) -> tuple[str, float] | None:
    match = _UNIT_RE.search(label)
    if not match:
        return None
    unit = match.group("unit").lower()
    unit = "USD" if unit in ("usd", "$", "dollars") else "tonnes"
    return f"{match.group('scale').lower()} {unit}", _SCALE[match.group("scale").lower()]


def _from_chart(chunk: Mapping, metric: set[str], year: str) -> dict | None:
    for line in chunk.get("chart") or []:
        category, _, values = line.partition(": ")
        if not metric <= _words(category) or (_qualifiers(category) - _qualifiers(" ".join(metric))):
            continue
        unit = _unit_of(category)
        for item in values.split(", "):
            series, _, raw = item.partition(" = ")
            if series.strip() == year and unit and _chart_entry(chunk, raw.strip()):
                return {"value": float(raw.replace(",", "")), "raw": raw.strip(), "unit": unit[0], "label": category}
    return None


def _from_table(chunk: Mapping, metric: set[str], year: str, question: str) -> dict | None:
    text = str(chunk.get("text", ""))
    tokens = list(_TOKEN_RE.finditer(text))
    for index, token in enumerate(tokens):
        if not _NUM_RE.match(token.group(0)) or index == 0 or _NUM_RE.match(tokens[index - 1].group(0)):
            continue
        label_tokens = []
        j = index - 1
        while j >= 0 and len(label_tokens) < 8 and not _NUM_RE.match(tokens[j].group(0)):
            label_tokens.insert(0, tokens[j].group(0))
            j -= 1
        label = " ".join(label_tokens)
        if not metric <= _words(label) or (_qualifiers(label) - _qualifiers(" ".join(metric))):
            continue
        unit = _unit_of(label)
        if not unit:
            continue
        scope = _scope_name_at(chunk, text, token.start())
        if scope and not _mentions(scope, question):
            continue
        # Year header: the closest run of >= 2 consecutive years before the row.
        header = []
        for k in range(j, -1, -1):
            if _YEAR_RE.match(tokens[k].group(0)):
                run = [tokens[k].group(0)]
                m = k - 1
                while m >= 0 and _YEAR_RE.match(tokens[m].group(0)):
                    run.insert(0, tokens[m].group(0))
                    m -= 1
                if len(run) >= 2:
                    header = run
                    break
        values = []
        n = index
        while n < len(tokens) and _NUM_RE.match(tokens[n].group(0)) and len(values) < len(header):
            values.append(tokens[n].group(0))
            n += 1
        if year in header and header.index(year) < len(values):
            raw = values[header.index(year)]
            if raw.startswith("(") or raw.endswith("%"):
                continue
            return {"value": float(_NUM_RE.match(raw).group(1).replace(",", "")), "raw": raw, "unit": unit[0], "label": label}
    return None


def find_operand(metric_text: str, year: str, chunks: Sequence[Mapping], question: str) -> dict | None:
    metric = _metric_words(metric_text)
    if not metric:
        return None
    for chunk in chunks:
        found = _from_chart(chunk, metric, year) or _from_table(chunk, metric, year, question)
        if found:
            return {**found, "page": chunk.get("page"), "source": chunk.get("source"), "chunk": chunk}
    return None


def _fmt(value: float, decimals: int, unit: str) -> str:
    number = f"{abs(value):,.{decimals}f}"
    if unit.endswith("USD"):
        return f"${number} {unit.split()[0]}"
    return f"{number} {unit}"


def plan(question: str) -> dict | None:
    """What to compute for this question, or None."""

    comparison = _COMPARISON_RE.search(question)
    years = re.findall(r"\b(?:19|20)\d{2}\b", question)
    if comparison and len(set(years)) == 1:
        a, b = comparison.group("a").strip(" ,?"), comparison.group("b").strip(" ,?")
        return {"kind": "comparison", "a": a, "b": b, "year": years[0]}
    change = _CHANGE_RE.search(question)
    if change:
        return {"kind": "change", "metric": change.group("metric"), "y1": change.group("y1"), "y2": change.group("y2")}
    return None


def calculate(question: str, chunks: Sequence[Mapping]) -> dict | None:
    """Operands and result from evidence; None when not applicable or not reliably found."""

    spec = plan(question)
    if spec is None:
        return None
    if spec["kind"] == "comparison":
        a = find_operand(spec["a"], spec["year"], chunks, question)
        b = find_operand(spec["b"], spec["year"], chunks, question)
        if not a or not b or a["unit"] != b["unit"]:
            return None
        decimals = max(_decimals(a["raw"]), _decimals(b["raw"]))
        diff = round(a["value"] - b["value"], decimals)
        word = "higher" if diff > 0 else "lower" if diff < 0 else "equal"
        name_a = _POSSESSIVE_RE.sub("", spec["a"]).strip()
        name_b = re.sub(r"^(?:its|their)\s+", "", _POSSESSIVE_RE.sub("", spec["b"]).strip(), flags=re.I)
        sentence = (f"{name_a[0].upper()}{name_a[1:]} ({_fmt(a['value'], decimals, a['unit'])}) was {word} than {name_b} "
                    f"({_fmt(b['value'], decimals, b['unit'])}) in {spec['year']}, a difference of {_fmt(diff, decimals, a['unit'])}.")
        return {**spec, "operands": [a, b], "difference": diff, "direction": word, "unit": a["unit"],
                "decimals": decimals, "sentence": sentence}
    v1 = find_operand(spec["metric"], spec["y1"], chunks, question)
    v2 = find_operand(spec["metric"], spec["y2"], chunks, question)
    if not v1 or not v2 or v1["unit"] != v2["unit"] or v1["value"] == 0:
        return None
    decimals = max(_decimals(v1["raw"]), _decimals(v2["raw"]))
    diff = round(v2["value"] - v1["value"], decimals)
    pct = round(diff / v1["value"] * 100, 1)
    word = "increased" if diff > 0 else "decreased" if diff < 0 else "was unchanged"
    metric = spec["metric"].strip()
    sentence = (f"{metric[0].upper()}{metric[1:]} {word} by {_fmt(diff, decimals, v1['unit'])} ({abs(pct):.1f}%), "
                f"from {_fmt(v1['value'], decimals, v1['unit'])} in {spec['y1']} to {_fmt(v2['value'], decimals, v2['unit'])} in {spec['y2']}.")
    return {**spec, "operands": [v1, v2], "difference": diff, "percent": pct,
            "direction": "increase" if diff > 0 else "decrease" if diff < 0 else "unchanged",
            "unit": v1["unit"], "decimals": decimals, "sentence": sentence}


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_ABSTAIN_RE = re.compile(r"could not find|cannot find|can't find|unable to", re.I)


def apply(answer: str, validation: Mapping, citations: list[dict], result: Mapping) -> tuple[str, list[dict]]:
    """Answer with contradicted sentences removed and the computed result stated; citations with operand passages."""

    contradicted = [c.get("claim_text", "") for c in validation.get("claims") or []
                    if c.get("status") == "UNSUPPORTED" and is_contradiction(c.get("explanation"))]
    sentences = [s for s in _SENTENCE_SPLIT_RE.split(answer or "") if s.strip()]
    kept = [s for s in sentences if not any(c and c.strip() in s for c in contradicted) and not _ABSTAIN_RE.search(s)]
    # Sentences that state their own (LLM) difference or change are replaced by the computed one.
    # Only phrases that state a figure ("by 0.6 million", "difference of 2,528", "by 26.5%"),
    # not other uses of "by" ("partly offset by a $0.4 billion expense").
    derived = re.compile(r"\b(?:by|difference(?:\s+(?:of|is|was))?|change of|increase of|decrease of)\s+[$€£]?\s*\(?\d", re.I)
    kept = [s for s in kept if not derived.search(s)]
    final = " ".join(kept + [result["sentence"]]).strip()

    out = list(citations)
    seen = {(c.get("source"), c.get("page")) for c in out}
    for operand in result["operands"]:
        key = (operand["source"], operand["page"])
        if key not in seen:
            seen.add(key)
            out.append(dict(operand["chunk"]))
    return final, out
