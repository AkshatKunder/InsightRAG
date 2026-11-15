"""Metric, scope and arithmetic checks for claim validation (Phase 3, Tasks 3 and 5).

Used by ``app.rag.citation_validator``. A claim number that also appears in a
passage is only accepted when the passage's figure is the same *measure*
and *reporting scope*:

* **metric qualifier**: the row label next to the number (or the recovered
  chart label) must not name a different measure, e.g. "shipments",
  "capacity", "ROM"/"aggregate" when the claim is about production;
* **scope**: a figure inside a segment or business-unit section (chunk
  metadata from ``app.rag.document_context``, or a "<Name> Performance for the
  year ended" heading / "<Name> segment" phrase in the passage) only supports a
  claim or question that names that segment or unit;
* **year**: a chart value's year (series) must match the claim year.

``arithmetic_issue`` checks a claim's own change statements: direction,
"by X" absolute change and "(Y%)" percentage against "from A to B".
"""

from __future__ import annotations

import re
from typing import Any

_TOKEN_RE = re.compile(r"\(?-?[\d][\d,]*(?:\.\d+)?\)?%?|[A-Za-z][A-Za-z&'-]*")
_NUMERIC_TOKEN_RE = re.compile(r"^\(?-?[\d][\d,]*(?:\.\d+)?\)?%?$")
_LABEL_WORDS = 10

# Measures that are easily confused; each entry is a set of surface forms.
_QUALIFIERS = {
    "production": {"production", "produced", "produce"},
    "shipments": {"shipments", "shipment", "shipped"},
    "capacity": {"capacity"},
    "rom": {"rom", "run-of-mine"},
    "aggregate": {"aggregate"},
    "consumption": {"consumption", "consumed"},
    "reserves": {"reserves", "reserve"},
}
_HEADED_SECTION_RE = re.compile(r"((?:[A-Z][\w&'-]*\s+(?:(?:and|of|&)\s+)?){1,5})Performance for the year ended")
_SEGMENT_PHRASE_RE = re.compile(r"\b(?:in|of|for)\s+the\s+((?:[A-Z][\w&'-]*\s+){1,4})segment\b")


def _qualifiers(text: str) -> set[str]:
    words = {w.lower() for w in re.findall(r"[A-Za-z][A-Za-z-]*", text)}
    if "run of mine" in text.lower():
        words.add("run-of-mine")
    return {name for name, forms in _QUALIFIERS.items() if words & forms}


def row_label(text: str, position: int) -> str:
    """Words labelling the number at ``position``: the words before its run of numbers."""

    tokens = list(_TOKEN_RE.finditer(text[:position]))
    index = len(tokens) - 1
    while index >= 0 and _NUMERIC_TOKEN_RE.match(tokens[index].group(0)):
        index -= 1
    words = []
    while index >= 0 and len(words) < _LABEL_WORDS:
        token = tokens[index].group(0)
        # Dates inside a label ("for the year ended December 31, 2025 totaled") are part of it.
        if _NUMERIC_TOKEN_RE.match(token) and not _is_date_token(token, tokens, index):
            break
        words.append(token)
        index -= 1
    return " ".join(reversed(words))


_MONTHS = {"january", "february", "march", "april", "may", "june", "july", "august",
           "september", "october", "november", "december"}


def _is_date_token(token: str, tokens, index: int) -> bool:
    digits = token.strip("(),%")
    if re.fullmatch(r"(19|20)\d{2}", digits):
        return True
    return index > 0 and tokens[index - 1].group(0).lower() in _MONTHS and digits.isdigit() and int(digits) <= 31


def is_date_part(text: str, position: int, raw: str) -> bool:
    """Is the number at ``position`` a year or a day of a month (not a reported value)?"""

    digits = raw.strip("$€£ (),%").replace(",", "")
    if re.fullmatch(r"(19|20)\d{2}", digits):
        return True
    before = re.findall(r"[A-Za-z]+", text[max(0, position - 12):position])
    return bool(before) and before[-1].lower() in _MONTHS and digits.isdigit() and int(digits) <= 31


def _sentence(text: str, position: int) -> str:
    start = max(text.rfind(". ", 0, position), text.rfind("\n", 0, position))
    end_candidates = [i for i in (text.find(". ", position), text.find("\n", position)) if i != -1]
    end = min(end_candidates) if end_candidates else len(text)
    return text[start + 1:end]


def _chart_entry(citation: dict[str, Any], raw_value: str) -> tuple[str, str] | None:
    """(category, series) of a chart value recovered by document_context, if unique."""

    matches = []
    for line in citation.get("chart") or []:
        category, _, values = line.partition(": ")
        for item in values.split(", "):
            series, _, value = item.partition(" = ")
            if value.strip() == raw_value:
                matches.append((category, series.strip()))
    return matches[0] if len(matches) == 1 else None


_CHART_UNIT_RE = re.compile(r"\((?P<scale>thousand|million|billion)s?\s+(?P<unit>[A-Za-z$]+)\)", re.I)
_SCALE = {"thousand": 1e3, "million": 1e6, "billion": 1e9}


def apply_chart_units(citation: dict[str, Any], evidence_values: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Give chart values the unit of their own chart row instead of a neighbouring caption's.

    pypdf places a chart's values next to its row captions, so a value such as
    55.6 (crude steel, million tonnes) would otherwise inherit "(billion USD)".
    """

    if not citation.get("chart"):
        return evidence_values
    for evidence in evidence_values:
        entry = _chart_entry(citation, str(evidence.get("raw", "")).strip())
        unit = _CHART_UNIT_RE.search(entry[0]) if entry else None
        if not unit:
            continue
        scale = unit.group("scale").lower()
        evidence["scale"] = scale
        evidence["base_value"] = evidence["value"] * _SCALE[scale]
        currency = unit.group("unit").upper()
        evidence["currency"] = "USD" if currency in ("USD", "$", "DOLLARS") else None
    return evidence_values


_LABEL_UNIT_RE = re.compile(r"\b(?P<scale>thousand|million|billion)s?\s+(?:of\s+)?(?:metric\s+)?(?:tonnes|tons)\b", re.I)


def apply_label_units(text: str, evidence_values: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Scale bare table cells by a tonnage unit in their row label: "(thousand tonnes) 7,755" -> 7.755 million."""

    for evidence in evidence_values:
        if evidence.get("scale") or evidence.get("currency") or evidence.get("percent"):
            continue
        # Only table cells: a prose number ("14 countries") is followed by a lowercase word.
        end = evidence.get("start", 0) + len(str(evidence.get("raw", "")).strip())
        if re.match(r"[ \t,]*[a-z]", text[end:end + 3]):
            continue
        # The unit must belong to this row: within the label's last few words.
        unit = _LABEL_UNIT_RE.search(" ".join(row_label(text, evidence.get("start", 0)).split()[-6:]))
        if unit:
            scale = unit.group("scale").lower()
            evidence["scale"] = scale
            evidence["base_value"] = evidence["value"] * _SCALE[scale]
    return evidence_values


def _scope_name_at(citation: dict[str, Any], text: str, position: int) -> str | None:
    """Name of the segment/unit whose section contains ``position`` (None = consolidated/unknown)."""

    if "scope" in citation or "sections" in citation:
        word_index = len(text[:position].split())
        scope = citation.get("scope")
        for section in sorted(citation.get("sections") or [], key=lambda s: s["word_offset"]):
            if section["word_offset"] <= word_index:
                scope = section["scope"]
        if scope and scope != "consolidated":
            return scope.partition(":")[2] or None
        return None
    headed = [m for m in _HEADED_SECTION_RE.finditer(text) if m.end() <= position]
    if headed:
        return " ".join(headed[-1].group(1).split())
    return None


def _norm(text: str) -> str:
    text = re.sub(r"(?<![a-z])jvs?(?![a-z])", "joint ventures", text.lower().replace("&", "and"))
    return " ".join(re.findall(r"[a-z0-9]+", text))


def _mentions(name: str, *texts: str) -> bool:
    needle = _norm(name)
    return bool(needle) and any(f" {needle} " in f" {_norm(t)} " for t in texts if t)


def scope_metric_conflict(
    claim_text: str,
    claim_periods: set[str],
    citation: dict[str, Any],
    text: str,
    evidence: dict[str, Any],
    question: str | None = None,
) -> str | None:
    """Reason why the matched passage number is a different metric/scope/year, or None."""

    position = evidence.get("start", 0)
    label = row_label(text, position)
    chart = _chart_entry(citation, evidence.get("raw", ""))
    if chart:
        label = chart[0]
        if claim_periods and chart[1] not in claim_periods:
            return f"Scope/metric mismatch: the chart value {evidence.get('raw')} is for {chart[1]}, not {', '.join(sorted(claim_periods))}."

    asked = _qualifiers(claim_text) | _qualifiers(question or "")
    found = _qualifiers(label)
    extra = found - asked
    if asked and extra:
        return f"Scope/metric mismatch: the passage figure is labelled '{label}', a different measure ({', '.join(sorted(extra))})."

    scope = _scope_name_at(citation, text, position)
    if scope is None:
        # Narrative figures name their segment in the sentence ("... in the Mining segment totaled 35.3 ...").
        phrase = _SEGMENT_PHRASE_RE.search(label + " ") or _SEGMENT_PHRASE_RE.search(_sentence(text, position) + " ")
        scope = " ".join(phrase.group(1).split()) if phrase else None
    if scope and not _mentions(scope, claim_text, question or ""):
        return f"Scope/metric mismatch: the passage figure reports the {scope} segment/unit, not the group-level figure the claim states."
    return None


# ---------------------------------------------------------------------------
# Arithmetic (Task 5)
# ---------------------------------------------------------------------------

_INCREASE_RE = re.compile(r"\b(increas\w*|rose|risen|grew|grow\w*|higher|up)\b", re.I)
_DECREASE_RE = re.compile(r"\b(decreas\w*|declin\w*|fell|fall\w*|dropped|lower|down|reduc\w*)\b", re.I)


def _half_step(quantity: dict[str, Any]) -> float:
    raw = str(quantity.get("raw", ""))
    digits = re.search(r"\d[\d,]*(?:\.(\d+))?", raw)
    decimals = len(digits.group(1)) if digits and digits.group(1) else 0
    scale = quantity["base_value"] / quantity["value"] if quantity.get("value") else 1.0
    return 0.5 * 10 ** -decimals * scale


_COMPARATIVE_RE = re.compile(r"\b(?P<word>higher|greater|larger|more|lower|smaller|less)\s+than\b", re.I)


def _comparison_issue(sentence: str, quantities: list[dict[str, Any]]) -> str | None:
    """"A (x) was lower than B (y)" must agree with the order of x and y."""

    match = _COMPARATIVE_RE.search(sentence)
    if not match:
        return None
    values = [q for q in quantities if not q["percent"]]
    before = [q for q in values if q.get("start", -1) < match.start()]
    after = [q for q in values if q.get("start", -1) > match.end()]
    if len(before) != 1 or len(after) != 1 or before[0].get("currency") != after[0].get("currency"):
        return None
    a, b = before[0], after[0]
    tol = _half_step(a) + _half_step(b)
    higher = match.group("word").lower() in ("higher", "greater", "larger", "more")
    if higher and a["base_value"] < b["base_value"] - tol:
        return f"Arithmetic error: says {a['raw']} is {match.group('word')} than {b['raw']}."
    if not higher and a["base_value"] > b["base_value"] + tol:
        return f"Arithmetic error: says {a['raw']} is {match.group('word')} than {b['raw']}."
    return None


def arithmetic_issue(sentence: str, quantities: list[dict[str, Any]]) -> str | None:
    """Check "from A to B" statements against the stated direction, absolute and percent change,
    and "A higher/lower than B" statements against the two values."""

    quantities = [q for q in quantities if not is_date_part(sentence, q.get("start", 0), str(q.get("raw", "")))]
    comparison = _comparison_issue(sentence, quantities)
    if comparison:
        return comparison
    match = re.search(r"\bfrom\b(?P<a>.*?)\bto\b(?P<b>.*)", sentence, re.I | re.S)
    if not match:
        return None
    a_start, b_start = match.start("a"), match.start("b")
    a = next((q for q in quantities if a_start <= q.get("start", -1) < match.end("a") and not q["percent"]), None)
    b = next((q for q in quantities if q.get("start", -1) >= b_start and not q["percent"]), None)
    if a is None or b is None or a["base_value"] == 0:
        return None
    before, after = a["base_value"], b["base_value"]
    head = sentence[:match.start()]
    tol = _half_step(a) + _half_step(b)
    increase, decrease = bool(_INCREASE_RE.search(head)), bool(_DECREASE_RE.search(head))
    if increase != decrease:
        if increase and after < before - tol:
            return f"Arithmetic error: states an increase but the values go from {a['raw']} to {b['raw']}."
        if decrease and after > before + tol:
            return f"Arithmetic error: states a decrease but the values go from {a['raw']} to {b['raw']}."
    change = after - before
    for q in quantities:
        if q.get("start", -1) >= match.start():
            continue
        if q["percent"]:
            low = (abs(change) - tol) / (abs(before) + _half_step(a))
            high = (abs(change) + tol) / max(abs(before) - _half_step(a), 1e-9)
            stated = q["value"] / 100
            if not (low - _half_step(q) / 100 - 1e-9 <= stated <= high + _half_step(q) / 100 + 1e-9):
                return f"Arithmetic error: the stated change {q['raw']} does not match {a['raw']} -> {b['raw']} ({change / before:+.1%})."
        elif q.get("currency") == a.get("currency") and re.search(r"\bby\s*[$€£]?\s*$", head[:q["start"]][-14:], re.I):
            if abs(abs(change) - q["base_value"]) > tol + _half_step(q) + 1e-9 * abs(q["base_value"]):
                return f"Arithmetic error: the stated change {q['raw']} does not equal the difference between {a['raw']} and {b['raw']}."
    return None


# Explanation prefixes of the Phase 3 checks that are verified contradictions
# (the evidence states something different), as opposed to "no matching value
# found". "Unsupported value:" is a not-found result: it marks the claim
# UNSUPPORTED but does not trigger revision or abstention, because it also hits
# correct answers written in shorthand (measured: Phase 3 Q021, "$61,352 / 125,554").
CONTRADICTION_PREFIXES = ("Scope/metric mismatch:", "Arithmetic error:")


def is_contradiction(explanation: str | None) -> bool:
    return bool(explanation) and explanation.startswith(CONTRADICTION_PREFIXES)
