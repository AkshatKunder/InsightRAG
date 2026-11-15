"""Deterministic, claim-level validation for generated RAG answers.

Each answer claim is checked against one evidence passage at a time. This keeps
numbers from unrelated passages from being combined into fabricated support.
"""
from __future__ import annotations

import math
import re
from typing import Any

from app.rag.scope_validation import apply_chart_units, apply_label_units, arithmetic_issue, is_date_part, scope_metric_conflict

_NUMBER_RE = re.compile(r"(?<![\w.])(?P<currency>[$€£]|usd\s+|eur\s+|dollars?\s+)?(?P<value>\d[\d,]*(?:\.\d+)?)\s*(?P<scale>million|millions|billion|billions|thousand|thousands|bn|m|k)?(?P<percent>%|percent)?(?!\w)", re.I)
_TABLE_UNIT_RE = re.compile(
    r"\(\s*(?:in\s+)?(?:(?P<prefix_currency>[$€£]|usd|eur|dollars?|euros?)\s+)?"
    r"(?P<scale>millions?|billions?|thousands?|bn|m|k)"
    r"(?:\s+of)?(?:\s+(?P<suffix_currency>u\.?s\.?\s*dollars?|usd|eur|dollars?|euros?))?"
    r"(?=\s*(?:,|\)))",
    re.I,
)
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9-]*")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_SCALES = {"billion": 1e9, "billions": 1e9, "bn": 1e9, "million": 1e6, "millions": 1e6, "m": 1e6, "thousand": 1e3, "thousands": 1e3, "k": 1e3}
_STOPWORDS = {"a", "an", "and", "approximately", "as", "at", "by", "for", "from", "in", "is", "it", "of", "on", "or", "the", "to", "was", "were", "with", "this", "that", "these", "those", "company", "group", "generated", "had", "have", "has"}
_ALIASES = {"employees": "employee", "employee": "employee", "workforce": "employee", "fte": "employee", "sales": "sales", "sale": "sales", "revenue": "sales", "countries": "country", "country": "country", "operations": "operation", "operating": "operation", "steelmaking": "steel-making", "remuneration": "remuneration", "compensation": "remuneration"}
_METRIC_TERMS = {"employee", "sales", "country", "operation", "steel-making", "remuneration"}
_NON_FACTUAL = re.compile(r"^(?:i\s+(?:could not|cannot|can't)|sorry\b|unknown\b)", re.I)
_ABSTENTION_RE = re.compile(
    r"\b(?:could not|cannot|can't|unable to)\s+(?:find|determine|locate|be determined)\b"
    r"|\bnot\s+(?:found|available|mentioned|provided)\s+in\s+the\s+(?:provided\s+)?(?:document|context)\b",
    re.I,
)
# A lowercase word right after a bare number marks a prose quantity such as
# "14 countries"; table cells are followed by other numbers or row labels.
_PROSE_FOLLOWER_RE = re.compile(r"[ \t]*[a-z]")
_COUNTED_NOUN_WINDOW = 4
_MAX_ROUNDING_FRACTION = 0.05
_TABLE_HEADER_DISTANCE = 600
_FOLLOWING_TABLE_CAPTION_DISTANCE = 3000
_TRANSACTION_RE = re.compile(
    r"\b(?:sale|disposal|divestiture|divestment)\s+of\s+(?:the\s+)?"
    r"(?:[a-z-]+\s+){0,4}(?:facility|plant|assets?|business|operations?)\b"
    r"|\b(?:facility|plant|assets?|business|operations?)\s+"
    r"(?:sale|disposal|divestiture|divestment)\b",
    re.I,
)


def _words(text: str) -> set[str]:
    return {_ALIASES.get(word.lower(), word.lower()) for word in _WORD_RE.findall(text) if word.lower() not in _STOPWORDS and len(word) > 1}


def _quantity(match: re.Match[str]) -> dict[str, Any]:
    scale = (match.group("scale") or "").lower()
    value = float(match.group("value").replace(",", ""))
    return {"raw": match.group(0).strip(), "value": value, "precision": _precision(match.group("value")), "base_value": value * _SCALES.get(scale, 1), "scale": scale or None, "currency": _currency_kind(match.group("currency")), "percent": bool(match.group("percent"))}


def _precision(raw_value: str) -> float:
    """Rounding step a written number implies: "61.4" -> 0.1, "125,000" -> 1000.

    Trailing zeros of an integer are ambiguous ("100" may be exact), so the
    implied step is capped at 5% of the value.
    """
    digits = raw_value.replace(",", "")
    if "." in digits:
        return 10.0 ** -len(digits.split(".", 1)[1])
    trailing_zeros = len(digits) - len(digits.rstrip("0"))
    if trailing_zeros == len(digits):
        return 1.0
    step = 10.0 ** trailing_zeros
    cap = abs(float(digits)) * _MAX_ROUNDING_FRACTION
    return max(1.0, min(step, 10.0 ** int(math.log10(cap)) if cap >= 1 else 1.0))


def _values_match(claim: dict[str, Any], evidence: dict[str, Any]) -> bool:
    """Values agree once rounded to the coarser of the two written precisions."""
    claim_step = claim["precision"] * claim["base_value"] / claim["value"] if claim["value"] else claim["precision"]
    evidence_step = evidence["precision"] * evidence["base_value"] / evidence["value"] if evidence["value"] else evidence["precision"]
    step = max(claim_step, evidence_step)
    return abs(claim["base_value"] - evidence["base_value"]) <= step / 2 + 1e-9 * max(abs(claim["base_value"]), 1.0)


def _currency_kind(value: str | None) -> str | None:
    """Normalize the limited currency forms recognized by the number parser."""
    currency = (value or "").strip().lower().replace(".", "")
    if not currency:
        return None
    if currency in {"$", "usd", "dollar", "dollars", "us dollar", "us dollars"}:
        return "USD"
    if currency in {"€", "eur", "euro", "euros"}:
        return "EUR"
    if currency == "£":
        return "GBP"
    return currency.upper()


def _table_unit_for_position(text: str, position: int) -> tuple[str | None, str | None]:
    """Find an explicit nearby table heading/caption for a bare table cell."""
    all_headers = list(_TABLE_UNIT_RE.finditer(text))
    headers = [match for match in all_headers if match.end() <= position]
    header = headers[-1] if headers and position - headers[-1].end() <= _TABLE_HEADER_DISTANCE else None
    if header is None:
        following = next((match for match in all_headers if match.start() >= position), None)
        line_start = text.rfind("\n", 0, position) + 1
        line_end = text.find("\n", position)
        line = text[line_start:] if line_end == -1 else text[line_start:line_end]
        if following and following.start() - position <= _FOLLOWING_TABLE_CAPTION_DISTANCE and ({"sales", "total"} & _words(line)):
            header = following
    if header is None:
        return None, None
    currency = _currency_kind(header.group("prefix_currency") or header.group("suffix_currency"))
    return (header.group("scale") or "").lower() or None, currency


def _following_terms(text: str, position: int) -> list[str]:
    """Return the normalized words directly after a number, stopping at the next number."""
    terms = []
    for token in re.finditer(r"\d[\d,.]*|[a-zA-Z][a-zA-Z0-9-]*", text[position:position + 120]):
        if token.group(0)[0].isdigit() or len(terms) >= _COUNTED_NOUN_WINDOW:
            break
        word = token.group(0).lower()
        terms.append(_ALIASES.get(word, word))
    return terms


def _counted_metric(quantity: dict[str, Any]) -> str | None:
    """Metric noun a bare count directly refers to, e.g. "14 countries" -> "country"."""
    if quantity["currency"] or quantity["percent"] or quantity["scale"]:
        return None
    following = quantity.get("following_terms") or []
    if following and following[0] in _METRIC_TERMS:
        return following[0]
    return None


def is_abstention(answer: str | None) -> bool:
    """Return whether the answer declines to answer instead of stating facts."""
    text = (answer or "").strip()
    if not text or not (_NON_FACTUAL.match(text) or _ABSTENTION_RE.search(text)):
        return False
    # A hedge that still states a figure ("could not find X, but sales were 5")
    # makes a factual claim and must be validated, not treated as abstention.
    return not _quantities(text)


def _quantities(text: str, exclude_years: bool = True, inherit_table_units: bool = False) -> list[dict[str, Any]]:
    values = []
    for match in _NUMBER_RE.finditer(text):
        quantity = _quantity(match)
        # Reports often write the currency after the value and scale
        # (for example, "61,352 million dollars").
        if not quantity["currency"]:
            suffix = re.match(r"\s*(usd|eur|dollars?|euros?)\b", text[match.end():], re.I)
            if suffix:
                quantity["currency"] = _currency_kind(suffix.group(1))
        if exclude_years and not quantity["scale"] and not quantity["currency"] and not quantity["percent"] and _YEAR_RE.fullmatch(match.group("value")):
            continue
        quantity["following_terms"] = _following_terms(text, match.end())
        quantity["start"] = match.start("value")
        if (
            inherit_table_units
            and not quantity["scale"]
            and not quantity["currency"]
            and not quantity["percent"]
            and not _PROSE_FOLLOWER_RE.match(text, match.end())
        ):
            inherited_scale, inherited_currency = _table_unit_for_position(text, match.start())
            if inherited_scale:
                quantity["scale"] = inherited_scale
                quantity["base_value"] = quantity["value"] * _SCALES.get(inherited_scale, 1)
            if inherited_currency:
                quantity["currency"] = inherited_currency
        values.append(quantity)
    return values


def extract_numbers(text: str) -> set[str]:
    """Return legacy-friendly normalized numeric values with scale conversions."""
    numbers: set[str] = set()
    for quantity in _quantities(text, exclude_years=False):
        numbers.add(f"{quantity['value']:g}")
        if quantity["base_value"] >= 1_000_000:
            numbers.add(f"{quantity['base_value'] / 1e9:g}")
            numbers.add(f"{quantity['base_value'] / 1e6:g}")
    return numbers


def _entities(text: str) -> set[str]:
    # Multi-word title-cased names and CamelCase names (for example,
    # ``ArcelorMittal``) are useful entity signals. Single title-cased sentence
    # starts such as "Total" are intentionally excluded.
    titled = {match.group(0).lower() for match in re.finditer(r"\b[A-Z][\w-]*(?:\s+[A-Z][\w-]*)+\b", text)}
    camel_case = {match.group(0).lower() for match in re.finditer(r"\b[A-Z][a-z]+[A-Z][A-Za-z-]*\b", text)}
    return titled | camel_case


def extract_claims(answer: str) -> list[dict[str, Any]]:
    """Extract declarative answer sentences and available comparison attributes."""
    if not answer or not answer.strip() or is_abstention(answer):
        return []
    claims = []
    for sentence in _SENTENCE_RE.split(answer.strip()):
        sentence = sentence.strip(" -\t")
        if not sentence:
            continue
        quantities, words = _quantities(sentence), _words(sentence)
        # "I could not find X" is not a claim, but "I could not find X, but
        # sales were $70 billion" still asserts a figure.
        if _NON_FACTUAL.match(sentence) and not quantities:
            continue
        if not quantities and len(words) < 2:
            continue
        claims.append({"claim_text": sentence, "entity": sorted(_entities(sentence)), "metric": sorted(words), "value": quantities[0]["value"] if quantities else None, "unit_or_scale": quantities[0]["scale"] if quantities else None, "quantities": quantities, "period": sorted(set(_YEAR_RE.findall(sentence))), "context": " ".join(sorted(words))})
    return claims


def _quantity_match(claim_values: list[dict[str, Any]], evidence_values: list[dict[str, Any]]) -> tuple[bool, str]:
    if not claim_values:
        return True, "No numerical value requires comparison."
    if not evidence_values:
        return False, "The passage contains no comparable numerical value."
    for claim in claim_values:
        counted = _counted_metric(claim)
        for evidence in evidence_values:
            if claim["percent"] != evidence["percent"] or claim["currency"] != evidence["currency"]:
                continue
            # "14 countries" is not supported by "14 research sites in 9 countries".
            if counted and counted not in evidence.get("following_terms", []):
                continue
            if _values_match(claim, evidence):
                return True, "Numerical value and scale are consistent."
    return False, "No passage value matches the claim value, unit type, and scale."


def _matching_evidence(
    claim_values: list[dict[str, Any]],
    evidence_values: list[dict[str, Any]],
    claim_text: str = "",
    evidence_text: str = "",
) -> list[dict[str, Any]]:
    """Every passage quantity that would satisfy ``_quantity_match`` for some claim value.

    Dates (years, days of a month) on either side are ignored: they are not the
    reported figure and would otherwise "match" unrelated passages.
    """
    matches = []
    for claim in claim_values:
        if claim_text and is_date_part(claim_text, claim.get("start", 0), claim.get("raw", "")):
            continue
        counted = _counted_metric(claim)
        for evidence in evidence_values:
            if evidence_text and is_date_part(evidence_text, evidence.get("start", 0), evidence.get("raw", "")):
                continue
            if claim["percent"] != evidence["percent"] or claim["currency"] != evidence["currency"]:
                continue
            if counted and counted not in evidence.get("following_terms", []):
                continue
            if _values_match(claim, evidence) and evidence not in matches:
                matches.append(evidence)
    return matches


def _supported_by(quantity: dict[str, Any], claim: dict[str, Any], citations: list[dict[str, Any]], question: str | None) -> bool:
    for citation in citations:
        text = str(citation.get("text") or "")
        evidence_values = apply_label_units(text, apply_chart_units(citation, _quantities(text, inherit_table_units=True)))
        matches = _matching_evidence([quantity], evidence_values, claim["claim_text"], text)
        if any(scope_metric_conflict(claim["claim_text"], set(claim["period"]), citation, text, e, question) is None for e in matches):
            return True
    return False


def _derived(quantity: dict[str, Any], others: list[dict[str, Any]]) -> bool:
    """Is ``quantity`` a difference, percentage change or ratio of two other claim values?"""
    target = abs(quantity["base_value"])
    tolerance = max(_precision(quantity.get("raw", "").strip("$€£ ()%").split()[0] if quantity.get("raw") else "1") * (quantity["base_value"] / quantity["value"] if quantity["value"] else 1), 1e-9) * 0.5
    for a in others:
        for b in others:
            if a is b or not a["base_value"]:
                continue
            candidates = [abs(a["base_value"] - b["base_value"]), abs(a["base_value"] / b["base_value"]) if b["base_value"] else None]
            if quantity["percent"]:
                candidates = [abs(a["base_value"] - b["base_value"]) / abs(b["base_value"]) * 100]
            for value in candidates:
                if value is not None and abs(value - target) <= tolerance + 0.02 * target:
                    return True
    return False


def _unsupported_value(claim: dict[str, Any], citations: list[dict[str, Any]], question: str | None) -> str | None:
    """Every reported number in a claim needs its own support (Phase 3); derived values are allowed."""
    values = [q for q in claim["quantities"] if not is_date_part(claim["claim_text"], q.get("start", 0), q.get("raw", ""))]
    if len(values) < 2:
        return None
    supported = [q for q in values if _supported_by(q, claim, citations, question)]
    for quantity in values:
        if quantity in supported or _derived(quantity, supported):
            continue
        return f"Unsupported value: {quantity['raw']} is not supported by any cited passage for this metric and scope."
    return None


def _metric_match(claim_words: set[str], evidence_words: set[str]) -> tuple[bool, str]:
    overlap = claim_words & evidence_words
    if (claim_words & _METRIC_TERMS) and not (claim_words & _METRIC_TERMS & evidence_words):
        return False, "The passage does not mention the claim's metric or context."
    if overlap:
        return True, "Metric/context terms overlap in this passage."
    return False, "The passage has no meaningful claim-context overlap."


def _context_conflict(claim_words: set[str], evidence_text: str) -> str | None:
    """Reject a narrower transaction metric masquerading as a company total."""
    if "sales" in claim_words and ("total" in claim_words or "company" in claim_words):
        if _TRANSACTION_RE.search(evidence_text):
            return "The passage describes a facility or asset transaction, not total company sales."
    return None


def _evaluate_claim(claim: dict[str, Any], citation: dict[str, Any], question: str | None = None) -> tuple[str, str]:
    text = str(citation.get("text") or "")
    if not text.strip():
        return "INSUFFICIENT_EVIDENCE", "The selected evidence passage is empty."
    claim_words, evidence_words = set(claim["metric"]), _words(text)
    conflict = _context_conflict(claim_words, text)
    if conflict:
        return "UNSUPPORTED", conflict
    metric_ok, metric_reason = _metric_match(claim_words, evidence_words)
    if not metric_ok:
        return "INSUFFICIENT_EVIDENCE", metric_reason
    claim_periods, evidence_periods = set(claim["period"]), set(_YEAR_RE.findall(text))
    if claim_periods and evidence_periods and not claim_periods & evidence_periods:
        return "UNSUPPORTED", "The passage reports a different reporting period."
    if claim["entity"]:
        evidence_entities = _entities(text)
        if evidence_entities and not set(claim["entity"]) & evidence_entities:
            return "UNSUPPORTED", "The passage identifies a different entity."
    evidence_values = apply_label_units(text, apply_chart_units(citation, _quantities(text, inherit_table_units=True)))
    value_ok, value_reason = _quantity_match(claim["quantities"], evidence_values)
    if not value_ok:
        return "UNSUPPORTED", value_reason
    if claim["quantities"]:
        # The same number must also be the same measure and reporting scope (Phase 3).
        matches = _matching_evidence(claim["quantities"], evidence_values, claim["claim_text"], text)
        conflicts = [
            scope_metric_conflict(claim["claim_text"], set(claim["period"]), citation, text, evidence, question)
            for evidence in matches
        ]
        if matches and all(conflicts):
            return "UNSUPPORTED", conflicts[0]
    return "SUPPORTED", f"{metric_reason} {value_reason}"


def _result(claims: list[dict[str, Any]], reason: str | None = None) -> dict[str, Any]:
    supported = [claim for claim in claims if claim["status"] == "SUPPORTED"]
    unsupported = [claim for claim in claims if claim["status"] == "UNSUPPORTED"]
    insufficient = [claim for claim in claims if claim["status"] == "INSUFFICIENT_EVIDENCE"]
    valid = bool(claims) and not unsupported and not insufficient
    if reason is None:
        reason = "Every factual claim is supported by an individual evidence passage." if valid else ("One or more factual claims conflict with the selected evidence." if unsupported else "One or more factual claims lack sufficient individual evidence.")
    return {"is_valid": valid, "overall_status": "SUPPORTED" if valid else ("UNSUPPORTED" if unsupported else "INSUFFICIENT_EVIDENCE"), "reason": reason, "claims": claims, "supported_claims": supported, "unsupported_claims": unsupported, "insufficient_evidence_claims": insufficient, "unsupported_numbers": sorted({str(q["value"]) for claim in unsupported for q in claim["quantities"]})}


def validate_citation(answer: str, citations: list[dict[str, Any]] | None, question: str | None = None) -> dict[str, Any]:
    """Validate factual answer claims against individual citation passages.

    The legacy ``is_valid`` and ``reason`` keys remain available; structured claim
    results also expose source/page and explanations for each decision.
    ``question`` (optional) lets a segment or measure named in the question
    justify citing segment-level or qualified figures. A supported claim whose
    own change arithmetic is inconsistent ("from A to B, up Y%") is UNSUPPORTED.
    """
    if is_abstention(answer):
        return {**_result([], "The answer declines to answer: the document context did not contain the information."), "overall_status": "ABSTAINED"}
    claims = extract_claims(answer)
    if not citations:
        results = [{**claim, "status": "INSUFFICIENT_EVIDENCE", "source": None, "page": None, "explanation": "No citations were selected."} for claim in claims]
        return _result(results, "No citations were selected.")
    if not claims:
        return _result([], "The answer contains no factual claims to validate.")
    results = []
    for claim in claims:
        evaluations = [_evaluate_claim(claim, citation, question) for citation in citations]
        matched = next((index for index, (status, _) in enumerate(evaluations) if status == "SUPPORTED"), None)
        if matched is None:
            matched = next((index for index, (status, _) in enumerate(evaluations) if status == "UNSUPPORTED"), None)
        if matched is None:
            status, explanation, citation = "INSUFFICIENT_EVIDENCE", "No individual evidence passage sufficiently supports this claim.", None
        else:
            status, explanation, citation = *evaluations[matched], citations[matched]
        if status == "SUPPORTED":
            issue = arithmetic_issue(claim["claim_text"], claim["quantities"]) or _unsupported_value(claim, citations, question)
            if issue:
                status, explanation = "UNSUPPORTED", issue
        results.append({**claim, "status": status, "source": citation.get("source", "Unknown source") if citation else None, "page": citation.get("page") if citation else None, "explanation": explanation})
    return _result(results)
