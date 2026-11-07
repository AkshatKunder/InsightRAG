"""Deterministic, claim-level validation for generated RAG answers.

Each answer claim is checked against one evidence passage at a time. This keeps
numbers from unrelated passages from being combined into fabricated support.
"""
from __future__ import annotations

import re
from typing import Any

_NUMBER_RE = re.compile(r"(?<![\w.])(?P<currency>[$€£]|usd\s+|eur\s+|dollars?\s+)?(?P<value>\d[\d,]*(?:\.\d+)?)\s*(?P<scale>million|millions|billion|billions|thousand|thousands|bn|m|k)?(?P<percent>%|percent)?(?!\w)", re.I)
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9-]*")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_SCALES = {"billion": 1e9, "billions": 1e9, "bn": 1e9, "million": 1e6, "millions": 1e6, "m": 1e6, "thousand": 1e3, "thousands": 1e3, "k": 1e3}
_STOPWORDS = {"a", "an", "and", "approximately", "as", "at", "by", "for", "from", "in", "is", "it", "of", "on", "or", "the", "to", "was", "were", "with", "this", "that", "these", "those", "company", "group", "generated", "had", "have", "has"}
_ALIASES = {"employees": "employee", "employee": "employee", "workforce": "employee", "fte": "employee", "sales": "sales", "sale": "sales", "revenue": "sales", "countries": "country", "country": "country", "operations": "operation", "operating": "operation", "steelmaking": "steel-making", "remuneration": "remuneration", "compensation": "remuneration"}
_METRIC_TERMS = {"employee", "sales", "country", "operation", "steel-making", "remuneration"}
_NON_FACTUAL = re.compile(r"^(?:i\s+(?:could not|cannot|can't)|sorry\b|unknown\b)", re.I)


def _words(text: str) -> set[str]:
    return {_ALIASES.get(word.lower(), word.lower()) for word in _WORD_RE.findall(text) if word.lower() not in _STOPWORDS and len(word) > 1}


def _quantity(match: re.Match[str]) -> dict[str, Any]:
    scale = (match.group("scale") or "").lower()
    value = float(match.group("value").replace(",", ""))
    return {"raw": match.group(0).strip(), "value": value, "base_value": value * _SCALES.get(scale, 1), "scale": scale or None, "currency": bool((match.group("currency") or "").strip()), "percent": bool(match.group("percent"))}


def _quantities(text: str, exclude_years: bool = True) -> list[dict[str, Any]]:
    values = []
    for match in _NUMBER_RE.finditer(text):
        quantity = _quantity(match)
        # Reports often write the currency after the value and scale
        # (for example, "61,352 million dollars").
        if not quantity["currency"] and re.match(r"\s*(?:usd|eur|dollars?|euros?)\b", text[match.end():], re.I):
            quantity["currency"] = True
        if exclude_years and not quantity["scale"] and not quantity["currency"] and not quantity["percent"] and _YEAR_RE.fullmatch(match.group("value")):
            continue
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
    if not answer or not answer.strip() or _NON_FACTUAL.match(answer.strip()):
        return []
    claims = []
    for sentence in _SENTENCE_RE.split(answer.strip()):
        sentence = sentence.strip(" -\t")
        if not sentence or _NON_FACTUAL.match(sentence):
            continue
        quantities, words = _quantities(sentence), _words(sentence)
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
        for evidence in evidence_values:
            if claim["percent"] != evidence["percent"] or claim["currency"] != evidence["currency"]:
                continue
            tolerance = max(abs(claim["base_value"]), abs(evidence["base_value"]), 1.0) * 0.01
            if abs(claim["base_value"] - evidence["base_value"]) <= tolerance:
                return True, "Numerical value and scale are consistent."
    return False, "No passage value matches the claim value, unit type, and scale."


def _metric_match(claim_words: set[str], evidence_words: set[str]) -> tuple[bool, str]:
    overlap = claim_words & evidence_words
    if (claim_words & _METRIC_TERMS) and not (claim_words & _METRIC_TERMS & evidence_words):
        return False, "The passage does not mention the claim's metric or context."
    if overlap:
        return True, "Metric/context terms overlap in this passage."
    return False, "The passage has no meaningful claim-context overlap."


def _context_conflict(claim_words: set[str], evidence_words: set[str]) -> str | None:
    """Reject a narrower transaction metric masquerading as a company total."""
    transaction_terms = {"facility", "plant", "asset", "divestiture", "disposal"}
    if "sales" in claim_words and ("total" in claim_words or "company" in claim_words):
        if transaction_terms & evidence_words:
            return "The passage describes a facility or asset transaction, not total company sales."
    return None


def _evaluate_claim(claim: dict[str, Any], citation: dict[str, Any]) -> tuple[str, str]:
    text = str(citation.get("text") or "")
    if not text.strip():
        return "INSUFFICIENT_EVIDENCE", "The selected evidence passage is empty."
    claim_words, evidence_words = set(claim["metric"]), _words(text)
    conflict = _context_conflict(claim_words, evidence_words)
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
    value_ok, value_reason = _quantity_match(claim["quantities"], _quantities(text))
    if not value_ok:
        return "UNSUPPORTED", value_reason
    return "SUPPORTED", f"{metric_reason} {value_reason}"


def _result(claims: list[dict[str, Any]], reason: str | None = None) -> dict[str, Any]:
    supported = [claim for claim in claims if claim["status"] == "SUPPORTED"]
    unsupported = [claim for claim in claims if claim["status"] == "UNSUPPORTED"]
    insufficient = [claim for claim in claims if claim["status"] == "INSUFFICIENT_EVIDENCE"]
    valid = bool(claims) and not unsupported and not insufficient
    if reason is None:
        reason = "Every factual claim is supported by an individual evidence passage." if valid else ("One or more factual claims conflict with the selected evidence." if unsupported else "One or more factual claims lack sufficient individual evidence.")
    return {"is_valid": valid, "overall_status": "SUPPORTED" if valid else ("UNSUPPORTED" if unsupported else "INSUFFICIENT_EVIDENCE"), "reason": reason, "claims": claims, "supported_claims": supported, "unsupported_claims": unsupported, "insufficient_evidence_claims": insufficient, "unsupported_numbers": sorted({str(q["value"]) for claim in unsupported for q in claim["quantities"]})}


def validate_citation(answer: str, citations: list[dict[str, Any]] | None) -> dict[str, Any]:
    """Validate factual answer claims against individual citation passages.

    The legacy ``is_valid`` and ``reason`` keys remain available; structured claim
    results also expose source/page and explanations for each decision.
    """
    claims = extract_claims(answer)
    if not citations:
        results = [{**claim, "status": "INSUFFICIENT_EVIDENCE", "source": None, "page": None, "explanation": "No citations were selected."} for claim in claims]
        return _result(results, "No citations were selected.")
    if not claims:
        return _result([], "The answer contains no factual claims to validate.")
    results = []
    for claim in claims:
        evaluations = [_evaluate_claim(claim, citation) for citation in citations]
        matched = next((index for index, (status, _) in enumerate(evaluations) if status == "SUPPORTED"), None)
        if matched is None:
            matched = next((index for index, (status, _) in enumerate(evaluations) if status == "UNSUPPORTED"), None)
        if matched is None:
            status, explanation, citation = "INSUFFICIENT_EVIDENCE", "No individual evidence passage sufficiently supports this claim.", None
        else:
            status, explanation, citation = *evaluations[matched], citations[matched]
        results.append({**claim, "status": status, "source": citation.get("source", "Unknown source") if citation else None, "page": citation.get("page") if citation else None, "explanation": explanation})
    return _result(results)
