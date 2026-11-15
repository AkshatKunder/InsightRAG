"""Deterministic checks of generated answers against benchmark ground truth.

Pure functions only (no models). This module is deliberately independent of
``app.rag.citation_validator``: the system's own validator is under test, so
answers are graded with a separate parser and separate rules.

A numeric expected fact is **matched** only when all of these hold for one
quantity in the answer:

* value   - rounding-consistent with the expected value after converting
            scale words (an answer of "$61.4 billion" matches 61,352 USD
            million because 61.352 rounds to 61.4), and stated with at
            least two significant figures, so "$3 billion" does not match 3,152;
* unit    - same unit class (currency, mass, count, percent); a bare number
            with no unit does not match a currency or tonnage fact;
* period  - the nearest year in the same sentence, if any, is the fact's
            year (an answer that pairs the value with another year fails);
* entity  - for segment facts, the segment name appears in the answer.

Non-numeric facts (names, lists, dates) use normalised term matching.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from datetime import date

# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------

# Fact unit (benchmark) -> (unit class, multiplier to base units)
FACT_UNITS = {
    "USD million": ("currency", 1e6),
    "USD billion": ("currency", 1e9),
    "USD": ("currency", 1.0),
    "USD per FTE employee": ("currency", 1.0),
    "million tonnes": ("mass", 1e6),
    "FTE employees": ("count", 1.0),
    "segments": ("count", 1.0),
    "percent": ("percent", 1.0),
}

SCALES = {"thousand": 1e3, "k": 1e3, "million": 1e6, "mn": 1e6, "m": 1e6, "billion": 1e9, "bn": 1e9, "b": 1e9}

_NUMBER = r"(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
_QUANTITY_RE = re.compile(
    r"(?P<neg>[-−–]|\()?\s*"
    r"(?P<cur>US\$|\$|USD\s?|€|EUR\s?)?\s?"
    + _NUMBER
    + r"(?:\s?(?P<scale>thousand|million|billion|mn|bn|[kmb])\b)?"
    r"(?:\s?(?P<unit>%|percent\b|per\s?cent\b|million\s+tonnes|metric\s+tonnes|metric\s+tons|tonnes|tons|mt\b|"
    r"full[- ]time\s+equivalent|fte\b|employees|people|workers|usd\b|us\s+dollars|dollars|segments))?",
    re.I,
)
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_SENTENCE_RE = re.compile(r"(?<=[.!?;])\s+(?=[A-Z(])|\n+")
_WORD_NUMBERS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
}


def _unit_class(cur: str | None, unit: str | None, scale: str | None) -> tuple[str | None, float]:
    """Return (unit class, implicit multiplier) for a parsed quantity."""

    unit_l = (unit or "").lower().strip()
    if cur and cur.strip() in ("€", "EUR"):
        return "currency_eur", 1.0
    if cur or unit_l in ("usd", "us dollars", "dollars"):
        return "currency", 1.0
    if unit_l in ("%", "percent", "per cent", "percent"):
        return "percent", 1.0
    if unit_l.startswith("percent") or unit_l.startswith("per"):
        return "percent", 1.0
    if unit_l == "mt":
        return "mass", 1e6  # "54.0 Mt" = million tonnes
    if "tonnes" in unit_l or "tons" in unit_l:
        return "mass", 1e6 if unit_l.startswith("million") else 1.0
    if unit_l and (unit_l.startswith("full") or unit_l in ("fte", "employees", "people", "workers")):
        return "count", 1.0
    if unit_l == "segments":
        return "count", 1.0
    return None, 1.0


def _significant_figures(raw: str) -> int:
    digits = raw.replace(",", "").replace(".", "").lstrip("0")
    return len(digits) or 1


def parse_quantities(text: str, exclude_years: bool = True) -> list[dict]:
    """All numeric quantities in ``text`` with value, scale, unit class and position.

    Years (1900-2099 with no unit, scale or currency) are skipped when
    ``exclude_years`` is true. ``base_value`` is the value in base units
    (USD, tonnes, people, percent points).
    """

    quantities = []
    for match in _QUANTITY_RE.finditer(text or ""):
        raw = match.group("num")
        value = float(raw.replace(",", ""))
        cur, scale, unit = match.group("cur"), (match.group("scale") or "").lower() or None, match.group("unit")
        if scale in ("m", "b", "k") and not cur:
            # A bare "m"/"b"/"k" after a plain number is too ambiguous (e.g. "5 m" metres).
            scale = None
        if exclude_years and not cur and not scale and not unit and "," not in raw and "." not in raw and _YEAR_RE.fullmatch(raw):
            continue
        # A number directly followed by a letter or digit is part of a token (e.g. "Q4", "4.2a").
        end = match.end("num") if not (scale or unit) else match.end()
        if end < len(text) and text[end:end + 1].isalpha() and not (scale or unit):
            continue
        unit_class, implicit = _unit_class(cur, unit, scale)
        multiplier = SCALES.get(scale, 1.0) * implicit
        if unit_class == "mass" and scale == "million" and implicit == 1e6:
            multiplier = 1e6  # "54.0 million tonnes": scale already applied
        decimals = len(raw.split(".")[1]) if "." in raw else 0
        negative = bool(match.group("neg")) and match.group("neg") in ("-", "−", "–", "(")
        quantities.append({
            "raw": match.group(0).strip(),
            "value": value,
            "negative": negative,
            "scale": scale,
            "unit_class": unit_class,
            "base_value": value * multiplier,
            "half_step": 0.5 * 10 ** (-decimals) * multiplier,
            "significant_figures": _significant_figures(raw),
            "start": match.start("num"),
            "end": match.end(),
        })
    return quantities


def values_match(quantity: Mapping, expected_base: float, tolerance_fraction: float = 0.0, min_sig_figs: int = 2) -> bool:
    """Rounding-consistent comparison on absolute values, optionally with a relative tolerance."""

    if quantity["significant_figures"] < min_sig_figs and abs(expected_base) >= 10:
        return False
    diff = abs(abs(quantity["base_value"]) - abs(expected_base))
    allowed = max(quantity["half_step"], tolerance_fraction * abs(expected_base)) + 1e-9 * max(abs(expected_base), 1.0)
    return diff <= allowed


# ---------------------------------------------------------------------------
# Periods, metrics and entities
# ---------------------------------------------------------------------------


def _sentences_with_spans(text: str) -> list[tuple[int, int]]:
    spans, start = [], 0
    for match in _SENTENCE_RE.finditer(text):
        spans.append((start, match.start()))
        start = match.end()
    spans.append((start, len(text)))
    return spans


_FOLLOWING_YEAR_RE = re.compile(
    r"^[\s,]*(?:\)\s*)?(?:in|for|during|of|at|as of|at the end of|by the end of|on)?\s*"
    r"(?:the\s+)?(?:fiscal\s+)?(?:year\s+)?(?:(?:ended\s+)?December\s+31,?\s*)?((?:19|20)\d{2})\b",
    re.I,
)


def year_for_quantity(text: str, start: int, end: int) -> str | None:
    """The reporting year a quantity is stated for, within its sentence.

    A year directly after the quantity ("$61,352 million in 2025") wins;
    otherwise the closest preceding year in the sentence ("In 2025, sales
    were ..."); otherwise the closest following year; otherwise None.
    """

    for s_start, s_end in _sentences_with_spans(text):
        if s_start <= start <= s_end:
            following = _FOLLOWING_YEAR_RE.match(text[end:s_end])
            if following:
                return following.group(1)
            before = [m for m in _YEAR_RE.finditer(text, s_start, start)]
            if before:
                return before[-1].group(0)
            after = [m for m in _YEAR_RE.finditer(text, end, s_end)]
            return after[0].group(0) if after else None
    return None


def nearest_year(text: str, position: int) -> str | None:
    """Backward-compatible helper: the year for a quantity starting at ``position``."""

    match = _QUANTITY_RE.search(text, position)
    end = match.end() if match and match.start("num") == position else position
    return year_for_quantity(text, position, end)


def fact_year(period: str | None) -> str | None:
    return period[:4] if period and re.match(r"^\d{4}", str(period)) else None


METRIC_KEYWORDS = {
    "total_sales": ("sales", "revenue"),
    "segment_sales": ("sales", "revenue"),
    "operating_income": ("operating income",),
    "cost_of_sales": ("cost of sales",),
    "net_income_attributable_to_parent": ("net income",),
    "capital_expenditures": ("capital expenditure", "capex"),
    "net_debt": ("net debt",),
    "employees_fte": ("employee", "fte", "workforce", "full-time equivalent", "headcount"),
    "segment_employees_fte": ("employee", "fte", "workforce", "full-time equivalent", "headcount"),
    "crude_steel_production": ("crude steel",),
    "iron_ore_production": ("iron ore",),
    "steel_shipments": ("shipment",),
    "rd_expense": ("research", "r&d", "development"),
}


def metric_named(metric: str, answer: str) -> bool | None:
    keywords = METRIC_KEYWORDS.get(metric)
    if not keywords:
        return None
    lower = answer.lower()
    return any(keyword in lower for keyword in keywords)


def normalise_terms(text: str) -> str:
    text = text.lower().replace("&", " and ").replace("’", "'")
    text = re.sub(r"\bjvs\b", "joint ventures", text)
    text = re.sub(r"\bjv\b", "joint venture", text)
    text = re.sub(r"\b([a-z])\.([a-z])\.", r"\1\2", text)  # n.v. -> nv, s.a. -> sa
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    text = re.sub(r"\b(company|nv|sa|inc|ltd|plc)\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def term_present(term: str, answer: str) -> bool:
    needle = normalise_terms(term)
    haystack = f" {normalise_terms(answer)} "
    return bool(needle) and f" {needle} " in haystack


_MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august",
           "september", "october", "november", "december"]


def date_present(iso: str, answer: str) -> bool:
    try:
        d = date.fromisoformat(iso)
    except ValueError:
        return False
    month = _MONTHS[d.month - 1]
    lower = answer.lower()
    patterns = [
        rf"\b{month}\s+{d.day}(st|nd|rd|th)?,?\s+{d.year}\b",
        rf"\b{d.day}(st|nd|rd|th)?\s+(of\s+)?{month},?\s+{d.year}\b",
        rf"\b{d.isoformat()}\b",
    ]
    return any(re.search(p, lower) for p in patterns)


def integer_present(value: int, answer: str) -> bool:
    if re.search(rf"(?<![\d.,]){value}(?![\d.,]\d)", answer):
        return True
    words = [w for w, n in _WORD_NUMBERS.items() if n == value]
    return any(re.search(rf"\b{w}\b", answer, re.I) for w in words)


# ---------------------------------------------------------------------------
# Fact checks
# ---------------------------------------------------------------------------


def check_numeric_fact(fact: Mapping, answer: str, question: str = "") -> dict:
    """Check one numeric fact with a unit. Returns per-dimension results."""

    unit_class, multiplier = FACT_UNITS.get(fact.get("unit"), (None, 1.0))
    expected_base = float(fact["value"]) * multiplier
    year = fact_year(fact.get("period"))
    quantities = parse_quantities(answer)

    value_hits = [q for q in quantities if values_match(q, expected_base)]
    # A bare number in the fact's own scale ("61,352" for 61,352 USD million) states
    # the right value but no unit: counted as value_found with unit_ok False.
    value_hits += [
        q for q in quantities
        if q["unit_class"] is None and q["scale"] is None and q not in value_hits
        and values_match({**q, "base_value": q["value"], "half_step": q["half_step"]}, float(fact["value"]))
    ]
    unit_hits = [q for q in value_hits if q["unit_class"] == unit_class]
    result = {
        "metric": fact.get("metric"),
        "expected_value": fact["value"],
        "expected_unit": fact.get("unit"),
        "period": fact.get("period"),
        "segment": fact.get("segment"),
        "value_found": bool(value_hits),
        "unit_ok": bool(unit_hits),
        "period_ok": None,
        "entity_ok": None,
        "metric_named": metric_named(fact.get("metric", ""), answer),
        "matched_text": None,
        "matched": False,
        "reason": "",
    }
    if not value_hits:
        result["reason"] = "expected value not stated (rounding-consistent, >= 2 significant figures)"
        return result
    if not unit_hits:
        result["reason"] = f"value stated without the expected unit class ({unit_class})"
        return result

    period_hits = []
    for q in unit_hits:
        near = year_for_quantity(answer, q["start"], q["end"])
        if year is None or near is None or near == year:
            period_hits.append((q, near))
    if year is not None:
        result["period_ok"] = bool(period_hits)
        if period_hits and all(near is None for _, near in period_hits):
            # No year next to the value: acceptable when the question fixes the period.
            result["period_ok"] = year in question or year in answer
    if not period_hits or result["period_ok"] is False:
        result["reason"] = f"value paired with a different or missing period (expected {fact.get('period')})"
        return result

    if fact.get("segment"):
        result["entity_ok"] = term_present(fact["segment"], answer)
        if not result["entity_ok"]:
            result["reason"] = f"segment {fact['segment']!r} not named"
            return result

    matched = period_hits[0][0]
    result["matched"] = True
    result["matched_text"] = matched["raw"]
    # Rounded when the stated value differs from the exact figure (e.g. $61.4 billion for 61,352 million).
    result["rounded"] = abs(abs(matched["base_value"]) - abs(expected_base)) > 1e-9 * max(abs(expected_base), 1.0)
    result["reason"] = "value, unit, period and entity consistent"
    return result


def check_fact(fact: Mapping, answer: str, question: str = "") -> dict:
    """Dispatch on the fact's value type."""

    value = fact.get("value")
    base = {"metric": fact.get("metric"), "expected_value": value, "numeric": False}
    if isinstance(value, (int, float)) and not isinstance(value, bool) and fact.get("unit") in FACT_UNITS:
        if fact.get("unit") == "segments":
            # The count must be stated as a count of segments ("six reportable segments"),
            # not picked up from list numbering such as "6. Others".
            words = "|".join([str(int(value))] + [w for w, n in _WORD_NUMBERS.items() if n == int(value)])
            ok = bool(re.search(rf"\b(?:{words})\s+(?:[\w-]+\s+){{0,3}}segments?\b", answer, re.I))
            return {**base, "numeric": True, "matched": ok, "value_found": ok, "unit_ok": ok,
                    "reason": "count stated" if ok else "count not stated"}
        return {**check_numeric_fact(fact, answer, question), "numeric": True}
    if isinstance(value, int) and not isinstance(value, bool):  # e.g. a year
        ok = integer_present(value, answer)
        return {**base, "matched": ok, "reason": "value stated" if ok else "value not stated"}
    if isinstance(value, list):
        missing = [item for item in value if not _item_present(item, answer)]
        return {**base, "matched": not missing, "missing_items": missing,
                "reason": "all items named" if not missing else f"missing: {', '.join(missing)}"}
    if isinstance(value, str):
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
            ok = date_present(value, answer)
        elif fact.get("metric") == "primary_business":
            ok = all(term_present(word, answer) for word in ("steel", "mining"))
        else:
            ok = term_present(value, answer)
        return {**base, "matched": ok, "reason": "stated" if ok else f"{value!r} not stated"}
    return {**base, "matched": False, "reason": "unsupported fact type"}


def _item_present(item: str, answer: str) -> bool:
    if term_present(item, answer):
        return True
    # A standalone company name such as "Arcelor" must not match inside "ArcelorMittal".
    return False


# ---------------------------------------------------------------------------
# Comparisons and calculations
# ---------------------------------------------------------------------------

_INCREASE = re.compile(r"\b(increas\w*|rose|risen|rise|grew|grow\w*|gain\w*|higher|up\b|improv\w*|exceed\w*|more than)\b", re.I)
_DECREASE = re.compile(r"\b(decreas\w*|declin\w*|fell|fall\w*|dropped|drop\b|lower|down\b|reduc\w*|less than|shrank)\b", re.I)
_UNCHANGED = re.compile(r"\b(unchanged|flat|stable|no change)\b", re.I)


def detect_direction(answer: str) -> str | None:
    """'increase', 'decrease', 'unchanged', 'mixed' or None, from direction words."""

    inc, dec, flat = bool(_INCREASE.search(answer)), bool(_DECREASE.search(answer)), bool(_UNCHANGED.search(answer))
    if inc and dec:
        return "mixed"
    if inc:
        return "increase"
    if dec:
        return "decrease"
    if flat:
        return "unchanged"
    return None


def check_change_value(expected: float, unit: str, answer: str, percent: bool = False) -> bool:
    """Is the absolute (or percent) change stated, ignoring its sign?"""

    if percent:
        return any(q["unit_class"] == "percent" and values_match(q, abs(expected), min_sig_figs=1) for q in parse_quantities(answer))
    unit_class, multiplier = FACT_UNITS.get(unit, (None, 1.0))
    expected_base = abs(expected) * multiplier
    return any(
        q["unit_class"] == unit_class and values_match(q, expected_base, min_sig_figs=1)
        for q in parse_quantities(answer)
    )


def check_comparison(comparison: Mapping, answer: str) -> dict:
    direction = detect_direction(answer)
    expected = comparison.get("direction")
    return {
        "expected_direction": expected,
        "detected_direction": direction,
        "direction_ok": direction == expected,
        "absolute_change_ok": check_change_value(comparison["absolute_change"], comparison.get("unit"), answer)
        if comparison.get("absolute_change") is not None else None,
        "percent_change_ok": check_change_value(comparison["percent_change"], "percent", answer, percent=True)
        if comparison.get("percent_change") is not None else None,
    }


def check_calculation(calculation: Mapping, answer: str, question: str = "") -> dict:
    """Is the derived result stated (within the benchmark tolerance or rounding)?"""

    unit_class, multiplier = FACT_UNITS.get(calculation.get("unit"), (None, 1.0))
    expected_base = float(calculation["result"]) * multiplier
    tolerance = float(calculation.get("tolerance_percent") or 0) / 100
    hits = [
        q for q in parse_quantities(answer)
        if values_match(q, expected_base, tolerance, min_sig_figs=1)
        and (q["unit_class"] in (unit_class, None))
    ]
    result = {"expected_result": calculation["result"], "unit": calculation.get("unit"), "result_ok": bool(hits),
              "matched_text": hits[0]["raw"] if hits else None, "direction_ok": None}
    if re.search(r"\bhigher or lower\b|\bmore or less\b", question, re.I):
        expected = "increase" if float(calculation["result"]) > 0 else "decrease"
        detected = detect_direction(answer)
        result["expected_direction"], result["detected_direction"] = expected, detected
        result["direction_ok"] = detected == expected
    return result


# ---------------------------------------------------------------------------
# Grounding of answer numbers
# ---------------------------------------------------------------------------


def number_in_text(quantity: Mapping, text: str) -> bool:
    """Does ``text`` state a number consistent with ``quantity`` (any scale)?

    Compares in the quantity's own scale and in common table scales
    (raw, thousands, millions, billions), so "$61.4 billion" is grounded by a
    table cell "61,352" under "(in $ millions)".
    """

    targets = [abs(quantity["base_value"])]
    for q in parse_quantities(text, exclude_years=False):
        for multiplier in (1.0, 1e3, 1e6, 1e9):
            candidate = abs(q["value"]) * (multiplier if q["scale"] is None and q["unit_class"] != "mass" else 1.0)
            if q["scale"] is not None or q["unit_class"] == "mass":
                candidate = abs(q["base_value"])
            for target in targets:
                step = max(quantity["half_step"], 0.5 * 10 ** -(len(q["raw"].split(".")[1]) if "." in q["raw"] else 0) * (multiplier if q["scale"] is None else 1))
                if target and math.isclose(candidate, target, rel_tol=0, abs_tol=step + 1e-9 * target):
                    return True
            if q["scale"] is not None or q["unit_class"] == "mass":
                break
    return False


def ungrounded_numbers(answer: str, texts: Sequence[str], allowed: Sequence[float] = ()) -> list[str]:
    """Answer numbers (excluding years) not found in any of ``texts``.

    ``allowed`` lists base values that are legitimately derived (expected
    changes or calculation results) and are not counted as ungrounded.
    """

    missing = []
    joined = "\n".join(texts)
    for q in parse_quantities(answer):
        if any(math.isclose(abs(q["base_value"]), abs(a), rel_tol=0, abs_tol=q["half_step"] + 1e-9 * abs(a)) for a in allowed):
            continue
        if not number_in_text(q, joined):
            missing.append(q["raw"])
    return missing
