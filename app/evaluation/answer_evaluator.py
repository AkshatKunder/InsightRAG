"""Grade pipeline answers, citations and abstentions against the benchmark (deterministic).

Per answerable question:

* **correctness** (``correct`` / ``partially_correct`` / ``incorrect`` /
  ``abstained``), from ``answer_checks``:
  - plain questions: every expected fact matched (value, unit, period, entity);
  - comparative: direction right AND (both period values matched OR the
    absolute or percent change stated);
  - calculation (derived) questions: result within the benchmark tolerance
    AND, for "higher or lower" questions, the right direction;
  - ``partially_correct``: at least one component satisfied but not the rule above.
* **completeness**: satisfied components / all components (facts plus
  direction / change / result components where the question has them).
* **citations**: precision (cited pages that are relevant pages), fact recall
  (facts with at least one cited page among the fact's pages), and support
  (matched numeric facts whose value appears in a cited passage).
* **grounding**: answer numbers not found in any cited passage, and not found
  anywhere in the generator's context (invented). Expected derived values
  (changes, calculation results) are exempt.
* **system claim validation**: the pipeline's own claim statuses, reported
  separately because the validator is part of the system under test.

Per unanswerable question: abstained or not, citations shown to the user
(the assistant prints citations only when it does not abstain), and numbers
in a non-abstaining answer (invented vs. copied from context).

No LLM judge is used.
"""

from __future__ import annotations

import statistics
from collections import Counter
from collections.abc import Mapping, Sequence

from app.evaluation.answer_checks import (
    FACT_UNITS,
    check_calculation,
    check_comparison,
    check_fact,
    number_in_text,
    parse_quantities,
    ungrounded_numbers,
)
from app.evaluation.retrieval_evaluator import relevant_pages

LABELS = ("correct", "partially_correct", "incorrect", "abstained", "error")


def _derived_values(question: Mapping) -> list[float]:
    """Base values of expected derived numbers (changes, calculation results)."""

    values = []
    comparison = question.get("expected_comparison")
    if comparison:
        _, multiplier = FACT_UNITS.get(comparison.get("unit"), (None, 1.0))
        if comparison.get("absolute_change") is not None:
            values.append(abs(comparison["absolute_change"]) * multiplier)
        if comparison.get("percent_change") is not None:
            values.append(abs(comparison["percent_change"]))
    calculation = question.get("calculation")
    if calculation:
        _, multiplier = FACT_UNITS.get(calculation.get("unit"), (None, 1.0))
        values.append(abs(calculation["result"]) * multiplier)
    return values


def _fact_value_in_texts(fact: Mapping, texts: Sequence[str]) -> bool:
    unit_class, multiplier = FACT_UNITS.get(fact.get("unit"), (None, 1.0))
    if unit_class is None or not isinstance(fact.get("value"), (int, float)):
        return False
    value = float(fact["value"])
    decimals = len(str(fact["value"]).split(".")[1]) if "." in str(fact["value"]) else 0
    quantity = {"base_value": value * multiplier, "half_step": 0.5 * 10 ** -decimals * multiplier}
    return number_in_text(quantity, "\n".join(texts))


def _is_abstention(record: Mapping) -> bool:
    validation = record.get("validation") or {}
    return validation.get("overall_status") == "ABSTAINED"


def evaluate_answerable(question: Mapping, record: Mapping) -> dict:
    answer = record.get("answer") or ""
    text = question["question"]
    out = {
        "id": question["id"],
        "category": question.get("category"),
        "answer": record.get("answer"),
        "validation_status": (record.get("validation") or {}).get("overall_status"),
        "error": record.get("error"),
    }
    if record.get("error"):
        return {**out, "label": "error", "completeness": 0.0}
    if _is_abstention(record):
        out.update(label="abstained", completeness=0.0, fact_checks=[], citations=_citation_metrics(question, record, []),
                   context_fact_recall=_fact_page_recall(question, [c.get("page") for c in record.get("context", [])]))
        return out

    facts = [check_fact(fact, answer, text) for fact in question.get("expected_facts", [])]
    components = [f["matched"] for f in facts]
    comparison = calculation = None
    rule_ok = all(components) if components else False

    if question.get("expected_comparison"):
        comparison = check_comparison(question["expected_comparison"], answer)
        change_ok = bool(comparison["absolute_change_ok"] or comparison["percent_change_ok"])
        components += [comparison["direction_ok"], change_ok]
        rule_ok = comparison["direction_ok"] and (all(f["matched"] for f in facts) or change_ok)
    if question.get("calculation"):
        calculation = check_calculation(question["calculation"], answer, text)
        components.append(calculation["result_ok"])
        if calculation["direction_ok"] is not None:
            components.append(calculation["direction_ok"])
        rule_ok = calculation["result_ok"] and calculation["direction_ok"] is not False

    if rule_ok:
        label = "correct"
    elif any(components):
        label = "partially_correct"
    else:
        label = "incorrect"

    context_texts = [c.get("text", "") for c in record.get("context", [])]
    citation_texts = [c.get("text", "") for c in record.get("citations", [])]
    derived = _derived_values(question)
    out.update(
        label=label,
        completeness=round(sum(1 for c in components if c) / len(components), 4) if components else 0.0,
        fact_checks=facts,
        comparison=comparison,
        calculation=calculation,
        citations=_citation_metrics(question, record, facts),
        claims=_claim_counts(record),
        answer_numbers=[q["raw"] for q in parse_quantities(answer)],
        numbers_not_in_citations=ungrounded_numbers(answer, citation_texts, derived),
        numbers_not_in_context=ungrounded_numbers(answer, context_texts, derived),
        context_fact_recall=_fact_page_recall(question, [c.get("page") for c in record.get("context", [])]),
    )
    return out


def _fact_page_recall(question: Mapping, pages: Sequence) -> float | None:
    facts = question.get("expected_facts") or []
    if not facts:
        return None
    pages = set(pages)
    return round(sum(1 for f in facts if set(f.get("pages") or []) & pages) / len(facts), 4)


def _citation_metrics(question: Mapping, record: Mapping, fact_checks: Sequence[Mapping]) -> dict:
    citations = record.get("citations", [])
    relevant = set(relevant_pages(question))
    cited_pages = [c.get("page") for c in citations]
    relevant_cited = [p for p in cited_pages if p in relevant]
    numeric_matched = [
        (fact, check) for fact, check in zip(question.get("expected_facts", []), fact_checks)
        if check.get("numeric") and check.get("matched") and fact.get("unit") in FACT_UNITS and fact.get("unit") != "segments"
    ]
    supported = [check for fact, check in numeric_matched if _fact_value_in_texts(fact, [c.get("text", "") for c in citations])]
    # A citation "supports" the answer when it contains at least one expected numeric fact value.
    supporting_citations = [
        c for c in citations
        if any(_fact_value_in_texts(fact, [c.get("text", "")]) for fact, _ in numeric_matched)
    ]
    return {
        "cited_pages": cited_pages,
        "num_citations": len(citations),
        "relevant_cited": len(relevant_cited),
        "precision": round(len(relevant_cited) / len(citations), 4) if citations else None,
        "fact_recall": _fact_page_recall(question, cited_pages) if citations else (0.0 if question.get("expected_facts") else None),
        "matched_numeric_facts": len(numeric_matched),
        "matched_numeric_facts_supported_by_citation": len(supported),
        "citations_containing_an_expected_value": len(supporting_citations) if numeric_matched else None,
    }


def _claim_counts(record: Mapping) -> dict:
    claims = (record.get("validation") or {}).get("claims", [])
    counts = Counter(claim.get("status") for claim in claims)
    return {"total": len(claims), "supported": counts.get("SUPPORTED", 0), "unsupported": counts.get("UNSUPPORTED", 0),
            "insufficient": counts.get("INSUFFICIENT_EVIDENCE", 0)}


def evaluate_unanswerable(question: Mapping, record: Mapping) -> dict:
    answer = record.get("answer") or ""
    abstained = _is_abstention(record)
    citations = record.get("citations", [])
    context_texts = [c.get("text", "") for c in record.get("context", [])]
    numbers = [q["raw"] for q in parse_quantities(answer)]
    invented = ungrounded_numbers(answer, context_texts)
    distractor = set(question.get("distractor_pages") or [])
    return {
        "id": question["id"],
        "category": question.get("category"),
        "answer": record.get("answer"),
        "validation_status": (record.get("validation") or {}).get("overall_status"),
        "error": record.get("error"),
        "abstained": abstained and not record.get("error"),
        "false_answer": not abstained and not record.get("error"),
        "citations_selected": len(citations),
        # app/main.py prints citations only when the status is not ABSTAINED.
        "citations_shown": (not abstained) and bool(citations),
        "cited_pages": [c.get("page") for c in citations],
        "cited_distractor_pages": [c.get("page") for c in citations if c.get("page") in distractor],
        "answer_numbers": numbers,
        "invented_numbers": invented,
        "numbers_copied_from_context": [n for n in numbers if n not in invented],
        "context_pages": [c.get("page") for c in record.get("context", [])],
    }


def evaluate_run(benchmark: Mapping, run: Mapping) -> dict:
    by_id = {record["id"]: record for record in run["questions"]}
    answerable, unanswerable = [], []
    for question in benchmark["questions"]:
        record = by_id.get(question["id"])
        if record is None:
            record = {"error": "missing from run", "answer": None}
        if question.get("answerable") is True:
            answerable.append(evaluate_answerable(question, record))
        else:
            unanswerable.append(evaluate_unanswerable(question, record))
    return {"run": run.get("run"), "answerable": answerable, "unanswerable": unanswerable,
            "metrics": run_metrics(answerable, unanswerable)}


def _rate(numerator: float, denominator: float) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def run_metrics(answerable: Sequence[Mapping], unanswerable: Sequence[Mapping]) -> dict:
    labels = Counter(r["label"] for r in answerable)
    n = len(answerable)
    answered = [r for r in answerable if r["label"] not in ("abstained", "error")]
    citations = [r["citations"] for r in answered]
    total_cited = sum(c["num_citations"] for c in citations)
    matched_numeric = sum(c["matched_numeric_facts"] for c in citations)
    claims = Counter()
    for r in answered:
        claims.update(r["claims"])
    statuses = Counter(r.get("validation_status") for r in answerable)

    by_category = {}
    for category in dict.fromkeys(r["category"] for r in answerable):
        rows = [r for r in answerable if r["category"] == category]
        by_category[category] = {
            "n": len(rows),
            "accuracy": _rate(sum(1 for r in rows if r["label"] == "correct"), len(rows)),
            "correct_or_partial": _rate(sum(1 for r in rows if r["label"] in ("correct", "partially_correct")), len(rows)),
            "completeness": round(statistics.mean(r["completeness"] for r in rows), 4),
        }

    u = len(unanswerable)
    return {
        "answerable": {
            "n": n,
            "labels": {label: labels.get(label, 0) for label in LABELS},
            "answer_accuracy": _rate(labels.get("correct", 0), n),
            "correct_or_partial_rate": _rate(labels.get("correct", 0) + labels.get("partially_correct", 0), n),
            "false_abstention_rate": _rate(labels.get("abstained", 0), n),
            "error_rate": _rate(labels.get("error", 0), n),
            "mean_completeness": round(statistics.mean(r["completeness"] for r in answerable), 4) if n else None,
            "citation_precision_micro": _rate(sum(c["relevant_cited"] for c in citations), total_cited),
            "citation_precision_macro": _rate(sum(c["precision"] for c in citations if c["precision"] is not None),
                                             sum(1 for c in citations if c["precision"] is not None)),
            "citation_fact_recall": _rate(sum(c["fact_recall"] or 0 for c in citations), len(citations)),
            "answers_without_citations": sum(1 for c in citations if c["num_citations"] == 0),
            "matched_numeric_fact_citation_support": _rate(sum(c["matched_numeric_facts_supported_by_citation"] for c in citations), matched_numeric),
            "system_claims": dict(claims),
            "supported_claim_rate": _rate(claims["supported"], claims["total"]),
            "unsupported_claim_rate": _rate(claims["unsupported"], claims["total"]),
            "insufficient_evidence_claim_rate": _rate(claims["insufficient"], claims["total"]),
            "answers_with_numbers_not_in_citations": sum(1 for r in answered if r["numbers_not_in_citations"]),
            "answers_with_numbers_not_in_context": sum(1 for r in answered if r["numbers_not_in_context"]),
            # Over every non-error answerable question, abstentions included (retrieval ceiling for generation).
            "context_fact_recall": _rate(
                sum(r.get("context_fact_recall") or 0 for r in answerable if r["label"] != "error"),
                sum(1 for r in answerable if r["label"] != "error"),
            ),
            "questions_with_incomplete_context": sorted(
                r["id"] for r in answerable if r["label"] != "error" and (r.get("context_fact_recall") or 0) < 1.0
            ),
            "validation_status": dict(statuses),
            "by_category": by_category,
        },
        "unanswerable": {
            "n": u,
            "correct_abstention_rate": _rate(sum(1 for r in unanswerable if r["abstained"]), u),
            "false_answer_rate": _rate(sum(1 for r in unanswerable if r["false_answer"]), u),
            "citations_shown_rate": _rate(sum(1 for r in unanswerable if r["citations_shown"]), u),
            "answers_with_invented_numbers": sum(1 for r in unanswerable if r["invented_numbers"]),
            "error_rate": _rate(sum(1 for r in unanswerable if r["error"]), u),
        },
    }


def _summarise(values: Sequence[float | None]) -> dict | None:
    values = [v for v in values if v is not None]
    if not values:
        return None
    return {"mean": round(statistics.mean(values), 4), "min": round(min(values), 4), "max": round(max(values), 4)}


def aggregate_runs(run_results: Sequence[Mapping]) -> dict:
    """Mean/min/max of every scalar metric across runs, plus per-question stability."""

    def collect(section: str, key: str):
        return [r["metrics"][section][key] for r in run_results]

    scalar = {}
    for section in ("answerable", "unanswerable"):
        scalar[section] = {}
        for key, value in run_results[0]["metrics"][section].items():
            if isinstance(value, (int, float)) and not isinstance(value, bool) or value is None:
                scalar[section][key] = _summarise(collect(section, key))

    stability = []
    for index, first in enumerate(run_results[0]["answerable"]):
        labels = [r["answerable"][index]["label"] for r in run_results]
        stability.append({"id": first["id"], "category": first["category"], "labels": labels,
                          "majority": Counter(labels).most_common(1)[0][0], "stable": len(set(labels)) == 1})
    abstention = []
    for index, first in enumerate(run_results[0]["unanswerable"]):
        flags = [r["unanswerable"][index]["abstained"] for r in run_results]
        abstention.append({"id": first["id"], "abstained": flags, "stable": len(set(flags)) == 1})

    categories = run_results[0]["metrics"]["answerable"]["by_category"]
    by_category = {
        c: {key: _summarise([r["metrics"]["answerable"]["by_category"][c][key] for r in run_results])
            for key in ("accuracy", "correct_or_partial", "completeness")}
        for c in categories
    }
    return {
        "num_runs": len(run_results),
        "metrics": scalar,
        "by_category": by_category,
        "answerable_stability": stability,
        "unanswerable_stability": abstention,
        "stable_questions": sum(1 for s in stability if s["stable"]),
    }
