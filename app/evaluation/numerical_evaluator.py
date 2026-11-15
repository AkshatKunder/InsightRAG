"""Numerical-accuracy and abstention breakdowns (Phase 2, Task 6). Pure functions.

Works on the per-run outputs of ``answer_evaluator.evaluate_run``. Every numeric
expected fact of an answered question is checked on separate dimensions:

=================  ==========================================================
value_found        a rounding-consistent value (>= 2 significant figures)
unit_ok            ... with the expected unit class (currency, mass, count)
period_ok          ... paired with the expected reporting year (None = no period)
entity_ok          ... with the expected segment named (None = no segment)
metric_named       the answer names the metric (sales, net debt, ...)
matched            all of the above that apply (the fact counts as correct)
rounded            matched with a rounded figure rather than the exact one
=================  ==========================================================

Facts are grouped by figure type (financial, production, shipments,
employees) and by question category (so comparative and multi-hop facts can
be read separately). Comparison direction, change values and calculation
results are reported as their own checks.
"""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence

DIMENSIONS = ("value_found", "unit_ok", "period_ok", "entity_ok", "metric_named", "matched")


def figure_type(metric: str | None, unit: str | None) -> str:
    metric = metric or ""
    if unit and unit.startswith("USD"):
        return "financial"
    if metric.endswith("_production"):
        return "production"
    if "shipments" in metric:
        return "shipments"
    if "employees" in metric:
        return "employees"
    return "other"


def _rate(values: Sequence) -> float | None:
    values = [v for v in values if v is not None]
    return round(sum(1 for v in values if v) / len(values), 4) if values else None


def numeric_fact_rows(run_result: Mapping) -> list[dict]:
    rows = []
    for question in run_result["answerable"]:
        if question["label"] in ("abstained", "error"):
            continue
        for check in question.get("fact_checks", []):
            if not check.get("numeric") or check.get("expected_unit") is None:
                continue
            rows.append({
                "id": question["id"],
                "category": question["category"],
                "metric": check["metric"],
                "figure_type": figure_type(check["metric"], check.get("expected_unit")),
                "expected_value": check["expected_value"],
                "expected_unit": check.get("expected_unit"),
                "period": check.get("period"),
                "segment": check.get("segment"),
                **{d: check.get(d) for d in DIMENSIONS},
                "rounded": check.get("rounded"),
                "matched_text": check.get("matched_text"),
                "reason": check.get("reason"),
            })
    return rows


def _dimension_rates(rows: Sequence[Mapping]) -> dict:
    out = {"n_facts": len(rows)}
    for d in DIMENSIONS:
        out[d] = _rate([r[d] for r in rows])
    out["rounded_share_of_matched"] = _rate([r["rounded"] for r in rows if r["matched"]])
    return out


def numerical_run_summary(run_result: Mapping) -> dict:
    rows = numeric_fact_rows(run_result)
    by_type = defaultdict(list)
    by_category = defaultdict(list)
    for row in rows:
        by_type[row["figure_type"]].append(row)
        by_category[row["category"]].append(row)

    comparisons = [q for q in run_result["answerable"] if q.get("comparison")]
    calculations = [q for q in run_result["answerable"] if q.get("calculation")]
    numeric_questions = [
        q for q in run_result["answerable"]
        if q["category"] in ("numerical", "financial", "comparative", "multi_hop")
    ]
    failures = Counter(r["reason"] for r in rows if not r["matched"])
    return {
        "overall": _dimension_rates(rows),
        "by_figure_type": {k: _dimension_rates(v) for k, v in sorted(by_type.items())},
        "by_category": {k: _dimension_rates(v) for k, v in by_category.items()},
        "comparisons": {
            "n": len(comparisons),
            "direction_ok": _rate([q["comparison"]["direction_ok"] for q in comparisons]),
            "absolute_change_ok": _rate([q["comparison"]["absolute_change_ok"] for q in comparisons]),
            "percent_change_ok": _rate([q["comparison"]["percent_change_ok"] for q in comparisons]),
        },
        "calculations": {
            "n": len(calculations),
            "result_ok": _rate([q["calculation"]["result_ok"] for q in calculations]),
            "direction_ok": _rate([q["calculation"]["direction_ok"] for q in calculations]),
        },
        "numeric_question_accuracy": _rate([q["label"] == "correct" for q in numeric_questions]),
        "numeric_questions": len(numeric_questions),
        "failure_reasons": dict(failures),
        "facts": rows,
    }


def _mean_min_max(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return {"mean": round(statistics.mean(values), 4), "min": round(min(values), 4), "max": round(max(values), 4)}


def aggregate_numerical(summaries: Sequence[Mapping]) -> dict:
    def across(path):
        out = []
        for s in summaries:
            node = s
            for key in path:
                node = node.get(key) if isinstance(node, Mapping) else None
            out.append(node)
        return _mean_min_max(out)

    result = {"num_runs": len(summaries), "overall": {}, "by_figure_type": {}, "by_category": {}}
    for d in DIMENSIONS + ("rounded_share_of_matched",):
        result["overall"][d] = across(("overall", d))
    for group in ("by_figure_type", "by_category"):
        keys = list(dict.fromkeys(k for s in summaries for k in s[group]))
        result[group] = {k: {d: across((group, k, d)) for d in DIMENSIONS} for k in keys}
        for k in keys:
            result[group][k]["n_facts"] = summaries[0][group].get(k, {}).get("n_facts")
    result["comparisons"] = {k: across(("comparisons", k)) for k in ("direction_ok", "absolute_change_ok", "percent_change_ok")}
    result["calculations"] = {k: across(("calculations", k)) for k in ("result_ok", "direction_ok")}
    result["numeric_question_accuracy"] = across(("numeric_question_accuracy",))
    result["failure_reasons_total"] = dict(sum((Counter(s["failure_reasons"]) for s in summaries), Counter()))
    return result


def abstention_summary(run_results: Sequence[Mapping]) -> dict:
    """Abstention behaviour across runs: unanswerable questions and false abstentions on answerable ones."""

    per_question = defaultdict(list)
    for run in run_results:
        for record in run["unanswerable"]:
            per_question[record["id"]].append({
                "run": run["run"],
                "abstained": record["abstained"],
                "false_answer": record["false_answer"],
                "citations_selected": record["citations_selected"],
                "citations_shown": record["citations_shown"],
                "cited_distractor_pages": record["cited_distractor_pages"],
                "invented_numbers": record["invented_numbers"],
                "numbers_copied_from_context": record["numbers_copied_from_context"],
                "validation_status": record["validation_status"],
                "answer": record["answer"],
                "error": record["error"],
            })
    flat = [entry for entries in per_question.values() for entry in entries]
    false_abstentions = [
        {"run": run["run"], "id": q["id"], "category": q["category"], "answer": q["answer"]}
        for run in run_results
        for q in run["answerable"]
        if q["label"] == "abstained"
    ]
    total_answerable = sum(len(run["answerable"]) for run in run_results)
    return {
        "unanswerable_trials": len(flat),
        "correct_abstention_rate": _rate([e["abstained"] for e in flat]),
        "false_answer_rate": _rate([e["false_answer"] for e in flat]),
        "citations_shown_rate": _rate([e["citations_shown"] for e in flat]),
        "trials_with_invented_numbers": sum(1 for e in flat if e["invented_numbers"]),
        "trials_with_numbers_copied_from_context": sum(1 for e in flat if e["numbers_copied_from_context"]),
        "errors": sum(1 for e in flat if e["error"]),
        "per_question": dict(per_question),
        "answerable_false_abstentions": false_abstentions,
        "answerable_false_abstention_rate": round(len(false_abstentions) / total_answerable, 4) if total_answerable else None,
    }
