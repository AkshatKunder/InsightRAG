"""Compare scored retrieval reports (pure functions, no models).

Inputs are reports produced by ``retrieval_evaluator.evaluate_results``:
``{"summary": {...}, "questions": [per-question records]}``.

* ``hit_rate`` / ``strategy_table`` summarise several strategies side by side.
* ``compare_reports`` compares two orderings of the same questions (for
  example a strategy before and after reranking) question by question.

Verdict for one question (before -> after):

1. reciprocal rank higher -> ``improved``; lower -> ``degraded``;
2. if equal, the sum of fact recall over all K values decides;
3. otherwise ``unchanged``.

Independently, every relevant page that sits lower in the after-ranking than
in the before-ranking (or drops out of it) is listed in ``relevant_moved_down``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

_EPS = 1e-9


def _round(value: float) -> float:
    return round(value, 4)


def hit_rate(report: Mapping, k: int) -> float:
    """Share of questions whose first relevant unit is within the top ``k``."""

    questions = report["questions"]
    if not questions:
        return 0.0
    hits = sum(1 for q in questions if q["first_relevant_rank"] is not None and q["first_relevant_rank"] <= k)
    return _round(hits / len(questions))


def strategy_table(reports: Mapping[str, Mapping], k_values: Sequence[int]) -> dict[str, dict]:
    """Overall metrics per strategy, plus hit rate@K."""

    table = {}
    for name, report in reports.items():
        overall = report["summary"]["overall"]
        table[name] = {
            "mrr": overall["mrr"],
            "recall": overall["recall"],
            "page_recall": overall["page_recall"],
            "precision": overall["precision"],
            "hit_rate": {str(k): hit_rate(report, k) for k in k_values},
            "num_questions": overall["num_questions"],
        }
    return table


def best_strategy(table: Mapping[str, Mapping], metric: str, k: int | None = None) -> list[str]:
    """Names of the strategies with the highest value of ``metric`` (ties all returned)."""

    def value(name):
        entry = table[name][metric]
        return entry if k is None else entry[str(k)]

    top = max(value(name) for name in table)
    return [name for name in table if abs(value(name) - top) < _EPS]


def _unit_positions(question: Mapping) -> dict[int, int]:
    """page -> 1-based position in the distinct-unit ranking (first occurrence)."""

    positions = {}
    for index, page in enumerate(question.get("ranked_pages", []), start=1):
        if page is not None and page not in positions:
            positions[page] = index
    return positions


def compare_question(before: Mapping, after: Mapping, k_values: Sequence[int]) -> dict:
    if before["id"] != after["id"]:
        raise ValueError(f"Comparing different questions: {before['id']} vs {after['id']}")

    rr_before, rr_after = before["reciprocal_rank"], after["reciprocal_rank"]
    recall_before = {k: before["metrics"]["recall"][str(k)] for k in k_values}
    recall_after = {k: after["metrics"]["recall"][str(k)] for k in k_values}

    if rr_after > rr_before + _EPS:
        verdict = "improved"
    elif rr_after < rr_before - _EPS:
        verdict = "degraded"
    else:
        delta = sum(recall_after.values()) - sum(recall_before.values())
        verdict = "improved" if delta > _EPS else "degraded" if delta < -_EPS else "unchanged"

    pos_before, pos_after = _unit_positions(before), _unit_positions(after)
    moves = []
    for page in before.get("relevant_pages", []):
        b, a = pos_before.get(page), pos_after.get(page)
        if b is None and a is None:
            continue
        moves.append({"page": page, "rank_before": b, "rank_after": a})
    moved_down = [m for m in moves if m["rank_before"] is not None and (m["rank_after"] is None or m["rank_after"] > m["rank_before"])]

    return {
        "id": before["id"],
        "category": before.get("category"),
        "verdict": verdict,
        "first_relevant_rank_before": before["first_relevant_rank"],
        "first_relevant_rank_after": after["first_relevant_rank"],
        "reciprocal_rank_before": rr_before,
        "reciprocal_rank_after": rr_after,
        "reciprocal_rank_delta": _round(rr_after - rr_before),
        "recall_before": {str(k): v for k, v in recall_before.items()},
        "recall_after": {str(k): v for k, v in recall_after.items()},
        "relevant_page_moves": moves,
        "relevant_moved_down": moved_down,
    }


def compare_reports(before: Mapping, after: Mapping, k_values: Sequence[int]) -> dict:
    """Question-by-question and aggregate comparison of two reports over the same questions."""

    before_q = {q["id"]: q for q in before["questions"]}
    after_q = {q["id"]: q for q in after["questions"]}
    if set(before_q) != set(after_q):
        raise ValueError("Reports cover different questions")

    per_question = [compare_question(before_q[qid], after_q[qid], k_values) for qid in before_q]
    by_verdict = {verdict: [c["id"] for c in per_question if c["verdict"] == verdict] for verdict in ("improved", "degraded", "unchanged")}

    b, a = before["summary"]["overall"], after["summary"]["overall"]
    deltas = {"mrr": _round(a["mrr"] - b["mrr"])}
    for metric in ("recall", "page_recall", "precision"):
        deltas[metric] = {str(k): _round(a[metric][str(k)] - b[metric][str(k)]) for k in k_values}
    hit = {
        str(k): {"before": hit_rate(before, k), "after": hit_rate(after, k)} for k in k_values
    }
    for entry in hit.values():
        entry["delta"] = _round(entry["after"] - entry["before"])

    categories = {}
    for category in before["summary"]["by_category"]:
        cb, ca = before["summary"]["by_category"][category], after["summary"]["by_category"][category]
        categories[category] = {
            "mrr_before": cb["mrr"],
            "mrr_after": ca["mrr"],
            "mrr_delta": _round(ca["mrr"] - cb["mrr"]),
            "recall_before": cb["recall"],
            "recall_after": ca["recall"],
        }

    return {
        "counts": {verdict: len(ids) for verdict, ids in by_verdict.items()},
        "questions_by_verdict": by_verdict,
        "questions_with_relevant_moved_down": [c["id"] for c in per_question if c["relevant_moved_down"]],
        "overall_before": b,
        "overall_after": a,
        "deltas": deltas,
        "hit_rate": hit,
        "by_category": categories,
        "per_question": per_question,
    }
