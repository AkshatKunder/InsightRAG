import pytest

from app.evaluation.comparison import best_strategy, compare_question, compare_reports, hit_rate, strategy_table
from app.evaluation.retrieval_evaluator import evaluate_results

SRC = "r.pdf"
K = [1, 3]


def bench():
    return {
        "source_document": SRC,
        "questions": [
            {"id": "Q1", "category": "a", "question": "?", "answerable": True, "expected_pages": [5]},
            {"id": "Q2", "category": "a", "question": "?", "answerable": True, "expected_pages": [7, 8],
             "expected_facts": [{"metric": "x", "pages": [7]}, {"metric": "y", "pages": [8]}]},
            {"id": "Q3", "category": "b", "question": "?", "answerable": True, "expected_pages": [9]},
        ],
    }


def pages(*ps):
    return [{"source": SRC, "page": p} for p in ps]


def report(rankings):
    return evaluate_results(bench(), rankings, K)


BEFORE = report({"Q1": pages(1, 5), "Q2": pages(7, 1, 8), "Q3": pages(9)})
AFTER = report({"Q1": pages(5, 1), "Q2": pages(7, 8, 1), "Q3": pages(1, 2, 9)})


def test_hit_rate():
    assert hit_rate(BEFORE, 1) == pytest.approx(2 / 3, abs=1e-4)
    assert hit_rate(AFTER, 1) == pytest.approx(2 / 3, abs=1e-4)
    assert hit_rate(AFTER, 3) == 1.0


def test_verdicts():
    result = compare_reports(BEFORE, AFTER, K)
    verdicts = {c["id"]: c["verdict"] for c in result["per_question"]}

    # Q2: same RR and same recall at every K (page 8 only moved 3 -> 2) -> unchanged.
    assert verdicts == {"Q1": "improved", "Q2": "unchanged", "Q3": "degraded"}
    assert result["counts"] == {"improved": 1, "degraded": 1, "unchanged": 1}
    assert result["questions_by_verdict"]["degraded"] == ["Q3"]


def test_tie_on_reciprocal_rank_decided_by_recall():
    q2 = compare_question(BEFORE["questions"][1], AFTER["questions"][1], K)
    assert q2["reciprocal_rank_delta"] == 0
    assert q2["recall_before"] == {"1": 0.5, "3": 1.0}
    assert q2["recall_after"] == {"1": 0.5, "3": 1.0}
    assert q2["verdict"] == "unchanged"
    assert q2["relevant_moved_down"] == []


def test_unchanged_when_identical():
    result = compare_reports(BEFORE, BEFORE, K)
    assert result["counts"] == {"improved": 0, "degraded": 0, "unchanged": 3}
    assert result["deltas"]["mrr"] == 0


def test_relevant_moved_down_and_dropped():
    before = report({"Q1": pages(5, 1), "Q2": pages(8, 7), "Q3": pages(9)})
    after = report({"Q1": pages(1, 5), "Q2": pages(7, 1), "Q3": pages(1)})

    result = compare_reports(before, after, K)
    by_id = {c["id"]: c for c in result["per_question"]}

    assert by_id["Q1"]["relevant_moved_down"] == [{"page": 5, "rank_before": 1, "rank_after": 2}]
    assert {"page": 8, "rank_before": 1, "rank_after": None} in by_id["Q2"]["relevant_moved_down"]
    assert by_id["Q3"]["relevant_moved_down"] == [{"page": 9, "rank_before": 1, "rank_after": None}]
    assert set(result["questions_with_relevant_moved_down"]) == {"Q1", "Q2", "Q3"}
    assert by_id["Q2"]["verdict"] == "degraded"  # recall@1 dropped (fact y lost), RR equal


def test_aggregate_deltas_and_categories():
    result = compare_reports(BEFORE, AFTER, K)
    assert result["deltas"]["mrr"] == pytest.approx(AFTER["summary"]["overall"]["mrr"] - BEFORE["summary"]["overall"]["mrr"], abs=1e-4)
    assert set(result["by_category"]) == {"a", "b"}
    assert result["hit_rate"]["1"]["delta"] == 0


def test_mismatched_reports_rejected():
    other = evaluate_results({**bench(), "questions": bench()["questions"][:2]}, {"Q1": [], "Q2": []}, K)
    with pytest.raises(ValueError):
        compare_reports(BEFORE, other, K)


def test_strategy_table_and_best():
    table = strategy_table({"before": BEFORE, "after": AFTER}, K)
    assert set(table["after"]) == {"mrr", "recall", "page_recall", "precision", "hit_rate", "num_questions"}
    # MRR before = (1/2 + 1 + 1) / 3 = 0.833; after = (1 + 1 + 1/3) / 3 = 0.778.
    assert best_strategy(table, "mrr") == ["before"]
    assert sorted(best_strategy(table, "hit_rate", 3)) == ["after", "before"]  # tie: both 1.0
