import copy
import json
from pathlib import Path

import pytest

from app.evaluation.retrieval_evaluator import (
    evaluate_benchmark,
    evaluate_question,
    evaluate_results,
    evidence_key,
    fact_page_groups,
    load_benchmark,
    relevant_pages,
)


ROOT = Path(__file__).resolve().parents[2]
REAL_BENCHMARK = ROOT / "data" / "evaluation" / "benchmark_v1.json"
SOURCE = "report.pdf"


def chunk(page, source=SOURCE, text="chunk text", **extra):
    return {"source": source, "page": page, "text": text, **extra}


def make_benchmark():
    return {
        "benchmark_name": "synthetic",
        "version": "test",
        "source_document": SOURCE,
        "questions": [
            {
                "id": "Q001",
                "category": "numerical",
                "question": "Total sales?",
                "answerable": True,
                "expected_pages": [10],
                "expected_facts": [{"metric": "sales", "period": "2025", "pages": [10, 20]}],
            },
            {
                "id": "Q002",
                "category": "multi_hop",
                "question": "Sales per employee?",
                "answerable": True,
                "expected_pages": [10, 30],
                "expected_facts": [
                    {"metric": "sales", "pages": [10]},
                    {"metric": "employees", "pages": [30]},
                ],
            },
            {
                "id": "Q003",
                "category": "unanswerable",
                "question": "Sales in 2030?",
                "answerable": False,
                "expected_pages": [],
            },
        ],
    }


def question(benchmark, qid):
    return next(q for q in benchmark["questions"] if q["id"] == qid)


# ---------------------------------------------------------------- matching


def test_evidence_key_uses_file_name_and_page():
    assert evidence_key(chunk(5)) == (SOURCE, 5)
    assert evidence_key(chunk(5, source="data/documents/report.pdf")) == (SOURCE, 5)
    assert evidence_key(chunk(5, source="data\\documents\\report.pdf")) == (SOURCE, 5)


@pytest.mark.parametrize(
    "result",
    [
        {"source": SOURCE},
        {"page": 3},
        {"source": SOURCE, "page": None},
        {"source": SOURCE, "page": "3"},
        {"source": SOURCE, "page": 0},
        {"source": SOURCE, "page": True},
        {"source": "", "page": 3},
        "not a dict",
    ],
)
def test_evidence_key_rejects_invalid_metadata(result):
    assert evidence_key(result) is None


def test_matching_requires_source_and_page():
    bench = make_benchmark()
    results = [chunk(10, source="other.pdf"), chunk(99), chunk(10)]

    scored = evaluate_question(question(bench, "Q001"), results, [1, 3], SOURCE)

    assert [r["relevant"] for r in scored["retrieved"]] == [False, False, True]
    assert scored["first_relevant_rank"] == 3
    assert scored["reciprocal_rank"] == pytest.approx(0.3333)
    assert scored["metrics"]["recall"] == {"1": 0.0, "3": 1.0}


def test_relevant_pages_include_fact_pages():
    q = question(make_benchmark(), "Q001")
    assert relevant_pages(q) == [10, 20]
    assert fact_page_groups(q) == [("sales (2025)", [10, 20])]


def test_fact_on_alternative_page_counts_for_recall_but_not_page_recall():
    bench = make_benchmark()

    scored = evaluate_question(question(bench, "Q001"), [chunk(20)], [1], SOURCE)

    assert scored["metrics"]["recall"]["1"] == 1.0
    assert scored["metrics"]["page_recall"]["1"] == 0.0
    assert scored["metrics"]["precision"]["1"] == 1.0


def test_question_without_facts_falls_back_to_expected_pages():
    q = {"id": "Q9", "answerable": True, "expected_pages": [4, 2]}
    assert fact_page_groups(q) == [("page_2", [2]), ("page_4", [4])]


# ---------------------------------------------------------------- scoring


def test_multiple_expected_pages():
    bench = make_benchmark()
    results = [chunk(30), chunk(1), chunk(10)]

    scored = evaluate_question(question(bench, "Q002"), results, [1, 3], SOURCE)

    assert scored["metrics"]["recall"] == {"1": 0.5, "3": 1.0}
    assert scored["metrics"]["page_recall"] == {"1": 0.5, "3": 1.0}
    assert scored["metrics"]["precision"] == {"1": 1.0, "3": 0.6667}
    assert scored["relevant_pages_retrieved"] == [30, 10]


def test_duplicate_chunks_do_not_inflate_metrics():
    bench = make_benchmark()
    results = [chunk(10, text="a"), chunk(10, text="b"), chunk(10, text="c")]

    scored = evaluate_question(question(bench, "Q002"), results, [3], SOURCE)

    assert scored["metrics"]["recall"]["3"] == 0.5
    assert scored["metrics"]["page_recall"]["3"] == 0.5
    assert scored["metrics"]["precision"]["3"] == pytest.approx(0.3333)
    assert [r["new_relevant_page"] for r in scored["retrieved"]] == [True, False, False]


def test_ranking_order_is_preserved_not_resorted_by_score():
    bench = make_benchmark()
    results = [chunk(99, score=0.1), chunk(10, score=0.9)]

    scored = evaluate_question(question(bench, "Q001"), results, [1], SOURCE)

    assert scored["retrieved_pages"] == [99, 10]
    assert scored["first_relevant_rank"] == 2
    assert scored["metrics"]["recall"]["1"] == 0.0


def test_empty_results():
    bench = make_benchmark()

    scored = evaluate_question(question(bench, "Q001"), [], [1, 5], SOURCE)

    assert scored["num_results"] == 0
    assert scored["reciprocal_rank"] == 0.0
    assert scored["first_relevant_rank"] is None
    for name in ("recall", "page_recall", "precision"):
        assert scored["metrics"][name] == {"1": 0.0, "5": 0.0}


def test_invalid_metadata_keeps_its_rank_but_is_not_relevant():
    bench = make_benchmark()
    results = [{"source": SOURCE, "page": None, "text": "x"}, {"text": "no metadata"}, chunk(10)]

    scored = evaluate_question(question(bench, "Q001"), results, [3], SOURCE)

    assert scored["num_invalid_metadata"] == 2
    assert [r["invalid_metadata"] for r in scored["retrieved"]] == [True, True, False]
    assert scored["first_relevant_rank"] == 3
    assert scored["metrics"]["precision"]["3"] == pytest.approx(0.3333)


@pytest.mark.parametrize("bad_k", [[0], [-3], [2.0], ["5"], [], [True]])
def test_invalid_k_values_rejected(bad_k):
    bench = make_benchmark()
    with pytest.raises(ValueError):
        evaluate_results(bench, {"Q001": [], "Q002": []}, bad_k)


def test_invalid_k_values_rejected_before_retrieval():
    calls = []

    with pytest.raises(ValueError):
        evaluate_benchmark(make_benchmark(), lambda text: calls.append(text) or [], [0])

    assert calls == []


def test_unanswerable_question_cannot_be_scored():
    bench = make_benchmark()
    with pytest.raises(ValueError):
        evaluate_question(question(bench, "Q003"), [], [1], SOURCE)


# ---------------------------------------------------------------- aggregation


def test_unanswerable_questions_excluded_from_averages():
    bench = make_benchmark()
    results = {
        "Q001": [chunk(10)],  # RR 1, recall 1
        "Q002": [chunk(99), chunk(30)],  # RR 0.5, recall@1 0, recall@3 0.5
        "Q003": [chunk(10)],  # ignored
    }

    report = evaluate_results(bench, results, [1, 3])
    summary = report["summary"]

    assert [r["id"] for r in report["questions"]] == ["Q001", "Q002"]
    assert summary["num_answerable"] == 2
    assert summary["num_unanswerable"] == 1
    assert summary["num_evaluated"] == 2
    assert summary["excluded_unanswerable_ids"] == ["Q003"]
    assert summary["overall"]["mrr"] == 0.75
    assert summary["overall"]["recall"] == {"1": 0.5, "3": 0.75}
    assert summary["overall"]["precision"] == {"1": 0.5, "3": 0.75}
    assert summary["by_category"]["multi_hop"]["mrr"] == 0.5
    assert "unanswerable" not in summary["by_category"]
    assert summary["questions_with_fewer_than_max_k_results"] == ["Q001", "Q002"]


def test_missing_results_for_answerable_question_raises():
    with pytest.raises(KeyError, match="Q002"):
        evaluate_results(make_benchmark(), {"Q001": []}, [1])


def test_evaluate_benchmark_only_queries_answerable_questions():
    asked = []

    def fake_retrieve(text):
        asked.append(text)
        return [chunk(10)]

    report = evaluate_benchmark(make_benchmark(), fake_retrieve, [1], config={"strategy": "fake"})

    assert asked == ["Total sales?", "Sales per employee?"]
    assert report["summary"]["config"] == {"strategy": "fake"}
    assert report["summary"]["questions_with_no_relevant_result"] == []


def test_evaluator_does_not_mutate_inputs():
    bench = make_benchmark()
    results = {"Q001": [chunk(10, score=1.0)], "Q002": [chunk(30)]}
    bench_before, results_before = copy.deepcopy(bench), copy.deepcopy(results)

    evaluate_results(bench, results, [1])

    assert bench == bench_before
    assert results == results_before


def test_numpy_scores_are_kept_as_plain_floats():
    np = pytest.importorskip("numpy")
    results = [chunk(10, score=np.float64(0.5), final_score=np.float32(0.25), bm25_rank=np.int64(3))]

    scored = evaluate_question(question(make_benchmark(), "Q001"), results, [1], SOURCE)
    entry = scored["retrieved"][0]

    assert entry["final_score"] == 0.25 and type(entry["final_score"]) is float
    assert entry["bm25_rank"] == 3 and type(entry["bm25_rank"]) is int
    json.dumps(scored)


def test_report_is_json_serialisable():
    report = evaluate_results(make_benchmark(), {"Q001": [chunk(10, score=0.5)], "Q002": []}, [1, 3])
    json.dumps(report)


# ---------------------------------------------------------------- real benchmark


def test_works_with_real_benchmark_file():
    bench = load_benchmark(REAL_BENCHMARK)

    def perfect_retriever(text):
        q = next(q for q in bench["questions"] if q["question"] == text)
        return [chunk(page, source=bench["source_document"]) for page in q["expected_pages"]]

    report = evaluate_benchmark(bench, perfect_retriever, [10])
    summary = report["summary"]

    assert summary["num_answerable"] == 22
    assert summary["num_unanswerable"] == 3
    assert summary["num_evaluated"] == 22
    assert summary["overall"]["recall"]["10"] == 1.0
    assert summary["overall"]["page_recall"]["10"] == 1.0
    assert summary["overall"]["precision"]["10"] == 1.0
    assert summary["overall"]["mrr"] == 1.0


def test_load_benchmark_rejects_bad_files(tmp_path):
    path = tmp_path / "bench.json"
    path.write_text(json.dumps({"questions": []}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_benchmark(path)
