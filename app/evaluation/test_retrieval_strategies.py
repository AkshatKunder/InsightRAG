import json

import pytest

from app.evaluation.retrieval_strategies import (
    RetrievalComponents,
    collapse_to_pages,
    pool_shortfalls,
    rankings_by_strategy,
    rerank_pool,
    run_benchmark,
    run_question,
    strategy_names,
    unanswerable_retrieval,
)

SRC = "report.pdf"


def chunk(page, text=None, score=1.0, source=SRC):
    return {"source": source, "page": page, "text": text or f"text {page}", "score": score}


def fake_components(bm25_pages, vector_pages, hybrid_pages, rerank=None, calls=None):
    calls = calls if calls is not None else []

    def bm25(query, k):
        calls.append(("bm25", query, k))
        return [chunk(p, score=100 - i) for i, p in enumerate(bm25_pages)][:k]

    def vector(query, k):
        calls.append(("vector", query, k))
        return [chunk(p, score=1 - i / 100) for i, p in enumerate(vector_pages)][:k]

    def hybrid(query, top_k, candidate_k):
        calls.append(("hybrid", query, top_k, candidate_k))
        return [chunk(p, score=1 / (61 + i)) for i, p in enumerate(hybrid_pages)][:top_k]

    return RetrievalComponents(bm25, vector, hybrid, expand_query=lambda q: q + " expanded", rerank=rerank)


def reverse_rerank(question, docs, top_k):
    """Fake reranker: reverses order; cross-encoder score favours odd pages."""

    out = []
    for i, d in enumerate(reversed(docs)):
        out.append({**d, "final_score": float(len(docs) - i), "reranker_score": float(d["page"] % 2)})
    return out[:top_k]


# ---------------------------------------------------------------- collapsing


def test_collapse_keeps_first_chunk_per_page_and_order():
    pool = collapse_to_pages([chunk(5, "a"), chunk(3), chunk(5, "b"), chunk(9)])
    assert [p["page"] for p in pool] == [5, 3, 9]
    assert pool[0]["text"] == "a"


def test_collapse_limit_counts_distinct_pages():
    pool = collapse_to_pages([chunk(1), chunk(1), chunk(2), chunk(3)], limit=2)
    assert [p["page"] for p in pool] == [1, 2]


def test_collapse_keeps_invalid_metadata_as_separate_units():
    pool = collapse_to_pages([{"text": "x"}, chunk(1), {"page": None, "source": SRC}])
    assert len(pool) == 3


# ---------------------------------------------------------------- pools


def test_base_strategies_share_expanded_query_and_fill_pool():
    calls = []
    comps = fake_components([1, 1, 2, 3], [4, 5, 6], [7, 8, 9], calls=calls)

    runs = run_question("q", comps, include_rerank=False, candidate_k=2, pool_size=3, max_candidate_k=8)

    assert [p["page"] for p in runs["bm25"]["results"]] == [1, 2, 3]
    assert [p["page"] for p in runs["vector"]["results"]] == [4, 5, 6]
    assert [p["page"] for p in runs["hybrid"]["results"]] == [7, 8, 9]
    assert ("bm25", "q expanded", 2) in calls and ("bm25", "q expanded", 4) in calls  # deepened for duplicates
    assert ("hybrid", "q", 3, 2) in calls  # hybrid expands internally
    assert runs["bm25"]["diagnostics"]["distinct_pages_available"] == 3


def test_fetch_stops_at_max_candidate_k():
    calls = []
    comps = fake_components([1] * 100, [2], [3], calls=calls)

    runs = run_question("q", comps, include_rerank=False, candidate_k=10, pool_size=5, max_candidate_k=40)

    depths = [c[2] for c in calls if c[0] == "bm25"]
    assert depths == [10, 20, 40]
    assert len(runs["bm25"]["results"]) == 1


def test_rerank_reorders_identical_pool():
    comps = fake_components([1, 2, 3], [4, 5, 6], [7, 8, 9], rerank=reverse_rerank)

    runs = run_question("q", comps, candidate_k=3, pool_size=3)

    base_pages = [p["page"] for p in runs["hybrid"]["results"]]
    blend = [p["page"] for p in runs["hybrid+rerank"]["results"]]
    ce = [p["page"] for p in runs["hybrid+ce"]["results"]]
    assert sorted(blend) == sorted(base_pages) == sorted(ce)
    assert blend == [9, 8, 7]
    assert ce == [9, 7, 8]  # odd pages first, ties keep the blend order
    assert set(runs) == set(strategy_names(True))


def test_rerank_pool_rejects_dropped_candidates():
    with pytest.raises(ValueError):
        rerank_pool("q", [chunk(1), chunk(2)], lambda q, d, k: d[:1])


def test_rerank_required_when_requested():
    with pytest.raises(ValueError):
        run_question("q", fake_components([1], [2], [3]), include_rerank=True)


def test_errors_are_recorded_not_skipped():
    comps = fake_components([1], [2], [3], rerank=reverse_rerank)

    def broken(query, k):
        raise RuntimeError("qdrant down")

    comps.vector_search = broken
    runs = run_question("q", comps, candidate_k=1, pool_size=1)

    assert runs["vector"]["results"] == [] and "qdrant down" in runs["vector"]["error"]
    assert "base strategy failed" in runs["vector+rerank"]["error"]
    assert runs["bm25"]["error"] is None


# ---------------------------------------------------------------- benchmark runs


def benchmark():
    return {
        "source_document": SRC,
        "questions": [
            {"id": "Q1", "category": "numerical", "question": "a?", "answerable": True, "expected_pages": [2]},
            {"id": "Q2", "category": "unanswerable", "question": "b?", "answerable": False,
             "expected_pages": [], "distractor_pages": [3]},
        ],
    }


def test_run_benchmark_covers_all_questions_and_is_json():
    comps = fake_components([1, 2, 3], [3, 2, 1], [2, 3, 1], rerank=reverse_rerank)

    runs = run_benchmark(benchmark(), comps, candidate_k=3, pool_size=3)
    json.dumps(runs)

    assert [r["id"] for r in runs["questions"]] == ["Q1", "Q2"]
    assert runs["errors"] == []
    rankings = rankings_by_strategy(runs)
    assert [r["page"] for r in rankings["hybrid"]["Q1"]] == [2, 3, 1]
    # Previews only on base strategies.
    record = runs["questions"][0]["strategies"]
    assert "text_preview" in record["bm25"]["results"][0]
    assert "text_preview" not in record["bm25+rerank"]["results"][0]
    assert pool_shortfalls(runs, 3) == []
    assert {s["strategy"] for s in pool_shortfalls(runs, 5)} == set(runs["strategies"])


def test_unanswerable_retrieval_reports_distractors():
    comps = fake_components([3, 1], [1, 2], [2, 3], rerank=reverse_rerank)
    runs = run_benchmark(benchmark(), comps, candidate_k=2, pool_size=2)

    report = unanswerable_retrieval(benchmark(), runs, k=1)

    assert [r["id"] for r in report] == ["Q2"]
    assert report[0]["strategies"]["bm25"]["distractor_pages_in_top_k"] == [3]
    assert report[0]["strategies"]["vector"]["distractor_pages_in_top_k"] == []
    assert report[0]["strategies"]["hybrid+rerank"]["max_reranker_score"] == 1.0
