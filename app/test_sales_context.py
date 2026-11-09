"""Mocked tests for hybrid retrieval followed by reranking context selection.

These tests intentionally exclude citation selection and answer generation.
"""

from __future__ import annotations


SALES_QUESTION = "What was ArcelorMittal's sales in 2025?"


def _select_sales_context(question: str, retriever, reranker):
    """Mirror the original script's retrieval and reranking sequence."""
    results = retriever.search(question, top_k=20, candidate_k=50)
    return reranker.rerank(question, results, top_k=5)


class _FakeHybridRetriever:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def search(self, question: str, top_k: int = 5, candidate_k: int = 50):
        self.calls.append((question, top_k, candidate_k))
        return self.results


class _FakeReranker:
    def __init__(self, results=None):
        self.results = results
        self.calls = []

    def rerank(self, question: str, documents, top_k: int = 5):
        self.calls.append((question, documents, top_k))
        if self.results is not None:
            return self.results
        return [
            {**document, "reranker_score": float(index), "final_score": float(index)}
            for index, document in enumerate(documents[:top_k], start=1)
        ]


def _hybrid_results(count: int = 2):
    return [
        {
            "text": f"Sales passage {index}",
            "source": "annual_report.pdf",
            "page": index,
            "score": 0.9 - index / 100,
            "rrf_score": 0.9 - index / 100,
            "bm25_score": float(index),
            "vector_score": 0.8 - index / 100,
        }
        for index in range(1, count + 1)
    ]


def test_hybrid_retrieval_receives_original_sales_question_and_limits():
    retriever = _FakeHybridRetriever(_hybrid_results())
    reranker = _FakeReranker()

    _select_sales_context(SALES_QUESTION, retriever, reranker)

    assert retriever.calls == [(SALES_QUESTION, 20, 50)]


def test_reranker_receives_original_question_exact_hybrid_results_and_top_k():
    results = _hybrid_results()
    retriever = _FakeHybridRetriever(results)
    reranker = _FakeReranker()

    _select_sales_context(SALES_QUESTION, retriever, reranker)

    assert reranker.calls == [(SALES_QUESTION, results, 5)]
    assert reranker.calls[0][1] is results


def test_reranked_context_preserves_text_source_page_and_score_metadata():
    retriever = _FakeHybridRetriever(_hybrid_results())
    reranker = _FakeReranker()

    contexts = _select_sales_context(SALES_QUESTION, retriever, reranker)

    assert contexts[0]["text"] == "Sales passage 1"
    assert contexts[0]["source"] == "annual_report.pdf"
    assert contexts[0]["page"] == 1
    assert contexts[0]["score"] == 0.89
    assert contexts[0]["rrf_score"] == 0.89
    assert contexts[0]["bm25_score"] == 1.0
    assert contexts[0]["vector_score"] == 0.79
    assert contexts[0]["reranker_score"] == 1.0
    assert contexts[0]["final_score"] == 1.0


def test_empty_hybrid_results_are_forwarded_and_produce_empty_context():
    retriever = _FakeHybridRetriever([])
    reranker = _FakeReranker()

    contexts = _select_sales_context(SALES_QUESTION, retriever, reranker)

    assert contexts == []
    assert reranker.calls == [(SALES_QUESTION, [], 5)]


def test_empty_reranked_results_are_returned_unchanged():
    retriever = _FakeHybridRetriever(_hybrid_results())
    reranker = _FakeReranker(results=[])

    assert _select_sales_context(SALES_QUESTION, retriever, reranker) == []


def test_reranked_context_is_limited_to_requested_number():
    retriever = _FakeHybridRetriever(_hybrid_results(count=7))
    reranker = _FakeReranker()

    contexts = _select_sales_context(SALES_QUESTION, retriever, reranker)

    assert len(contexts) == 5
    assert reranker.calls[0][2] == 5


def test_context_selection_does_not_mutate_original_hybrid_results():
    results = _hybrid_results()
    original = [dict(result) for result in results]
    retriever = _FakeHybridRetriever(results)
    reranker = _FakeReranker()

    _select_sales_context(SALES_QUESTION, retriever, reranker)

    assert results == original
    assert "reranker_score" not in results[0]
    assert "final_score" not in results[0]


def test_complete_mocked_hybrid_to_reranking_context_flow():
    results = _hybrid_results(count=3)
    retriever = _FakeHybridRetriever(results)
    reranker = _FakeReranker()

    contexts = _select_sales_context(SALES_QUESTION, retriever, reranker)

    assert retriever.calls == [(SALES_QUESTION, 20, 50)]
    assert reranker.calls == [(SALES_QUESTION, results, 5)]
    assert [context["page"] for context in contexts] == [1, 2, 3]
