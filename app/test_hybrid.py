"""Deterministic unit tests for weighted hybrid retrieval.

The production fusion module is imported with all expensive dependencies mocked,
so test collection never opens PDFs, models, or Qdrant databases.
"""

from __future__ import annotations

import importlib
import sys
import types

import pytest


def _result(text: str, page: int, score: float, source: str = "sample.pdf") -> dict[str, object]:
    return {"text": text, "source": source, "page": page, "score": score}


@pytest.fixture
def hybrid_module(monkeypatch: pytest.MonkeyPatch):
    """Load real fusion code with lightweight doubles for all dependencies."""
    calls: dict[str, object] = {"load_pdf": [], "chunk_pages": []}

    class FakeBM25Retriever:
        def __init__(self, chunks):
            calls["bm25_chunks"] = chunks
            self.results = []
            calls["bm25"] = self

        def search(self, query, top_k=5):
            calls["bm25_search"] = (query, top_k)
            return self.results

    class FakeRetriever:
        def __init__(self):
            self.results = []
            calls["vector"] = self

        def search(self, query, top_k=5):
            calls["vector_search"] = (query, top_k)
            return self.results

    loader = types.ModuleType("app.rag.loader")
    loader.load_pdf = lambda path: calls["load_pdf"].append(path) or [{"text": "page"}]
    chunker = types.ModuleType("app.rag.chunker")
    chunker.chunk_pages = lambda pages: calls["chunk_pages"].append(pages) or [{"text": "chunk"}]
    bm25 = types.ModuleType("app.rag.bm25_retriever")
    bm25.BM25Retriever = FakeBM25Retriever
    retriever = types.ModuleType("app.rag.retriever")
    retriever.Retriever = FakeRetriever
    expander = types.ModuleType("app.rag.query_expander")
    expander.expand_query = lambda query: f"expanded: {query}"

    for name, module in {
        "app.rag.loader": loader,
        "app.rag.chunker": chunker,
        "app.rag.bm25_retriever": bm25,
        "app.rag.retriever": retriever,
        "app.rag.query_expander": expander,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.delitem(sys.modules, "app.rag.hybrid_retriever", raising=False)
    return importlib.import_module("app.rag.hybrid_retriever"), calls


@pytest.fixture
def retriever(hybrid_module):
    module, calls = hybrid_module
    return module.HybridRetriever(pdf_path="unit-test.pdf"), module, calls


def test_initializes_bm25_and_vector_components(retriever):
    instance, _, calls = retriever

    assert calls["load_pdf"] == ["unit-test.pdf"]
    assert calls["chunk_pages"] == [[{"text": "page"}]]
    assert calls["bm25_chunks"] == [{"text": "chunk"}]
    assert instance.bm25 is calls["bm25"]
    assert instance.vector_retriever is calls["vector"]


def test_requests_bm25_and_vector_candidates_and_respects_limits(retriever):
    instance, _, calls = retriever
    instance.bm25.results = [_result("bm25", 1, 1.0), _result("other", 2, 0.5)]
    instance.vector_retriever.results = [_result("vector", 3, 0.9)]

    results = instance.search("sales", top_k=2, candidate_k=3)

    assert calls["bm25_search"] == ("expanded: sales", 3)
    assert calls["vector_search"] == ("expanded: sales", 3)
    assert len(results) == 2


def test_weighted_rrf_uses_configured_weights_and_rank_constant(retriever):
    instance, module, _ = retriever
    shared = _result("shared", 1, 10.0)
    instance.bm25.results = [shared]
    instance.vector_retriever.results = [shared]

    result = instance.search("question", top_k=1, candidate_k=5)[0]

    assert module.BM25_WEIGHT == 0.4
    assert module.VECTOR_WEIGHT == 0.6
    assert module.RRF_RANK_CONSTANT == 60
    assert result["rrf_score"] == pytest.approx(0.4 / 61 + 0.6 / 61)
    assert result["score"] == result["rrf_score"]


def test_duplicate_chunks_are_merged_and_not_counted_twice_per_retriever(retriever):
    instance, module, _ = retriever
    duplicate = _result("same", 1, 0.8)
    instance.bm25.results = [duplicate, duplicate]
    instance.vector_retriever.results = []

    results = instance.search("question", top_k=2, candidate_k=5)

    assert len(results) == 1
    assert results[0]["bm25_rank"] == 1
    assert results[0]["rrf_score"] == pytest.approx(module.BM25_WEIGHT / 61)


def test_page_diversity_and_multiple_pages_are_preserved(retriever):
    instance, _, _ = retriever
    instance.bm25.results = [
        _result("first page one", 1, 1.0),
        _result("second page one", 1, 0.9),
        _result("page two", 2, 0.8),
    ]
    instance.vector_retriever.results = []

    results = instance.search("question", top_k=3, candidate_k=5)

    assert [(result["text"], result["page"]) for result in results] == [
        ("first page one", 1),
        ("page two", 2),
    ]


@pytest.mark.parametrize("bm25_results,vector_results", [
    ([], [_result("vector only", 2, 0.7)]),
    ([_result("bm25 only", 1, 0.7)], []),
])
def test_empty_retriever_results_are_handled(retriever, bm25_results, vector_results):
    instance, _, _ = retriever
    instance.bm25.results = bm25_results
    instance.vector_retriever.results = vector_results

    results = instance.search("question", top_k=2, candidate_k=5)

    assert len(results) == 1


def test_raw_results_without_optional_metadata_are_preserved(retriever):
    instance, _, _ = retriever
    # HybridRetriever's real retrieval API only requires these four raw fields.
    instance.bm25.results = [_result("raw bm25", 4, 2.5, source="report.pdf")]
    instance.vector_retriever.results = [_result("raw vector", 5, 0.75, source="report.pdf")]

    results = instance.search("question", top_k=2, candidate_k=5)

    by_text = {result["text"]: result for result in results}
    assert by_text["raw bm25"]["source"] == "report.pdf"
    assert by_text["raw bm25"]["page"] == 4
    assert by_text["raw bm25"]["bm25_score"] == 2.5
    assert by_text["raw bm25"]["bm25_rank"] == 1
    assert by_text["raw bm25"]["vector_score"] is None
    assert by_text["raw vector"]["source"] == "report.pdf"
    assert by_text["raw vector"]["page"] == 5
    assert by_text["raw vector"]["vector_score"] == 0.75
    assert by_text["raw vector"]["vector_rank"] == 1
    assert by_text["raw vector"]["bm25_score"] is None
