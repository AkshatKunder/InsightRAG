"""Deterministic unit tests for lexical BM25 retrieval."""

from __future__ import annotations

import pytest

from app.rag import bm25_retriever


@pytest.fixture
def chunks():
    return [
        {"text": "steel sales revenue increased", "source": "report.pdf", "page": 1},
        {"text": "employee workforce numbers increased", "source": "report.pdf", "page": 2},
        {"text": "sales sales sales annual sales", "source": "report.pdf", "page": 3},
    ]


def test_initializes_with_valid_chunks(chunks):
    retriever = bm25_retriever.BM25Retriever(chunks)

    assert retriever.chunks is chunks
    assert retriever.bm25 is not None


def test_tokenizes_documents_to_lowercase_words(monkeypatch: pytest.MonkeyPatch):
    captured = {}

    class FakeBM25:
        def __init__(self, tokenized_chunks):
            captured["chunks"] = tokenized_chunks

    monkeypatch.setattr(bm25_retriever, "BM25Okapi", FakeBM25)
    bm25_retriever.BM25Retriever([
        {"text": "Total SALES, Increased", "source": "sample.pdf", "page": 1},
    ])

    assert captured["chunks"] == [["total", "sales,", "increased"]]


def test_search_returns_relevant_results_sorted_by_descending_score(chunks):
    retriever = bm25_retriever.BM25Retriever(chunks)

    results = retriever.search("sales", top_k=3)

    assert results[0]["text"] == "sales sales sales annual sales"
    assert [result["score"] for result in results] == sorted(
        (result["score"] for result in results), reverse=True
    )


def test_top_k_limits_results(chunks):
    retriever = bm25_retriever.BM25Retriever(chunks)

    assert len(retriever.search("sales", top_k=1)) == 1


def test_empty_query_returns_no_results(chunks):
    retriever = bm25_retriever.BM25Retriever(chunks)

    assert retriever.search("", top_k=5) == []
    assert retriever.search("   ", top_k=5) == []


def test_empty_chunk_collection_returns_no_results():
    retriever = bm25_retriever.BM25Retriever([])

    assert retriever.bm25 is None
    assert retriever.search("sales") == []


def test_result_preserves_chunk_metadata(chunks):
    retriever = bm25_retriever.BM25Retriever(chunks)

    result = retriever.search("workforce", top_k=1)[0]

    assert result["text"] == "employee workforce numbers increased"
    assert result["source"] == "report.pdf"
    assert result["page"] == 2
    assert isinstance(result["score"], float)


def test_queries_without_matching_terms_return_zero_score_results(chunks):
    retriever = bm25_retriever.BM25Retriever(chunks)

    results = retriever.search("unrelatedterm", top_k=2)

    assert len(results) == 2
    assert all(result["score"] == 0.0 for result in results)
