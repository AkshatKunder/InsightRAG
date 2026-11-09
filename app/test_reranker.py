"""Deterministic unit tests for cross-encoder reranking."""

from __future__ import annotations

import importlib
import sys
import types

import pytest


@pytest.fixture
def reranker_module(monkeypatch: pytest.MonkeyPatch):
    """Import the wrapper with a CrossEncoder double, never a real model."""
    calls: dict[str, object] = {"scores": [0.2, 0.8]}

    class FakeCrossEncoder:
        def __init__(self, model_name: str):
            calls["model_name"] = model_name

        def predict(self, pairs):
            calls["pairs"] = pairs
            return calls["scores"]

    fake_package = types.ModuleType("sentence_transformers")
    fake_package.CrossEncoder = FakeCrossEncoder
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_package)
    monkeypatch.delitem(sys.modules, "app.rag.reranker", raising=False)
    module = importlib.import_module("app.rag.reranker")
    return module, calls


@pytest.fixture
def documents():
    return [
        {"text": "first passage", "source": "report.pdf", "page": 1, "score": 0.2, "rrf_score": 0.2},
        {"text": "second passage", "source": "report.pdf", "page": 2, "score": 0.8, "rrf_score": 0.8},
        {"text": "third passage", "source": "report.pdf", "page": 3, "score": 0.5, "rrf_score": 0.5},
    ]


def test_initializes_cross_encoder_with_configured_model(reranker_module):
    module, calls = reranker_module

    reranker = module.Reranker()

    assert calls["model_name"] == module.MODEL_NAME
    assert isinstance(reranker.model, module.CrossEncoder)


def test_constructs_query_document_pairs_and_requests_scores(reranker_module, documents):
    module, calls = reranker_module
    calls["scores"] = [0.1, 0.2, 0.3]

    module.Reranker().rerank("sales question", documents)

    assert calls["pairs"] == [
        ["sales question", "first passage"],
        ["sales question", "second passage"],
        ["sales question", "third passage"],
    ]


def test_scores_and_reranks_in_descending_final_score(reranker_module, documents):
    module, calls = reranker_module
    calls["scores"] = [0.1, 0.9, 0.5]

    results = module.Reranker().rerank("question", documents, top_k=3)

    assert [result["text"] for result in results] == ["second passage", "third passage", "first passage"]
    assert [result["final_score"] for result in results] == sorted(
        (result["final_score"] for result in results), reverse=True
    )
    assert [result["reranker_score"] for result in results] == [0.9, 0.5, 0.1]


def test_top_k_limits_reranked_results(reranker_module, documents):
    module, calls = reranker_module
    calls["scores"] = [0.1, 0.9, 0.5]

    results = module.Reranker().rerank("question", documents, top_k=2)

    assert [result["text"] for result in results] == ["second passage", "third passage"]


def test_preserves_original_metadata_and_adds_final_score(reranker_module):
    module, calls = reranker_module
    calls["scores"] = [0.7]
    original = {"text": "passage", "source": "report.pdf", "page": 7, "score": 0.4, "custom": "keep"}

    result = module.Reranker().rerank("question", [original])[0]

    assert result["text"] == "passage"
    assert result["source"] == "report.pdf"
    assert result["page"] == 7
    assert result["custom"] == "keep"
    assert result["reranker_score"] == 0.7
    assert result["final_score"] == pytest.approx(0.5)
    assert original == {"text": "passage", "source": "report.pdf", "page": 7, "score": 0.4, "custom": "keep"}


def test_empty_documents_return_empty_without_scoring(reranker_module):
    module, calls = reranker_module

    assert module.Reranker().rerank("question", []) == []
    assert "pairs" not in calls


def test_returns_all_documents_when_fewer_than_top_k(reranker_module):
    module, calls = reranker_module
    calls["scores"] = [0.4]

    results = module.Reranker().rerank("question", [{"text": "only", "score": 0.3}], top_k=5)

    assert len(results) == 1
    assert results[0]["text"] == "only"


def test_missing_optional_source_and_page_metadata_are_supported(reranker_module):
    module, calls = reranker_module
    calls["scores"] = [0.4]

    result = module.Reranker().rerank("question", [{"text": "minimal", "score": 0.3}])[0]

    assert result["text"] == "minimal"
    assert result["score"] == 0.3
    assert result["final_score"] == pytest.approx(0.5)
    assert "source" not in result
    assert "page" not in result


def test_equal_scores_preserve_input_order_deterministically(reranker_module, documents):
    module, calls = reranker_module
    calls["scores"] = [1.0, 1.0, 1.0]
    equal_rrf_documents = [{**document, "score": 0.5} for document in documents]

    results = module.Reranker().rerank("question", equal_rrf_documents, top_k=3)

    assert [result["text"] for result in results] == ["first passage", "second passage", "third passage"]
    assert all(result["final_score"] == pytest.approx(0.5) for result in results)
