"""Mock-only unit tests for vector retrieval."""

from __future__ import annotations

import importlib
import sys
import types

import pytest


class _Embedding(list):
    def tolist(self) -> list[float]:
        return list(self)


def _point(text: str, source: str, page: int, score: float, **extra):
    return types.SimpleNamespace(
        payload={"text": text, "source": source, "page": page, **extra},
        score=score,
    )


@pytest.fixture
def retriever_module(monkeypatch: pytest.MonkeyPatch):
    """Load the real wrapper after replacing model and database dependencies."""
    calls: dict[str, object] = {"embedding": _Embedding([0.1, 0.2, 0.3]), "points": []}

    class FakeEmbeddingModel:
        def embed_query(self, query: str):
            calls["embed_query"] = query
            return calls["embedding"]

    class FakeClient:
        def query_points(self, **kwargs):
            calls["query_points"] = kwargs
            return types.SimpleNamespace(points=calls["points"])

    class FakeVectorStore:
        def __init__(self):
            self.client = FakeClient()
            calls["vector_store"] = self

    embeddings = types.ModuleType("app.rag.embeddings")
    embeddings.EmbeddingModel = FakeEmbeddingModel
    vector_store = types.ModuleType("app.rag.vector_store")
    vector_store.VectorStore = FakeVectorStore
    monkeypatch.setitem(sys.modules, "app.rag.embeddings", embeddings)
    monkeypatch.setitem(sys.modules, "app.rag.vector_store", vector_store)
    monkeypatch.delitem(sys.modules, "app.rag.retriever", raising=False)
    module = importlib.import_module("app.rag.retriever")
    return module, calls


def test_initializes_embedding_model_and_vector_store(retriever_module):
    module, calls = retriever_module

    retriever = module.Retriever()

    assert isinstance(retriever.embedding_model, module.EmbeddingModel)
    assert retriever.vector_store is calls["vector_store"]


def test_embeds_query_and_requests_vector_search_with_top_k(retriever_module):
    module, calls = retriever_module
    retriever = module.Retriever()

    retriever.search("employee count", top_k=7)

    assert calls["embed_query"] == "employee count"
    assert calls["query_points"] == {
        "collection_name": "insightrag_documents",
        "query": [0.1, 0.2, 0.3],
        "limit": 7,
        "with_payload": True,
    }


def test_transforms_vector_points_and_preserves_result_metadata(retriever_module):
    module, calls = retriever_module
    calls["points"] = [
        _point("first chunk", "report.pdf", 3, 0.91),
        _point("second chunk", "report.pdf", 4, 0.82),
    ]

    results = module.Retriever().search("sales", top_k=2)

    assert results == [
        {"text": "first chunk", "source": "report.pdf", "page": 3, "score": 0.91},
        {"text": "second chunk", "source": "report.pdf", "page": 4, "score": 0.82},
    ]


def test_forwards_empty_query_without_initializing_external_services(retriever_module):
    module, calls = retriever_module

    assert module.Retriever().search("", top_k=1) == []
    assert calls["embed_query"] == ""
    assert calls["query_points"]["limit"] == 1


def test_returns_empty_list_when_vector_store_has_no_matches(retriever_module):
    module, calls = retriever_module
    calls["points"] = []

    assert module.Retriever().search("unmatched", top_k=5) == []


def test_required_payload_only_is_supported_and_optional_metadata_is_ignored(retriever_module):
    module, calls = retriever_module
    calls["points"] = [
        _point("minimal chunk", "report.pdf", 9, 0.5, section="financials", unused_id="abc"),
    ]

    results = module.Retriever().search("sales", top_k=1)

    assert results == [{"text": "minimal chunk", "source": "report.pdf", "page": 9, "score": 0.5}]
