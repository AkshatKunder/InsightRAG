"""Safe unit and opt-in integration tests for the Qdrant vector store.

Integration tests use pytest-managed temporary directories only.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.rag import vector_store


class _Vector(list):
    """Small ndarray-compatible vector double used without embedding models."""

    def tolist(self) -> list[float]:
        return list(self)


def _vector(value: float = 0.0) -> _Vector:
    return _Vector([value] * vector_store.VECTOR_SIZE)


@pytest.fixture
def mocked_store(monkeypatch: pytest.MonkeyPatch):
    """Construct a VectorStore with an in-memory Qdrant client double."""
    calls: dict[str, object] = {}

    class FakeClient:
        def __init__(self, path: str):
            calls["path"] = path
            self.collections = []

        def get_collections(self):
            return SimpleNamespace(collections=self.collections)

        def create_collection(self, **kwargs):
            calls["create_collection"] = kwargs

        def upsert(self, **kwargs):
            calls["upsert"] = kwargs

        def count(self, **kwargs):
            calls["count"] = kwargs
            return SimpleNamespace(count=7)

        def close(self):
            calls["closed"] = True

    monkeypatch.setattr(vector_store, "QdrantClient", FakeClient)
    store = vector_store.VectorStore(path="unit-test-qdrant")
    return store, calls


def test_initializes_qdrant_client_with_configured_path(mocked_store):
    _, calls = mocked_store

    assert calls["path"] == "unit-test-qdrant"


def test_create_collection_when_it_does_not_exist(mocked_store):
    store, calls = mocked_store

    store.create_collection()

    created = calls["create_collection"]
    assert created["collection_name"] == vector_store.COLLECTION_NAME
    assert created["vectors_config"].size == vector_store.VECTOR_SIZE
    assert created["vectors_config"].distance == vector_store.Distance.COSINE


def test_create_collection_does_not_recreate_existing_collection(mocked_store):
    store, calls = mocked_store
    store.client.collections = [SimpleNamespace(name=vector_store.COLLECTION_NAME)]

    store.create_collection()

    assert "create_collection" not in calls


def test_add_documents_creates_points_with_vectors_and_metadata(mocked_store):
    store, calls = mocked_store
    chunks = [
        {"text": "first chunk", "source": "sample.pdf", "page": 1},
        {"text": "second chunk", "source": "sample.pdf", "page": 2},
    ]

    store.add_documents(chunks, [_vector(0.1), _vector(0.2)])

    upsert = calls["upsert"]
    assert upsert["collection_name"] == vector_store.COLLECTION_NAME
    points = upsert["points"]
    assert [point.id for point in points] == [0, 1]
    assert [point.vector for point in points] == [_vector(0.1), _vector(0.2)]
    assert [point.payload for point in points] == [
        {"text": "first chunk", "source": "sample.pdf", "page": 1},
        {"text": "second chunk", "source": "sample.pdf", "page": 2},
    ]


def test_count_documents_returns_client_count(mocked_store):
    store, calls = mocked_store

    assert store.count_documents() == 7
    assert calls["count"] == {"collection_name": vector_store.COLLECTION_NAME, "exact": True}


def test_close_closes_qdrant_client(mocked_store):
    store, calls = mocked_store

    store.close()

    assert calls["closed"] is True


@pytest.mark.integration
def test_temporary_qdrant_store_round_trip(tmp_path):
    """Exercise Qdrant locally without reading or writing the project database."""
    store = vector_store.VectorStore(path=str(tmp_path / "qdrant"))
    chunks = [
        {"text": "first synthetic chunk", "source": "integration.pdf", "page": 1},
        {"text": "second synthetic chunk", "source": "integration.pdf", "page": 2},
    ]
    try:
        store.create_collection()
        store.add_documents(chunks, [_vector(0.1), _vector(0.2)])

        assert store.count_documents() == 2
        points = store.client.retrieve(
            collection_name=vector_store.COLLECTION_NAME,
            ids=[0, 1],
            with_payload=True,
            with_vectors=False,
        )
        assert [point.payload for point in points] == chunks
    finally:
        store.close()
