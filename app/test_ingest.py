"""Fast tests for the ingestion pipeline using model and store doubles."""

from __future__ import annotations

from pathlib import Path

import pytest

from app import ingest as ingest_module


class FakeEmbeddingModel:
    def __init__(self):
        self.texts = None

    def embed_documents(self, texts):
        self.texts = list(texts)
        return [[float(i)] for i in range(len(texts))]


class FakeStore:
    def __init__(self):
        self.calls = []
        self.chunks = None
        self.embeddings = None

    def reset_collection(self):
        self.calls.append("reset")

    def add_documents(self, chunks, embeddings):
        self.calls.append("add")
        self.chunks = chunks
        self.embeddings = embeddings

    def count_documents(self):
        return len(self.chunks or [])

    def close(self):
        self.calls.append("close")


@pytest.fixture
def pdf_file(tmp_path: Path) -> Path:
    path = tmp_path / "report.pdf"
    path.write_bytes(b"%PDF-placeholder")
    return path


def _fake_pages(monkeypatch, pages):
    monkeypatch.setattr(ingest_module, "load_pdf", lambda path: pages)


def test_ingest_embeds_chunks_and_stores_them(monkeypatch, pdf_file):
    words = " ".join(f"w{i}" for i in range(500))
    _fake_pages(monkeypatch, [
        {"source": "report.pdf", "page": 1, "text": words},
        {"source": "report.pdf", "page": 2, "text": "short page"},
    ])
    model, store = FakeEmbeddingModel(), FakeStore()

    count = ingest_module.ingest(pdf_file, embedding_model=model, vector_store=store)

    # 500 words with chunk_size=400/overlap=100 -> 2 chunks, plus 1 for page 2.
    assert count == 3
    assert store.calls == ["reset", "add"]
    assert model.texts == [chunk["text"] for chunk in store.chunks]
    assert [chunk["page"] for chunk in store.chunks] == [1, 1, 2]
    assert len(store.embeddings) == 3


def test_ingest_uses_same_chunking_as_hybrid_retriever(monkeypatch, pdf_file):
    from app.rag.chunker import chunk_pages

    pages = [{"source": "report.pdf", "page": 1, "text": " ".join(["x"] * 900)}]
    _fake_pages(monkeypatch, pages)
    store = FakeStore()

    ingest_module.ingest(pdf_file, embedding_model=FakeEmbeddingModel(), vector_store=store)

    assert store.chunks == chunk_pages(pages)


def test_ingest_does_not_close_injected_store(monkeypatch, pdf_file):
    _fake_pages(monkeypatch, [{"source": "report.pdf", "page": 1, "text": "text"}])
    store = FakeStore()

    ingest_module.ingest(pdf_file, embedding_model=FakeEmbeddingModel(), vector_store=store)

    assert "close" not in store.calls


def test_ingest_rejects_missing_pdf(tmp_path):
    with pytest.raises(FileNotFoundError):
        ingest_module.ingest(tmp_path / "missing.pdf", embedding_model=FakeEmbeddingModel(), vector_store=FakeStore())


def test_ingest_rejects_pdf_without_text(monkeypatch, pdf_file):
    _fake_pages(monkeypatch, [])
    store = FakeStore()

    with pytest.raises(ValueError):
        ingest_module.ingest(pdf_file, embedding_model=FakeEmbeddingModel(), vector_store=store)

    assert store.calls == []


def test_main_passes_cli_arguments(monkeypatch):
    received = {}
    monkeypatch.setattr(
        ingest_module, "ingest", lambda pdf, qdrant: received.update(pdf=pdf, qdrant=qdrant)
    )

    ingest_module.main(["--pdf", "a.pdf", "--qdrant-path", "q"])

    assert received == {"pdf": "a.pdf", "qdrant": "q"}
