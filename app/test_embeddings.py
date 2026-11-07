"""Unit tests for embeddings.

Run the real-model check explicitly with:
``.venv\\Scripts\\python.exe -m pytest -m integration app/test_embeddings.py``.
"""

from __future__ import annotations

import importlib
import math
import sys
import types

import pytest


@pytest.fixture
def embedding_module(monkeypatch: pytest.MonkeyPatch):
    """Import the wrapper with a model double, never the real package/model."""
    calls: dict[str, object] = {}

    class FakeSentenceTransformer:
        def __init__(self, model_name: str):
            calls["model_name"] = model_name

        def encode(self, value, **kwargs):
            calls["encode_value"] = value
            calls["encode_kwargs"] = kwargs
            return calls["output"]

    calls["output"] = object()
    fake_package = types.ModuleType("sentence_transformers")
    fake_package.SentenceTransformer = FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_package)
    monkeypatch.delitem(sys.modules, "app.rag.embeddings", raising=False)
    module = importlib.import_module("app.rag.embeddings")
    return module, calls


def test_initializes_sentence_transformer_with_configured_model(embedding_module):
    module, calls = embedding_module

    embedding_model = module.EmbeddingModel()

    assert calls["model_name"] == module.MODEL_NAME
    assert isinstance(embedding_model.model, module.SentenceTransformer)


def test_embed_documents_calls_encode_with_expected_arguments(embedding_module):
    module, calls = embedding_module
    embedding_model = module.EmbeddingModel()
    texts = ["first document", "second document"]

    result = embedding_model.embed_documents(texts)

    assert calls["encode_value"] == texts
    assert calls["encode_kwargs"] == {
        "normalize_embeddings": True,
        "show_progress_bar": True,
    }
    assert result is calls["output"]


def test_embed_query_calls_encode_with_expected_arguments(embedding_module):
    module, calls = embedding_module
    embedding_model = module.EmbeddingModel()

    result = embedding_model.embed_query("What were total sales?")

    assert calls["encode_value"] == "What were total sales?"
    assert calls["encode_kwargs"] == {"normalize_embeddings": True}
    assert result is calls["output"]


@pytest.mark.integration
def test_real_model_produces_normalized_384_dimension_embeddings(monkeypatch: pytest.MonkeyPatch):
    """Opt-in smoke test; may download/load the local BGE model."""
    monkeypatch.delitem(sys.modules, "app.rag.embeddings", raising=False)
    module = importlib.import_module("app.rag.embeddings")
    embedding_model = module.EmbeddingModel()

    embeddings = embedding_model.embed_documents([
        "Total sales increased during the reporting period.",
        "The workforce includes full-time equivalent employees.",
    ])

    assert embeddings.shape == (2, 384)
    assert all(math.isfinite(float(value)) for row in embeddings for value in row)
    assert all(abs(sum(float(value) ** 2 for value in row) - 1.0) < 1e-5 for row in embeddings)
