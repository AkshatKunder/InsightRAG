"""The existing RAG pipeline over every PDF of the document library (Phase 5 adapter).

``HybridRetriever`` and ``build_scope_index`` are written for one PDF. This module adds only what is
needed for several:

* ``LibraryRetriever`` subclasses ``HybridRetriever``. Its chunks (and so its BM25 index) come from
  every library PDF, and its vector store is opened at the configured Qdrant path. ``search`` (query
  expansion, BM25, vector search and weighted RRF) is inherited unchanged.
* ``LibraryScopeIndex`` keeps one ``ScopeIndex`` per document, because chart values are keyed by page
  number and must not be mixed between documents.

``RAGPipeline`` (generation, evidence selection, citation validation, abstention) is used as is. With a
single PDF, the chunks and the scope index are the same as those built by ``build_pipeline``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from app.rag.bm25_retriever import BM25Retriever
from app.rag.chunker import chunk_pages
from app.rag.hybrid_retriever import HybridRetriever
from app.rag.loader import load_pdf


class LibraryRetriever(HybridRetriever):
    def __init__(self, pdf_paths: Sequence[Path | str], qdrant_path: Path | str, embedding_model=None,
                 vector_store=None, pages_by_source: Mapping[str, list] | None = None):
        # HybridRetriever.__init__ is deliberately not called: it reads a single PDF and opens the default store.
        from app.rag.retriever import Retriever

        if not pdf_paths:
            raise ValueError("No documents in the library.")
        self.pdf_paths = [Path(p) for p in pdf_paths]
        self.pages_by_source = dict(pages_by_source or {})
        self.chunks = []
        for path in self.pdf_paths:
            pages = self.pages_by_source.get(path.name)
            if pages is None:
                pages = self.pages_by_source[path.name] = load_pdf(path)
            self.chunks.extend(chunk_pages(pages))
        if not self.chunks:
            raise ValueError("The library documents contain no extractable text.")
        self.bm25 = BM25Retriever(self.chunks)
        if embedding_model is None:
            from app.rag.embeddings import EmbeddingModel

            embedding_model = EmbeddingModel()
        if vector_store is None:
            from app.rag.vector_store import VectorStore

            vector_store = VectorStore(path=str(qdrant_path))
        self.vector_retriever = Retriever.__new__(Retriever)
        self.vector_retriever.embedding_model = embedding_model
        self.vector_retriever.vector_store = vector_store

    def close(self) -> None:
        self.vector_retriever.vector_store.close()


class LibraryScopeIndex:
    """One ``ScopeIndex`` per document, with the same ``segment_names`` / ``enrich`` / ``metadata`` interface."""

    def __init__(self, indexes: Mapping[str, object]):
        self.indexes = dict(indexes)
        names: list[str] = []
        for index in self.indexes.values():
            names += [n for n in index.segment_names if n not in names]
        self.segment_names = names

    def metadata(self, chunk: Mapping) -> dict:
        index = self.indexes.get(chunk.get("source"))
        return index.metadata(chunk) if index is not None else {}

    def enrich(self, results: Iterable[Mapping]) -> list[dict]:
        out = []
        for item in results:
            index = self.indexes.get(item.get("source"))
            out.extend(index.enrich([item]) if index is not None else [dict(item)])
        return out


def build_scope_index(retriever: LibraryRetriever):
    from app.rag.chart_extractor import chart_values_by_page
    from app.rag.document_context import ScopeIndex

    indexes = {}
    for path in retriever.pdf_paths:
        try:
            charts = chart_values_by_page(path)
        except Exception:  # same tolerance as app.rag.pipeline.build_scope_index
            charts = {}
        chunks = [c for c in retriever.chunks if c["source"] == path.name]
        indexes[path.name] = ScopeIndex(chunks, retriever.pages_by_source[path.name], charts)
    if len(indexes) == 1:
        return next(iter(indexes.values()))  # identical to the single-PDF pipeline
    return LibraryScopeIndex(indexes)


def build_library_pipeline(pdf_paths: Sequence[Path | str], qdrant_path: Path | str, generation_options: dict | None = None,
                           model: str | None = None, embedding_model=None, reranker=None, vector_store=None,
                           pages_by_source: Mapping[str, list] | None = None):
    """``RAGPipeline`` with the default (Phase 3.1) configuration over the given PDFs."""

    from app.rag.generator import Generator
    from app.rag.pipeline import PipelineConfig, RAGPipeline
    from app.rag.reranker import Reranker

    retriever = LibraryRetriever(pdf_paths, qdrant_path, embedding_model=embedding_model, vector_store=vector_store,
                                 pages_by_source=pages_by_source)
    owns_store = vector_store is None
    try:
        config = PipelineConfig()
        scope_index = build_scope_index(retriever) if config.scope_context else None
        generator = Generator(options=generation_options)
        if model:
            generator.model = model
        return RAGPipeline(retriever, reranker or Reranker(), generator, config, scope_index)
    except Exception:
        if owns_store:
            retriever.close()
        raise
