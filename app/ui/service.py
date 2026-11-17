"""Application orchestration between the Streamlit UI and the existing InsightRAG services (Phase 5).

The UI never routes, generates SQL or validates citations itself. Every question goes to the existing
``engine.Assistant.ask`` (router, ``StructuredQA``, ``RAGPipeline``). This service:

* builds the backend lazily and keeps one Qdrant client open (local Qdrant locks its folder);
* reports component health (Ollama, model, vector index, documents, tables) so features can be
  disabled with a reason instead of failing;
* keeps structured questions available when document search is not, and says so;
* serialises queries and index changes with a lock (one Streamlit process serves every browser tab);
* turns backend exceptions into short user-facing messages and logs without question text or data.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from app.structured.engine import ABSTENTION, Assistant, StructuredQA, document_vocabulary

from app.ui.config import DETERMINISTIC_OPTIONS, AppConfig, redact_url
from app.ui.datasets import DatasetError, DatasetManager
from app.ui.documents import DocumentError, DocumentLibrary, IndexStatus

log = logging.getLogger("insightrag.ui.service")


@dataclass
class ComponentStatus:
    ok: bool
    message: str


@dataclass
class Health:
    ollama: ComponentStatus
    model: ComponentStatus
    vector_index: ComponentStatus
    documents: IndexStatus
    tables: ComponentStatus
    checked_at: float = field(default_factory=time.time)

    @property
    def can_answer_documents(self) -> bool:
        return self.ollama.ok and self.model.ok and self.vector_index.ok and self.documents.ok

    @property
    def can_answer_data(self) -> bool:
        return self.ollama.ok and self.model.ok and self.tables.ok


def explain_error(error: BaseException, config: AppConfig) -> str:
    """A short, actionable message; never the raw exception text of unknown errors."""

    text = str(error)
    name = type(error).__name__
    try:
        import ollama

        if isinstance(error, ollama.ResponseError):
            if getattr(error, "status_code", None) == 404 or "not found" in text.lower():
                return (f"The Ollama model '{config.model}' is not installed. Run `ollama pull {config.model}` and try again.")
            return f"Ollama returned an error ({getattr(error, 'status_code', 'unknown status')}). Check the Ollama server log."
    except ImportError:
        pass
    if isinstance(error, (ConnectionError, TimeoutError)) or name in ("ConnectError", "ConnectTimeout", "ReadTimeout"):
        return (f"Cannot reach Ollama at {redact_url(config.ollama_host)}. Start it (`ollama serve`, or open the Ollama app) and "
                "try again.")
    if "already accessed by another instance" in text:
        return ("The local vector index is in use by another process (e.g. `python -m app.main` or a second app "
                "instance). Close it, then use 'Re-check' in Settings & status.")
    if isinstance(error, (DocumentError, DatasetError)):
        return text
    return f"Unexpected error ({name}). The details were written to the application log."


class InsightService:
    def __init__(self, config: AppConfig, complete: Callable[[str, dict | None, str], str] | None = None,
                 rag_builder: Callable[..., object] | None = None, store_factory: Callable[[str], object] | None = None,
                 embedding_factory: Callable[[], object] | None = None, ollama_client=None):
        self.config = config
        self.documents = DocumentLibrary(config.documents_dir, config.qdrant_path, config.max_pdf_mb)
        self.datasets = DatasetManager(config.db_path, config.dataset_upload_dir, config.max_dataset_mb)
        self._complete_fn = complete
        self._rag_builder = rag_builder
        self._store_factory = store_factory
        self._embedding_factory = embedding_factory
        self._ollama_client = ollama_client
        self._lock = threading.RLock()
        self._store = None
        self._store_error: str | None = None
        self._embedding_model = None
        self._reranker = None
        self._rag = None
        self._rag_error: str | None = None
        self._health: Health | None = None
        self.structured = StructuredQA(config.db_path, complete=self._complete)

    # ------------------------------------------------------------------ LLM
    def _complete(self, prompt: str) -> str:
        options = self.config.generation_options or None
        if self._complete_fn is not None:
            return self._complete_fn(prompt, options, self.config.model)
        from app.rag.generator import complete

        return complete(prompt, options, model=self.config.model)[0]

    def set_deterministic(self, deterministic: bool) -> None:
        with self._lock:
            self.config.deterministic = deterministic
            self.config.generation_options = dict(DETERMINISTIC_OPTIONS) if deterministic else {}
            if self._rag is not None and hasattr(self._rag, "generator"):
                self._rag.generator.options = dict(self.config.generation_options) or None

    # ------------------------------------------------------------------ components
    def _open_store(self):
        if self._store is not None:
            return self._store
        try:
            if self._store_factory is not None:
                self._store = self._store_factory(str(self.config.qdrant_path))
            else:
                from app.rag.vector_store import VectorStore

                self._store = VectorStore(path=str(self.config.qdrant_path))
            self._store_error = None
        except Exception as error:  # locked folder, corrupted storage, ...
            log.warning("Vector index unavailable: %s", type(error).__name__)
            self._store_error = explain_error(error, self.config)
            self._store = None
        return self._store

    def _embeddings(self):
        if self._embedding_model is None:
            if self._embedding_factory is not None:
                self._embedding_model = self._embedding_factory()
            else:
                from app.rag.embeddings import EmbeddingModel

                self._embedding_model = EmbeddingModel()
        return self._embedding_model

    def _check_ollama(self) -> tuple[ComponentStatus, ComponentStatus]:
        try:
            if self._ollama_client is not None:
                client = self._ollama_client
            else:
                import ollama

                client = ollama.Client(host=self.config.ollama_host)
            listing = client.list()
            models = listing.get("models", []) if isinstance(listing, dict) else getattr(listing, "models", [])
            names = {getattr(m, "model", None) or (m.get("model") or m.get("name") if isinstance(m, dict) else None)
                     for m in models}
        except Exception as error:
            log.warning("Ollama check failed: %s", type(error).__name__)
            return (ComponentStatus(False, explain_error(ConnectionError(), self.config)),
                    ComponentStatus(False, "Unknown (Ollama is not reachable)."))
        wanted = self.config.model
        found = wanted in names or (":" not in wanted and f"{wanted}:latest" in names)
        model = ComponentStatus(True, f"{wanted} is installed.") if found else ComponentStatus(
            False, f"Model '{wanted}' is not installed. Run `ollama pull {wanted}`.")
        return ComponentStatus(True, f"Reachable at {redact_url(self.config.ollama_host)}."), model

    def health(self, refresh: bool = False) -> Health:
        with self._lock:
            if self._health is not None and not refresh:
                return self._health
            ollama_status, model_status = self._check_ollama()
            store = self._open_store()
            if store is None:
                vector = ComponentStatus(False, self._store_error or "The vector index could not be opened.")
                docs = IndexStatus(False, self.documents.documents(), None, "The vector index is unavailable.")
            else:
                vector = ComponentStatus(True, f"Local Qdrant at {self.config.qdrant_path}.")
                try:
                    docs = self.documents.index_status(store)
                except Exception as error:
                    log.warning("Index status failed: %s", type(error).__name__)
                    docs = IndexStatus(False, self.documents.documents(), None, explain_error(error, self.config))
            try:
                tables = self.structured.schema.tables
                tables_status = (ComponentStatus(True, f"{len(tables)} table(s) loaded.") if tables
                                 else ComponentStatus(False, "No datasets loaded yet. Upload a CSV or Excel file."))
            except Exception as error:
                log.warning("Schema discovery failed: %s", type(error).__name__)
                tables_status = ComponentStatus(False, "The structured database could not be read.")
            self._health = Health(ollama_status, model_status, vector, docs, tables_status)
            return self._health

    # ------------------------------------------------------------------ RAG pipeline
    def _build_rag(self) -> None:
        """(Re)build the document pipeline over the library; sets ``_rag`` or ``_rag_error``."""

        self._rag, self._rag_error = None, None
        health = self.health(refresh=True)
        if not health.vector_index.ok:
            self._rag_error = health.vector_index.message
            return
        if not health.documents.ok:
            self._rag_error = health.documents.message
            return
        try:
            paths = self.documents.pdf_paths()
            if self._rag_builder is not None:
                self._rag = self._rag_builder(paths, store=self._store, embedding_model=self._embeddings(),
                                              options=self.config.generation_options or None, model=self.config.model)
            else:
                self._rag = self._default_rag(paths)
        except Exception as error:
            log.exception("Document pipeline could not be built")
            self._rag_error = explain_error(error, self.config)

    def _default_rag(self, paths):
        from app.rag.reranker import Reranker

        from app.ui.library import build_library_pipeline

        if self._reranker is None:
            self._reranker = Reranker()
        return build_library_pipeline(paths, self.config.qdrant_path, self.config.generation_options or None,
                                      model=self.config.model, embedding_model=self._embeddings(),
                                      reranker=self._reranker, vector_store=self._store,
                                      pages_by_source=self.documents.pages_by_source())

    def document_search_status(self) -> ComponentStatus:
        if self._rag is not None:
            return ComponentStatus(True, "Ready.")
        return ComponentStatus(False, self._rag_error or "Not loaded yet (it loads with the first question).")

    def warm_up(self) -> None:
        with self._lock:
            if self._rag is None:
                self._build_rag()

    def _rag_answer(self, question: str) -> dict:
        if self._rag is None:
            reason = self._rag_error or "document search is not available"
            return {"answer": f"Document search is unavailable: {reason}", "citations": [], "context": [],
                    "validation": {"overall_status": "ABSTAINED", "reason": reason, "claims": []}, "unavailable": reason}
        return self._rag.answer(question)

    # ------------------------------------------------------------------ questions
    def ask(self, question: str) -> dict:
        """``{"ok": bool, "result": <Assistant result> | None, "error": str | None, "elapsed_s": float}``."""

        question = (question or "").strip()
        if not question:
            return {"ok": False, "result": None, "error": "Please enter a question.", "elapsed_s": 0.0}
        started = time.perf_counter()
        with self._lock:
            try:
                if self._rag is None:
                    self._build_rag()
                vocab = document_vocabulary(self._rag.retriever.chunks) if self._rag is not None else set()
                assistant = Assistant(self.structured, self._rag_answer, vocab)
                result = assistant.ask(question)
                elapsed = time.perf_counter() - started
                log.info("Answered question (route=%s, chars=%d, %.1fs)", result.get("route"), len(question), elapsed)
                return {"ok": True, "result": result, "error": None, "elapsed_s": elapsed}
            except Exception as error:
                elapsed = time.perf_counter() - started
                log.exception("Question failed (chars=%d)", len(question))
                if self._health is not None:
                    self._health = None  # re-check components after a failure
                return {"ok": False, "result": None, "error": explain_error(error, self.config), "elapsed_s": elapsed}

    # ------------------------------------------------------------------ uploads
    def add_document(self, filename: str, data: bytes) -> dict:
        with self._lock:
            name, pages = self.documents.check_new(filename, data)
            store = self._open_store()
            if store is None:
                raise DocumentError(self._store_error or "The vector index is unavailable; the document was not added.")
            path = self.documents.save(name, data)
            self.documents.remember_pages(path, pages)
            try:
                vectors = self.documents.append_to_index(pages, self._embeddings(), store)
            except Exception as error:
                self.documents.remove_file(name)
                log.exception("Indexing failed; the uploaded file was removed")
                raise DocumentError(f"'{name}' could not be indexed: {explain_error(error, self.config)}") from error
            log.info("Document added (%d pages, %d chunks)", len(pages), vectors)
            self._build_rag()
            return {"name": path.name, "pages": len(pages), "chunks": vectors,
                    "document_search": self.document_search_status().message}

    def rebuild_index(self) -> dict:
        with self._lock:
            store = self._open_store()
            if store is None:
                raise DocumentError(self._store_error or "The vector index is unavailable.")
            count = self.documents.rebuild_index(self._embeddings(), store)
            log.info("Document index rebuilt (%d vectors)", count)
            self._build_rag()
            return {"vectors": count, "document_search": self.document_search_status().message}

    def add_dataset(self, filename: str, data: bytes, replace: bool = False) -> dict:
        with self._lock:
            result = self.datasets.upload(filename, data, replace=replace)
            self.structured.refresh()
            self._health = None
            log.info("Dataset loaded (%d table(s), replaced=%s)", len(result.tables), result.replaced)
            return {"dataset": result.dataset, "tables": result.tables, "replaced": result.replaced}

    def retry_components(self) -> Health:
        with self._lock:
            if self._store is None:
                self._store_error = None
            self._rag, self._rag_error = None, None
            return self.health(refresh=True)

    def close(self) -> None:
        with self._lock:
            if self._store is not None:
                try:
                    self._store.close()
                except Exception:
                    pass
                self._store = None
            self._rag = None


__all__ = ["ABSTENTION", "ComponentStatus", "Health", "InsightService", "explain_error"]
