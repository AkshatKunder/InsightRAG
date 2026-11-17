"""Document library: PDF upload validation, storage and indexing (Phase 5).

* Uploads are validated (extension, ``%PDF`` signature, size, readable by pypdf, not encrypted,
  extractable text) before anything is written.
* A file is never overwritten: a name that already exists, or identical content stored under another
  name, is refused.
* Indexing reuses the existing ingestion pieces (``load_pdf``, ``chunk_pages``, ``EmbeddingModel``,
  ``VectorStore``). A new PDF is appended to the collection with fresh point ids. A full rebuild
  (``reset_collection`` + ``add_documents``, as in ``app.ingest``) runs only on request.
* ``index_status`` compares the vector count per source with the chunk count of each library PDF, so a
  missing, partial or stale index is reported instead of being answered from silently.
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from app.rag.chunker import chunk_pages
from app.rag.loader import load_pdf

log = logging.getLogger("insightrag.ui.documents")
_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._ -]+")


class DocumentError(ValueError):
    """A user-facing reason why a document cannot be added or indexed."""


@dataclass
class DocumentInfo:
    name: str
    path: Path
    size_bytes: int
    pages: int = 0
    chunks: int = 0
    indexed_vectors: int | None = None
    sha256: str = ""

    @property
    def indexed(self) -> bool:
        return self.indexed_vectors is not None and self.indexed_vectors == self.chunks and self.chunks > 0


@dataclass
class IndexStatus:
    ok: bool
    documents: list[DocumentInfo] = field(default_factory=list)
    total_vectors: int | None = None
    message: str = ""


def safe_filename(name: str) -> str:
    base = Path(str(name)).name  # drop any directory component
    stem, suffix = os.path.splitext(base)
    stem = _SAFE_NAME_RE.sub("_", stem).strip(" ._") or "document"
    return f"{stem[:120]}{suffix.lower()}"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def validate_pdf(filename: str, data: bytes, max_mb: int) -> list[dict]:
    """Return the PDF's text pages (as ``load_pdf`` would) or raise ``DocumentError``."""

    if Path(filename).suffix.lower() != ".pdf":
        raise DocumentError(f"'{filename}' is not a PDF. Only .pdf documents can be uploaded.")
    if not data:
        raise DocumentError(f"'{filename}' is empty.")
    if len(data) > max_mb * 1024 * 1024:
        raise DocumentError(f"'{filename}' is {len(data) / 1024 / 1024:.1f} MB; the limit is {max_mb} MB.")
    if not data.lstrip()[:5] == b"%PDF-":
        raise DocumentError(f"'{filename}' does not look like a PDF file (missing %PDF header).")
    try:
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            raise DocumentError(f"'{filename}' is password-protected; upload an unencrypted copy.")
        pages = []
        for number, page in enumerate(reader.pages, start=1):
            text = page.extract_text()
            if text and text.strip():
                pages.append({"source": safe_filename(filename), "page": number, "text": text.strip()})
    except DocumentError:
        raise
    except Exception as error:  # pypdf raises several error types for damaged files
        raise DocumentError(f"'{filename}' could not be read as a PDF ({type(error).__name__}).") from error
    if not pages:
        raise DocumentError(f"'{filename}' has no extractable text (scanned PDFs need OCR, which is not supported).")
    return pages


class DocumentLibrary:
    def __init__(self, documents_dir: Path | str, qdrant_path: Path | str, max_mb: int = 50):
        self.documents_dir = Path(documents_dir)
        self.qdrant_path = Path(qdrant_path)
        self.max_mb = max_mb
        self._cache: dict[tuple, tuple[str, int, int]] = {}  # (name, size, mtime) -> (sha, pages, chunks)
        self._pages: dict[tuple, list[dict]] = {}  # parsed text pages, shared with the retriever

    # ------------------------------------------------------------------ listing
    def pdf_paths(self) -> list[Path]:
        if not self.documents_dir.is_dir():
            return []
        return sorted(p for p in self.documents_dir.iterdir() if p.is_file() and p.suffix.lower() == ".pdf")

    @staticmethod
    def _key(path: Path) -> tuple:
        stat = path.stat()
        return (path.name, stat.st_size, stat.st_mtime_ns)

    def pages(self, path: Path) -> list[dict]:
        """``load_pdf`` output, parsed once per file version (large PDFs take tens of seconds)."""

        key = self._key(path)
        if key not in self._pages:
            self._pages[key] = load_pdf(path)
        return self._pages[key]

    def remember_pages(self, path: Path, pages: list[dict]) -> None:
        self._pages[self._key(path)] = pages

    def pages_by_source(self) -> dict[str, list[dict]]:
        return {p.name: self.pages(p) for p in self.pdf_paths()}

    def _facts(self, path: Path) -> tuple[str, int, int]:
        key = self._key(path)
        if key not in self._cache:
            pages = self.pages(path)
            self._cache[key] = (sha256(path.read_bytes()), len(pages), len(chunk_pages(pages)) if pages else 0)
        return self._cache[key]

    def documents(self) -> list[DocumentInfo]:
        docs = []
        for path in self.pdf_paths():
            try:
                digest, pages, chunks = self._facts(path)
            except Exception as error:
                log.warning("Could not read %s: %s", path.name, type(error).__name__)
                digest, pages, chunks = "", 0, 0
            docs.append(DocumentInfo(path.name, path, path.stat().st_size, pages, chunks, sha256=digest))
        return docs

    # ------------------------------------------------------------------ index status
    @staticmethod
    def _count(store, source: str | None = None) -> int:
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        from app.rag.vector_store import COLLECTION_NAME

        names = [c.name for c in store.client.get_collections().collections]
        if COLLECTION_NAME not in names:
            return 0
        flt = Filter(must=[FieldCondition(key="source", match=MatchValue(value=source))]) if source else None
        return store.client.count(collection_name=COLLECTION_NAME, count_filter=flt, exact=True).count

    def index_status(self, store) -> IndexStatus:
        """Vectors per document vs chunks per document, using an open ``VectorStore``."""

        docs = self.documents()
        if not docs:
            return IndexStatus(False, [], 0, "No PDF documents are in the library yet. Upload a PDF to ask document questions.")
        total = self._count(store)
        for doc in docs:
            doc.indexed_vectors = self._count(store, doc.name)
        expected = sum(d.chunks for d in docs)
        missing = [d.name for d in docs if not d.indexed]
        if missing:
            return IndexStatus(False, docs, total, "Not indexed (or partially indexed): " + ", ".join(missing)
                               + ". Use 'Rebuild document index' in Settings.")
        if total != expected:
            return IndexStatus(False, docs, total, f"The index holds {total} vectors but the library has {expected} chunks "
                               "(vectors of removed documents?). Use 'Rebuild document index' in Settings.")
        return IndexStatus(True, docs, total, f"{len(docs)} document(s), {total} indexed chunks.")

    # ------------------------------------------------------------------ upload
    def check_new(self, filename: str, data: bytes) -> tuple[str, list[dict]]:
        """Validate an upload and check that it does not overwrite or duplicate a library file."""

        name = safe_filename(filename)
        pages = validate_pdf(name, data, self.max_mb)
        target = self.documents_dir / name
        if target.exists():
            raise DocumentError(f"A document named '{name}' already exists. Rename the file to add it as a new document.")
        digest = sha256(data)
        for doc in self.documents():
            if doc.sha256 == digest:
                raise DocumentError(f"'{filename}' has the same content as '{doc.name}', which is already in the library.")
        return name, pages

    def save(self, name: str, data: bytes) -> Path:
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        target = self.documents_dir / name
        try:
            with open(target, "xb") as handle:  # exclusive create: never overwrites
                handle.write(data)
        except FileExistsError as error:
            raise DocumentError(f"A document named '{name}' already exists.") from error
        return target

    # ------------------------------------------------------------------ indexing
    @staticmethod
    def append_to_index(pages: list[dict], embedding_model, store) -> int:
        """Embed and append one document's chunks (same payload as ``VectorStore.add_documents``)."""

        from qdrant_client.models import PointStruct

        from app.rag.vector_store import COLLECTION_NAME

        chunks = chunk_pages(pages)
        if not chunks:
            raise DocumentError("The document produced no text chunks.")
        store.create_collection()
        start = DocumentLibrary._count(store)
        embeddings = embedding_model.embed_documents([c["text"] for c in chunks])
        points = [PointStruct(id=start + i, vector=e.tolist(),
                              payload={"text": c["text"], "source": c["source"], "page": c["page"]})
                  for i, (c, e) in enumerate(zip(chunks, embeddings))]
        try:
            store.client.upsert(collection_name=COLLECTION_NAME, points=points)
        except Exception:
            from qdrant_client.models import PointIdsList

            try:  # leave no partial document behind
                store.client.delete(collection_name=COLLECTION_NAME, points_selector=PointIdsList(points=[p.id for p in points]))
            except Exception:
                log.exception("Could not remove partially indexed points")
            raise
        return len(points)

    def rebuild_index(self, embedding_model, store) -> int:
        """Re-index every library PDF from scratch (``reset_collection`` + ``add_documents``)."""

        chunks = []
        for path in self.pdf_paths():
            chunks.extend(chunk_pages(self.pages(path)))
        store.reset_collection()
        if chunks:
            store.add_documents(chunks, embedding_model.embed_documents([c["text"] for c in chunks]))
        return store.count_documents()

    def remove_file(self, name: str) -> None:
        """Used only to undo a save whose indexing failed in the same request."""

        path = self.documents_dir / safe_filename(name)
        if path.is_file():
            path.unlink()
