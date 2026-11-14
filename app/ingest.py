"""Build the local Qdrant index from a PDF.

Run from the project root:

    python -m app.ingest
    python -m app.ingest --pdf data/documents/annual_report.pdf --qdrant-path data/qdrant

The collection is rebuilt from scratch on every run. Chunking uses the same
defaults as ``HybridRetriever`` so vector and BM25 results refer to identical
chunks and can be fused.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from app.rag.chunker import chunk_pages
from app.rag.loader import load_pdf


DEFAULT_PDF_PATH = "data/documents/annual_report.pdf"
DEFAULT_QDRANT_PATH = "data/qdrant"


def ingest(pdf_path, qdrant_path=DEFAULT_QDRANT_PATH, embedding_model=None, vector_store=None):
    """
    Load, chunk, embed and store a PDF. Returns the number of stored vectors.

    ``embedding_model`` and ``vector_store`` can be injected (e.g. in tests);
    otherwise the production models are created here.
    """

    pdf_path = Path(pdf_path)
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pages = load_pdf(pdf_path)
    if not pages:
        raise ValueError(f"No extractable text found in {pdf_path}")

    chunks = chunk_pages(pages)
    print(f"Loaded {len(pages)} pages -> {len(chunks)} chunks from {pdf_path.name}")

    if embedding_model is None:
        from app.rag.embeddings import EmbeddingModel

        embedding_model = EmbeddingModel()

    owns_store = vector_store is None
    if owns_store:
        from app.rag.vector_store import VectorStore

        vector_store = VectorStore(path=str(qdrant_path))

    try:
        embeddings = embedding_model.embed_documents(
            [chunk["text"] for chunk in chunks]
        )

        vector_store.reset_collection()
        vector_store.add_documents(chunks, embeddings)

        count = vector_store.count_documents()
        print(f"Index ready: {count} vectors in {qdrant_path}")
        return count
    finally:
        if owns_store:
            vector_store.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Index a PDF into the local Qdrant store.")
    parser.add_argument("--pdf", default=DEFAULT_PDF_PATH, help=f"PDF to index (default: {DEFAULT_PDF_PATH})")
    parser.add_argument(
        "--qdrant-path",
        default=DEFAULT_QDRANT_PATH,
        help=f"Local Qdrant directory (default: {DEFAULT_QDRANT_PATH})",
    )
    args = parser.parse_args(argv)

    ingest(args.pdf, args.qdrant_path)


if __name__ == "__main__":
    main()
