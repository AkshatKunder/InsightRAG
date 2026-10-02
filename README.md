# InsightRAG

A lightweight RAG-based application project structure for building document-based retrieval and generation workflows.

## Project structure

- `app/` – application source code
- `data/documents/` – local documents for ingestion or indexing
- `tests/` – application tests
- `.env` – environment variables
- `requirements.txt` – Python dependencies

## Quick start

1. Create a virtual environment:
   ```bash
   python -m venv .venv
   ```
2. Activate it:
   ```bash
   .venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Add your environment variables in `.env`.

## Notes

This is the initial scaffold for the InsightRAG project. You can extend it with indexing, chunking, retrievers, embeddings, and API routes as needed.
