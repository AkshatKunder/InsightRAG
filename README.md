# InsightRAG

A local hybrid RAG assistant for question answering over a PDF, with
claim-level citation validation.

Pipeline: PDF → word chunks → BM25 + vector search (Qdrant, `bge-small-en-v1.5`)
fused with weighted RRF → cross-encoder reranking → answer from a local Ollama
model → evidence selection and claim validation.

## Project structure

- `app/main.py` – interactive CLI question loop
- `app/ingest.py` – builds the Qdrant index from a PDF
- `app/rag/` – loader, chunker, retrievers, reranker, generator, citation selection/validation
- `scripts/` – retrieval diagnostics
- `data/documents/` – local PDFs (not committed)
- `data/qdrant/` – local vector index (not committed)

## Setup

1. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Install [Ollama](https://ollama.com) and pull the generation model:
   ```bash
   ollama pull llama3.2:3b
   ```
4. Put the document at `data/documents/annual_report.pdf`.

The embedding and reranker models are downloaded from Hugging Face on first use.

## Usage

Build (or rebuild) the index. This replaces any existing collection:

```bash
python -m app.ingest
```

Options: `--pdf <path>` and `--qdrant-path <dir>`.

Start the assistant (Ollama must be running):

```bash
python -m app.main
```

### Structured data (CSV / Excel) and natural language to SQL

Load CSV or `.xlsx` files into a local SQLite database, then ask questions that are routed to the
document, the data, or both:

```bash
python scripts/create_sample_datasets.py
python -m app.assistant --ingest data/structured/samples/segment_kpis.csv data/structured/samples/steel_orders_sample.xlsx
python -m app.assistant --ask "What is the total order revenue per customer region in the orders data?"
```

Generated SQL is validated and executed read-only; see [docs/structured_data.md](docs/structured_data.md).

## Tests

```bash
python -m pytest
```

Tests marked `integration` load real models and are skipped by default; run
them with `python -m pytest -m integration`.
