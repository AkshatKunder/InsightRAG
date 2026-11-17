# InsightRAG

A local AI business-intelligence assistant. Ask questions in a chat about your **PDF documents** and
your **CSV / Excel datasets**:

* **Document questions** are answered with hybrid retrieval-augmented generation (RAG). Every answer
  is checked claim by claim against the cited pages, and the assistant abstains when the evidence does
  not support it.
* **Data questions** are translated to SQL. The SQL is validated and executed read-only. Changes,
  shares and ratios are calculated in Python from the raw rows, never by the model.
* **Hybrid questions** that name both sources are split. Each part is answered by its own engine, and
  the findings are shown side by side.

Everything runs locally: Ollama (`llama3.2:3b`), sentence-transformers models, embedded Qdrant and
SQLite. No paid API is used.

## Architecture

```text
┌──────────────────────────── Streamlit UI (app/streamlit_app.py, app/ui/components.py) ───────────────────────────┐
│ sidebar: conversations · documents · datasets · settings/status       chat: messages, tables, charts, citations │
└──────────────────────────────────────────────┬───────────────────────────────────────────────────────────────────┘
                                               │ display model (app/ui/presentation.py)
┌──────────────────────────────────────────────▼───────────────────────────────────────────────────────────────────┐
│ Application orchestration (app/ui/service.py)                                                                    │
│ health checks · error messages · lock · conversations (app/ui/conversations.py) · config (app/ui/config.py)      │
│ document library (app/ui/documents.py, app/ui/library.py) · dataset manager (app/ui/datasets.py)                 │
└──────────────┬──────────────────────────────────────────────────────────────────────────────┬────────────────────┘
               │ Assistant.ask(question)   (existing, unchanged)                              │ uploads
┌──────────────▼──────────────────────────── app/structured/engine.py ─────┐   ┌──────────────▼────────────────────┐
│ router.classify → "document" | "sql" | "hybrid" (split_hybrid)           │   │ PDF: load_pdf → chunk_pages →      │
│                                                                          │   │      EmbeddingModel → Qdrant       │
│  RAGPipeline (app/rag/pipeline.py)          StructuredQA                 │   │ CSV/XLSX: ingestion.ingest_file →  │
│  BM25 + vector (Qdrant) → RRF → rerank      NL→SQL (Ollama) → validate → │   │      SQLite (+ metadata tables)    │
│  → Ollama answer → evidence selection       read-only execute (authorizer,│  └────────────────────────────────────┘
│  → citation validation → abstention         timeout, row limit) → scope/ │
│                                              period/share/ratio checks → │
│                                              analytics + chart selection │
└──────────────────────────────────────────────────────────────────────────┘
        ▲ Ollama (local LLM)      ▲ data/qdrant (embedded)      ▲ data/structured/insightrag.db
```

The UI layer never routes questions, generates SQL or validates citations itself. It calls the existing
`engine.Assistant` and only arranges its result: raw query rows versus values derived in Python, SQL,
charts, citations and claim checks.

### Project structure

- `app/streamlit_app.py`: web application entry point
- `app/ui/`: UI components, presentation, orchestration service, document and dataset management,
  configuration, tests
- `app/main.py`: document-only command-line loop
- `app/assistant.py`: command line for documents + data (`--ingest`, `--ask`)
- `app/ingest.py`: builds the Qdrant index from one PDF (command line)
- `app/rag/`: loader, chunker, retrievers, reranker, generator, citation selection and validation
- `app/structured/`: ingestion, schema, router, NL→SQL, SQL safety, scope/period/share checks, charts
- `docs/structured_data.md`: details of the structured-data engine
- `scripts/create_sample_datasets.py`: recreates the sample CSV/XLSX datasets
- `data/evaluation/`: benchmark questions (reference data), the final Phase 4.3 evaluation report and the
  Phase 5 implementation report
- `data/documents/`: PDFs (not committed) · `data/qdrant/`: vector index (not committed) ·
  `data/structured/`: SQLite database and uploaded datasets (not committed)

## Prerequisites

- Python 3.11 (Windows, macOS or Linux)
- [Ollama](https://ollama.com) running locally
- About 3 GB of free disk space for the Ollama model; the embedding and reranker models (~200 MB) are
  downloaded from Hugging Face on first use.

## Installation

```bash
python -m venv .venv
```

```bash
.venv\Scripts\activate
```

(macOS/Linux: `source .venv/bin/activate`)

```bash
pip install -r requirements.txt
```

### Ollama model

```bash
ollama pull llama3.2:3b
```

Ollama must be running (the desktop app, or `ollama serve`). The app checks the server and the model at
start and shows their status under **Settings & status**.

### Qdrant

No server is needed. Qdrant runs embedded and stores the index in a local folder (`data/qdrant` by
default). An embedded Qdrant folder can be opened by **one process at a time**. Close
`python -m app.main` / `python -m app.assistant` before starting the web app (and vice versa).

To build the index for a PDF from the command line (this replaces the existing collection):

```bash
python -m app.ingest --pdf data/documents/annual_report.pdf
```

In the web app, uploaded PDFs are added to the existing index without rebuilding it.

### Environment configuration

All settings are optional environment variables. They can also go in a `.env` file in the project root,
which is read with python-dotenv; existing variables take precedence. No API keys or secrets are needed.

| Variable | Default | Meaning |
|---|---|---|
| `INSIGHTRAG_DOCUMENTS_DIR` | `data/documents` | PDF library folder |
| `INSIGHTRAG_QDRANT_PATH` | `data/qdrant` | Embedded Qdrant folder |
| `INSIGHTRAG_DB_PATH` | `data/structured/insightrag.db` | SQLite database |
| `INSIGHTRAG_DATASET_UPLOAD_DIR` | `data/structured/uploads` | Where uploaded CSV/XLSX files are kept |
| `INSIGHTRAG_MODEL` | `llama3.2:3b` | Ollama model |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server (read by the Ollama client) |
| `INSIGHTRAG_DETERMINISTIC` | `1` | `1`: temperature 0, seed 42; `0`: Ollama defaults |
| `INSIGHTRAG_MAX_PDF_MB` | `50` | PDF upload limit |
| `INSIGHTRAG_MAX_DATASET_MB` | `50` | Dataset upload limit (never above the ingestion limit of 50 MB) |

The theme, the Streamlit server upload limit and the disabled source-file watcher (enable auto-reload during
development with `--server.fileWatcherType auto`) are set in `.streamlit/config.toml`.

## Starting the application

```bash
streamlit run app/streamlit_app.py
```

Open http://localhost:8501. The first start parses the PDF library and loads the retrieval models. For
the 6 MB sample annual report this takes about 1–2 minutes on a laptop CPU. Later page loads and
questions do not repeat it.

## Using the application

**Documents.** In the sidebar, open **Documents**, choose one or more PDFs and click **Add documents**.
Each file is checked before it is stored:
- it must be a `.pdf` file with a `%PDF` header, within the size limit, readable, not
  password-protected and with a text layer (OCR is not supported);
- an existing file name or an identical file under another name is refused, so nothing is overwritten.

The text is chunked and embedded, and the chunks are appended to the index with their source file name
and page number. Every library document shows its pages, chunks and index state. If the index does not
match the library (for example after files were copied in by hand), the status says so; use
**Settings & status → Rebuild document index** (re-embeds every PDF).

**Datasets.** Open **Datasets**, choose `.csv` or `.xlsx` files and click **Add datasets**. Each Excel
sheet becomes a table. Column types are inferred and names made SQL-safe. A dataset with an existing
name is refused unless **Replace existing dataset** is ticked. **Browse tables and columns** lists each
table's columns, original headers, inferred types, missing/distinct counts and a 20-row read-only
preview. The UI never accepts SQL text.

**Chat.** Type a question or click an example. Each answer shows where it came from:
- **Document (RAG):** the answer, the validation status, the cited pages (expand a citation to read the
  evidence text), the claim-by-claim validation and any calculation from cited values.
- **Structured data (SQL):** the answer, the values derived in Python (period change, share of total,
  ratio; marked *calculated*), the raw query rows (marked *raw*), a chart when the chart rules allow one,
  validation notes, the tables used, and the validated SQL in an expandable section. Rejected or
  model-computed results are shown separately and marked as not used.
- **Hybrid:** structured-data findings and document evidence in two columns.

Use **New conversation** for a separate history and **Clear this conversation** to empty the current
one. Conversations last for the browser session. There are no accounts and nothing is persisted. Each
question is answered on its own; follow-up questions do not see earlier turns.

### Example questions

With the sample annual report and the sample datasets (`python scripts/create_sample_datasets.py`
creates them in `data/structured/samples/`; upload them in the app):

- What were ArcelorMittal's total sales in 2025?
- What are ArcelorMittal's reportable segments?
- What is the total order revenue per customer region in the orders data?
- How did Europe sales change between 2024 and 2025 in the segment KPI table?
- What share of 2025 employees in the segment KPI data does each segment have?
- Compare Europe's 2025 sales in the segment KPI table with the Europe sales figure in the annual
  report.

Questions about the data are recognised when they name the table or its columns or values ("in the
orders data", "in the segment KPI table").

## Troubleshooting

| Symptom | Fix |
|---|---|
| "Cannot reach Ollama at …" | Start Ollama (`ollama serve` or the desktop app), or set `OLLAMA_HOST`; then **Settings & status → Re-check**. |
| "The Ollama model '…' is not installed" | `ollama pull llama3.2:3b` (or the model in `INSIGHTRAG_MODEL`). |
| "The local vector index is in use by another process" | Close `python -m app.main`, `python -m app.assistant` or a second app instance, then **Re-check**. Data questions keep working meanwhile. |
| A document shows "not indexed" | **Settings & status → Rebuild document index**. |
| "has no extractable text" | The PDF is scanned; run OCR first (not built in). |
| A dataset upload says it is already loaded | Rename the file or tick **Replace existing dataset**. |
| A data question is answered from the document | Name the table or data ("… in the orders data"). |
| First start is slow | Expected once per server start (PDF parsing and model loading). |

Application logs go to the terminal running Streamlit. They record routes, timings and error types, but
not question text or data values.

## Command-line tools

```bash
python -m app.main
```

```bash
python -m app.assistant --ask "What is the total order revenue per customer region in the orders data?"
```

Generated SQL is validated and executed read-only; see [docs/structured_data.md](docs/structured_data.md).

## Tests

```bash
python -m pytest
```

Tests marked `integration` load real models and are skipped by default; run them with
`python -m pytest -m integration`. The application tests (`app/ui/test_ui.py`) use mocked backends and
need neither Ollama nor model downloads.
