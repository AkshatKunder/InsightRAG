# InsightRAG Phase 5: user interface and application

Phase 5 adds a local Streamlit web application on top of the existing InsightRAG backend. The RAG
pipeline, router, NL→SQL engine, SQL safety layer and validators are **unchanged**: no file under
`app/rag/` or `app/structured/` was modified. Every question goes through the existing
`engine.Assistant.ask`.

## 1. Architecture and files added

```text
Streamlit UI ─ app/streamlit_app.py (pages, sidebar, chat) + app/ui/components.py (rendering only)
      │  display model: app/ui/presentation.py (raw vs derived, citations, SQL, chart, notes)
Orchestration ─ app/ui/service.py (lazy backend, health checks, error messages, lock, uploads)
      │         app/ui/conversations.py (in-session conversations) · app/ui/config.py (environment settings)
      │         app/ui/documents.py (PDF validation/storage/indexing) · app/ui/datasets.py (CSV/XLSX)
      │         app/ui/library.py (existing HybridRetriever/ScopeIndex over several PDFs)
Existing services ─ engine.Assistant → router · RAGPipeline · StructuredQA (nl2sql, sql_safety, scope,
                    periods, shares, analytics, charts) · ingestion · Qdrant · SQLite · Ollama
```

The full diagram is in `README.md` (section "Architecture").

| File | Purpose |
|---|---|
| `app/streamlit_app.py` | Entry point: sidebar (branding, conversations, documents, datasets, settings/status), chat, dataset browser, welcome screen |
| `app/ui/components.py` | Streamlit rendering of answers, tables, charts (Altair), citations, claim checks, welcome screen |
| `app/ui/presentation.py` | Pure display model built from an `Assistant` result. Labels raw values, values derived in Python, rejected results and model-computed columns that were ignored. No Streamlit import |
| `app/ui/service.py` | Orchestration: builds the backend lazily, keeps one Qdrant client open, health checks, `explain_error`, lock, document/dataset uploads, rebuild index |
| `app/ui/documents.py` | PDF validation, no-overwrite storage, duplicate detection (name and SHA-256), append to the index, rebuild, per-document index status |
| `app/ui/library.py` | `LibraryRetriever` (a `HybridRetriever` subclass over several PDFs; `search` inherited unchanged), `LibraryScopeIndex` (one `ScopeIndex` per document), `build_library_pipeline` |
| `app/ui/datasets.py` | CSV/XLSX validation, storage, ingestion via `ingest_file`, explicit replace, schema listing, fixed read-only previews |
| `app/ui/conversations.py` | In-session conversations: new, select, delete, clear, interrupted/failed questions |
| `app/ui/config.py` | Settings from environment variables / `.env`; host URLs are shown without credentials |
| `app/ui/test_ui.py` | 22 tests with mocked backends |
| `.streamlit/config.toml` | Dark theme, upload limit, minimal toolbar, file watcher off, no usage statistics |

Modified: `README.md` (rewritten for the application) and `requirements.txt` (`streamlit>=1.40`, the only
new dependency; pandas and Altair come with Streamlit).

## 2. Features implemented

**Interface (Task 3).**
- Dark theme with InsightRAG branding.
- Sidebar: new conversation, conversation list (switch/delete), documents (list with pages, chunks
  and index state, plus upload), datasets (list, upload, "Browse tables and columns"), and Settings &
  status.
  - Settings & status shows Ollama, model, vector index, document search and datasets, each with a
    reason.
  - It also has a deterministic-generation toggle, a routing-details toggle, Re-check, Rebuild
    document index and the active configuration.
- Chat: messages, a spinner while answering, the elapsed time per answer and error boxes.
- Welcome screen: introduction, RAG/SQL/hybrid cards, supported formats and clickable example
  questions.

**Document management (Task 4).**
- Validation before anything is written:
  - `.pdf` extension, `%PDF` header and size limit (50 MB by default);
  - readable by pypdf, not encrypted, with a text layer.
- Storage in the library folder with exclusive create, so nothing is overwritten. An existing name or
  an identical file (SHA-256) is refused.
- The new PDF's chunks are embedded and appended to the existing Qdrant collection with new point
  ids. The payload is the same as `VectorStore.add_documents` (text, source file name, page). If
  indexing fails, the file and any partial points are removed.
- Per-document index status: vectors per source compared with chunks per document. A missing,
  partial or stale index is reported. Rebuild index runs only on request.

**Dataset management (Task 5).**
- CSV/XLSX upload:
  - extension, size and xlsx signature are checked;
  - ingestion goes through `ingest_file` (type inference, safe identifiers, one transaction);
  - uploads are stored without overwrite;
  - an existing dataset is replaced only when the user ticks **Replace**.
- The dataset browser shows each table's columns, original headers, inferred types, missing/distinct
  counts and examples, plus a 20-row preview.
- Previews are fixed `SELECT * FROM "<schema table>" LIMIT n` statements run through
  `sql_safety.execute`. The table name must come from the discovered schema. The UI has no SQL input.

**Chat orchestration (Tasks 6 and 8).**
- `InsightService.ask` calls `Assistant.ask`, so routing, hybrid splitting, citation validation,
  evidence selection, abstention, read-only SQL, the authorizer, timeouts, row limits, scope
  validation and the deterministic change/share/ratio calculations are all the existing code paths.
- Conversations are separate per browser session and per conversation, and nothing is persisted.
- A question left unanswered by a stopped run is marked "interrupted" on the next run. Failed
  questions keep their error message.

**Results presentation (Task 7).**
- Every answer is labelled **Document (RAG)**, **Structured data (SQL)** or **Hybrid**, and abstentions
  are flagged.
- SQL answers show:
  - the engine's answer text;
  - "Derived" tables (period change, share of total, ratio, column comparison), marked *calculated*,
    with a note that Python computed them;
  - the raw query rows, marked *raw*, with units taken from column suffixes (USD m, FTE, kt, …);
  - the chart chosen by `charts.choose_chart` (pie/bar/line, drawn with Altair) and the reason for it;
  - validation notes (verified denominator, scope filters, regenerated query, ignored model-computed
    columns);
  - the tables used;
  - the SQL in an expander;
  - the original result before model-computed columns were removed;
  - rejected results, marked as not used.
- Document answers show:
  - the validation status badge and the reason;
  - the source files and pages;
  - one expander per citation with the evidence text;
  - claim-by-claim validation;
  - any calculation from cited values.
- Hybrid answers show the split sub-questions and two columns: structured-data findings and document
  evidence.

**Configuration and reliability (Task 9).**
- Environment variables (`INSIGHTRAG_*`, `OLLAMA_HOST`) and `.env` (python-dotenv, already a
  dependency). No secrets are needed or stored, and host URLs are shown without credentials.
- Health checks report whether Ollama is reachable, the model is installed, and the Qdrant folder is
  available (it is locked while another process uses it).
- `explain_error` turns these exceptions into short messages:
  - Ollama connection errors and missing models;
  - a locked Qdrant folder;
  - ingestion errors and unknown errors. For unknown errors only the exception type is shown, never
    its text.
- Degraded operation: data questions keep working when document search is unavailable, and a
  document question then explains why it cannot be answered.
- A lock serialises questions and index changes.
- Logs record route, question length, timing and error types, never question text or data values.

## 3. Tests

| Run | Result |
|---|---|
| Full suite `python -m pytest` | **506 passed**, 2 deselected (integration). Before Phase 5: 484 passed, so all Phase 4.3 tests still pass |
| New `app/ui/test_ui.py` | 22 passed (about 7 s; no Ollama, no model downloads, throw-away Qdrant/SQLite under `tmp_path`) |

New tests by area of the specification:
- **Startup:** the Streamlit `AppTest` runs `app/streamlit_app.py` with a mocked service. It checks
  that the page renders, the example questions appear, a chat question renders the route badge, the
  table and the SQL, and New conversation returns to the welcome screen.
- **Upload validation:**
  - PDF: wrong extension, missing header, empty, too large, no text, damaged, and path stripping in
    file names.
  - Dataset: wrong extension, empty, too large, fake xlsx, header-only CSV. Rejected files are not
    kept.
- **Document ingestion:** a real embedded Qdrant with fake embeddings. Covers the index status
  before and after indexing, kept source/page payloads, refusal of duplicate names and identical
  content, no overwrite, append ids, rebuild, and removal of the file when indexing fails. The
  service rebuilds the pipeline over the whole library after an upload.
- **Dataset ingestion:** CSV types and original headers, read-only preview, refusal of
  non-schema/injection table names, duplicate refusal, explicit replace, stored uploads never
  overwritten, Excel sheets, and a corrupt workbook.
- **Routing:**
  - a document question reaches the RAG callable only;
  - a share question reaches `StructuredQA`, with a verified denominator and a "Share of total"
    derived table;
  - a hybrid question uses both engines;
  - write/DROP requests stay read-only.
- **Session isolation:** two stores, separate conversations, no empty duplicates, interrupted and
  failed questions, clear and delete.
- **Error handling:**
  - Ollama unreachable or model missing;
  - Qdrant locked, with SQL still working and the document question explaining;
  - no documents;
  - unexpected backend exceptions are contained, and their text is not leaked;
  - credentials are redacted from the configuration display.
- **Presentation:** raw vs derived labels and units, notes, truncation warning, original vs rejected
  results, citations, claims, calculation operands, hybrid sections.

**Single-PDF equivalence check** (manual script): with the one sample PDF, the library adapter gives the
same 781 chunks, the same scope metadata (`ScopeIndex._meta` and segment names) and the same parsed
pages as `build_pipeline`. The default document pipeline therefore behaves as before.

**Benchmarks.** No backend code changed. The structured and hybrid benchmarks were **not** re-run in
Phase 5. Historical results under `data/evaluation/` were not touched (`git status` shows no change
there).

## 4. Manual validation (local, 2026-11-16)

Streamlit 1.67, Ollama `llama3.2:3b` on CPU, deterministic generation. The app ran against **copies** of
`data/documents`, `data/qdrant` and `data/structured/insightrag.db` in a temporary folder (set through
the `INSIGHTRAG_*` variables), so the project's own index and database were not modified. The browser was
the Claude desktop app's built-in browser.

| Check | Result |
|---|---|
| App starts | Yes. The first start (parsing the 291-page PDF, chart extraction, loading the embedding and reranker models) took about 2 min behind a spinner. All status lights were green |
| Document question ("total sales in 2025") | Document (RAG), 12.9 s: "$61,352 million". Supported, citations p.95/101/203, and the evidence text opens in the citation expander |
| Structured share question (S16 wording) | SQL, 3.9 s: the first query was rejected, regenerated once, the denominator 104,042 verified, a derived share table plus a raw table, and a pie chart rendered |
| Structured aggregation ("revenue per customer region") | SQL, 2.9 s: 3 rows, bar chart, SQL expander |
| Hybrid question (H01) | Hybrid, 10.4 s: split sub-questions shown; data 28,793 from `segment_kpis` beside the document's "$28,793 million" (Supported, p.95/172), in two columns |
| PDF upload: invalid | A `.pdf` holding text was refused: "does not look like a PDF file (missing %PDF header)" |
| PDF upload: valid | `contoso_brief.pdf` (2 pages) → "2 pages, 2 chunks indexed". The library showed 2 documents, both indexed after a restart |
| Question about the new PDF | Document (RAG), 9.5 s: "Contoso operated 4 electric arc furnaces …", Supported, first citation `contoso_brief.pdf` p.2 |
| CSV upload | `regional_headcount.csv` loaded (3 rows). A second upload with the same name was refused with the replace hint |
| Excel upload | `plant_shipments.xlsx` → table `plant_shipments_shipments` (sheet `shipments`, 4 rows). The dataset browser showed the types (TEXT/INTEGER/INTEGER), the original headers and the preview |
| Period change on uploaded Excel data | "Gent shipments 2024→2025": 410 → 455, +45 (+10.98%), scope filter `plant = 'Gent'` checked, derived table labelled |
| Write requests | "Delete all orders …" and "Drop the customers table …" were refused as read-only, and no SQL ran |
| Conversations | A new conversation opens on the welcome screen. Switching back showed the first conversation's 12 messages without the other conversation's question |
| Degraded mode (second instance, `OLLAMA_HOST` set to a closed port, same Qdrant folder) | Status: Ollama unreachable, model unknown, vector index "in use by another process", datasets OK. A data question gave a clear "Cannot reach Ollama at …" error, and the app kept running |
| Server log | Only route/length/timing lines from the app. After disabling Streamlit's file watcher there were no error lines (before that, the watcher logged harmless `torchvision` import tracebacks from `transformers`) |

Issues found and fixed during manual validation:
1. A read-only refusal was shown twice.
2. The status message pointed to a "Retry" button that is actually called "Re-check".
3. Documents were shown as "not indexed" when the index was only unavailable.
4. A rejected upload could not be deleted on Windows while openpyxl still held the file (found by the
   tests).

## 5. Known limitations

- **Startup time:** the first start parses every PDF and loads two models (about 2 min for the sample
  report on CPU). It is cached for the life of the server process, not across restarts.
- **No conversational memory:** each question is answered on its own, as in the backend. Separate
  conversations are separate histories, not separate model contexts.
- **Embedded Qdrant lock:** the app and the CLI tools cannot use the same index folder at the same
  time. The app reports this and keeps data questions available.
- **One process, serialised questions:** a lock protects the shared index and models, so several
  users or tabs queue behind each other.
- **Document removal:** there is no delete button. Removing a PDF means deleting the file and using
  Rebuild document index, which re-embeds every PDF.
- **Datasets:** there is no delete button either; a dataset can be replaced, not removed, from the
  UI.
- **Several documents:** retrieval, reranking and validation work across the library. However:
  - the existing generator and abstention wording still say "the provided document";
  - evidence selection can add citations from other documents. In the Contoso test, two
    annual-report pages were cited after the correct `contoso_brief.pdf` page; validation was still
    Supported;
  - segment names for query rewriting are merged across documents;
  - the system was tuned and benchmarked on one annual report only.
- **Routing:** unchanged. Data questions must name the table, its columns or its values. The example
  questions are fixed and assume the sample report and datasets.
- **Uploads in manual testing:** files were put into Streamlit's own upload inputs with browser
  JavaScript (`DataTransfer`) because the test browser has no file picker. Streamlit's normal upload
  path received them, but the operating system's file-chooser dialog was not exercised.
- **Interrupted queries:** a stopped run is recorded as interrupted on the next run. The backend call
  that was running (an Ollama request) is not cancelled; it finishes in the background. This was
  covered by a unit test only, not manually.
- **OCR:** scanned PDFs are refused (no text layer).
- **Presentation details:** raw tables show numbers as stored (no thousands separators). Units are
  inferred from column-name suffixes only, so an aliased column such as `sales` shows no unit.

## 6. Running the application

```bash
pip install -r requirements.txt
```

```bash
ollama pull llama3.2:3b
```

```bash
streamlit run app/streamlit_app.py
```

Open http://localhost:8501. Close `python -m app.main` / `python -m app.assistant` first (they hold the
Qdrant folder). Configuration, uploads, example questions and troubleshooting are described in
`README.md`.
