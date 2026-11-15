# Structured data and natural language to SQL (Phase 4)

InsightRAG can answer questions over CSV and Excel data alongside the annual report. Structured data
lives in a local SQLite database. The local Ollama model (`llama3.2:3b`) writes SQL from the discovered
schema, and every query is validated and run read-only. Questions are routed to the document pipeline,
the database, or both.

```
question ─► router ─┬─► document  ─► Phase 3.1 RAG pipeline (unchanged)
                    ├─► sql       ─► NL-to-SQL ─► validation ─► read-only execution ─► analytics / chart / answer
                    └─► hybrid    ─► both, independently ─► combined answer with source attribution
```

| Module | Responsibility |
|---|---|
| `app/structured/ingestion.py` | CSV/XLSX reading, validation, type inference, safe identifiers, SQLite storage, metadata |
| `app/structured/schema.py` | Schema discovery: columns, types, ranges, sample values, verified keys and joins; LLM schema text |
| `app/structured/sql_safety.py` | Text checks, SQLite authorizer, read-only connection, row and time limits |
| `app/structured/nl2sql.py` | Prompt, SQL extraction, validation, lint and repair loop |
| `app/structured/analytics.py` | Deterministic percentage changes, period comparisons, shares, summary statistics |
| `app/structured/charts.py` | Chart-ready data (bar, line, pie) or no chart |
| `app/structured/router.py` | Document / SQL / hybrid classification |
| `app/structured/engine.py` | `StructuredQA` (SQL answers) and `Assistant` (routing and combination) |
| `app/assistant.py` | Command-line interface |

`app/main.py` (document-only RAG) is unchanged.

## Usage

```bash
python scripts/create_sample_datasets.py
python -m app.assistant --ingest data/structured/samples/segment_kpis.csv data/structured/samples/steel_orders_sample.xlsx
python -m app.assistant --schema
python -m app.assistant --ask "What is the total order revenue per customer region in the orders data?"
python -m app.assistant --ask "Compare Europe's 2025 sales in the segment KPI table with the Europe sales figure in the annual report."
python -m app.assistant              # interactive; add --no-rag to skip loading the document pipeline
```

From Python:

```python
from app.rag.generator import complete
from app.structured.engine import StructuredQA

qa = StructuredQA("data/structured/insightrag.db", complete=lambda p: complete(p, {"temperature": 0, "seed": 42})[0])
result = qa.ask("Show the monthly order revenue in 2025 from the orders table.")
result["sql"], result["columns"], result["rows"], result["answer"], result["chart"]
```

Each result contains the SQL, the columns and rows (capped at 200, with a `truncated` flag), the
answer text, the `analytics` (computed changes and shares, with their source rows), a `chart` spec, the
source tables and files, and an `error` field. If the query fails, `error` explains why and the answer
contains no figures.

## Ingestion

- **Formats:** `.csv` (delimiter detected automatically; UTF-8 with or without a BOM, or Windows-1252)
  and `.xlsx`. Every non-empty sheet becomes its own table; empty sheets are skipped.
- **Validation:** files that are empty, binary, larger than 50 MB or over 1,000,000 rows are rejected,
  as are files with no header or data rows, or rows wider than the header. Nothing is written when a
  file fails, because each file loads in a single transaction.
- **Identifiers:** original column names are kept in metadata. Database identifiers are lower-case
  `[a-z0-9_]`, unique, not starting with a digit and never an SQL keyword ("Sales (USD m)" becomes
  `sales_usd_m`, "Margin %" becomes `margin_pct`). Blank and duplicate headers are renamed, with a warning.
- **Types:** INTEGER, REAL, DATE (stored as ISO `YYYY-MM-DD` text; Excel dates and common
  day/month formats are converted) or TEXT. Thousands separators, currency symbols and accounting
  negatives such as `(5)` are parsed. `NA`, `N/A`, `null`, `-` and blank cells become NULL, and missing
  values are counted per column.
- **Metadata tables:** `_datasets` (name, file, SHA-256, type, load time), `_tables` (sheet, rows,
  columns, warnings) and `_columns` (original and database names, type, missing and distinct counts).
  Re-ingesting a dataset replaces its tables.

## Schema discovery

- **Primary key:** a column named as an identifier (`id`, `*_id`, `*_code`, `*_key`) whose values are
  all present and unique. A unique value column is not treated as a key.
- **Relationships:** reported only after a query confirms that every non-null child value exists in
  the parent key, and the column names agree. No join is assumed from names alone. The LLM is told to
  join only through these verified joins.
- **Text for the LLM:** tables with row counts and source files; typed columns with original names,
  ranges, up to 8 sample values for categorical columns, and notes for integer year columns.

## SQL generation

The prompt contains the schema text, eleven rules, and three generic examples over an unrelated
`sales` table. No dataset-specific queries are included. The model writes one SQLite SELECT, or
`NO_SQL` when the tables cannot answer.

The engine then takes these steps:

1. Extract the first statement from the reply; anything after it is discarded.
2. Validate it (see the next section).
3. Run a schema-aware lint that rejects date functions applied to numeric columns.
4. If it is invalid, send the error back to the model, up to two times.

The model is told not to compute differences, percentage changes or shares in SQL. It returns the
values per period or category, and `analytics.py` does the arithmetic in Python. A change question
whose result is missing one of the named periods is reported as incomplete instead of being answered.

## Safety controls

1. **Write requests are refused before any SQL is generated**, for example "delete/drop/update/insert …
   rows/table/data".
2. **Text checks:** exactly one statement, no SQL comments, and it must start with SELECT or WITH. No
   DROP, DELETE, UPDATE, ALTER, INSERT, CREATE, REPLACE INTO, ATTACH, DETACH, PRAGMA, VACUUM, REINDEX,
   TRUNCATE, transaction keywords or file functions may appear outside string literals.
3. **SQLite authorizer**, active while the statement is prepared and run: only SELECT, column reads on
   loaded tables and the query's own CTEs, and ordinary functions are allowed. The metadata tables,
   `sqlite_master` and every write, DDL, PRAGMA or ATTACH are denied.
4. **Read-only connection:** `mode=ro` URI and `PRAGMA query_only = ON`.
5. **Limits:** at most 200 rows are returned (configurable, with a `truncated` flag), and a progress
   handler stops queries after 5 seconds.
6. **No credentials:** SQLite is local and file-based, with no secrets. The database file is git-ignored.

## Routing

`router.classify` is deterministic.

- **Structured signal:** a question gets one when it refers to tabular data ("table", "dataset",
  "spreadsheet", "the orders data") or uses a distinctive schema term, meaning a table, column or
  categorical value that never appears in the annual report.
- **Document signal:** a question gets one when it refers to the report or document.

| Signals | Route |
|---|---|
| Both | hybrid |
| Structured only | sql |
| Neither | document (the unchanged Phase 3.1 pipeline) |

Words shared with the report, such as "sales" or "Europe", don't trigger SQL on their own.

For hybrid questions, the document pipeline and the SQL engine run independently. The combined answer
labels each part ("From the structured data (table …)" and "From the document (pages …)"), and the
sources list contains both tables (with the SQL) and document pages. If neither side has evidence, the
assistant abstains.

## Charts

| Chart | Used when |
|---|---|
| line | One period column with at least 3 distinct, ordered points |
| bar | One categorical column with 2–30 categories |
| pie | The question asks for a share or breakdown, the SQL uses SUM or COUNT, there are 2–8 non-negative values, and the total is positive |

There is no chart for single values, numeric columns in different units, repeated category labels, or
too many categories. The spec includes the reason for the choice.

## Sample datasets

- `data/structured/samples/segment_kpis.csv` holds **real** figures from the annual report: segment
  sales and operating income (page 95) and FTE employees (page 130) for 2024 and 2025.
- `data/structured/samples/steel_orders_sample.xlsx` is **synthetic** test data: fictional customers and
  120 orders, generated with a fixed seed. It is not ArcelorMittal data.

## Evaluation

```bash
python scripts/evaluate_structured.py            # 2 deterministic runs (temperature 0, seed 42)
```

The benchmark is `data/evaluation/structured_benchmark_v1.json`: 28 questions with gold SQL, and the 25
questions of `benchmark_v2.json` are also used for routing. The report is
`data/evaluation/phase4_evaluation_report.md`, and the results are in `data/evaluation/results/phase4/`.
