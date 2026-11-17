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
| `app/structured/periods.py` | Quarter ranges and requested years (Phase 4.1) |
| `app/structured/scope.py` | Requested filters, result-scope validation, model-derived columns (Phase 4.2) |
| `app/structured/shares.py` | Share/ratio intent, model-computed fractions, validated shares and ratios (Phase 4.3) |
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

### Phase 4.1 additions

- **Quarters** (`app/structured/periods.py`): "first quarter of 2025", "Q1 2025", "2025 Q4" and "3rd
  quarter 2024" are converted to exact date ranges with the calendar. Q1 is January 1 to March 31,
  leap years included, and Q4 ends on December 31. The range is given to the model, and a query that
  doesn't use both dates is sent back for repair.
- **Multi-period changes:** for "change between 2024 and 2025" the prompt asks for one row per year.
  If the result misses a requested year, the engine retries once, naming the missing year. If a year is
  still missing, it answers that the change cannot be calculated (`abstained`). It never computes a
  change from one period. Only real years count as found periods: 20245 is not 2024.
- **Name resolution:** when the question asks for an entity (for example customers) and a result column
  contains only values of that entity's verified primary key, the unique name column is added next to
  it. The names come from a separate validated read-only query, and the row order (ranking, top N) is
  unchanged. The prompt also tells the model to select name columns.
- **Hybrid citation fallback** (in the assistant; the RAG pipeline is unchanged): a claim in the
  document answer that the selected citations don't support gets the first retrieved context chunk
  that supports it on its own. The answer is then validated again. No chunk is added if none supports
  the figure.

### Phase 4.2 additions: result scope for multi-period comparisons

These apply only to change questions naming two or more years ("How did Europe sales change between
2024 and 2025?"). The prompt itself is unchanged; the checks run on the executed query
(`app/structured/scope.py`).

- **Requested filters come from the data.** A categorical value named in the question is detected when
  it is a value of a TEXT column of a table the query reads, for example `segment = 'Europe'`. The
  values are looked up with validated read-only queries. A value contained in a longer matched value is
  ignored ("America" inside "North America").
- **Scope validation.** The result must be limited to the requested scope:
  - If the filter column is returned, every row must carry a requested value.
  - If it is not returned, the SQL must filter on it in `WHERE`, `HAVING` or a join `ON`. A
    `CASE WHEN` in the select list is not a filter. In a WHERE clause, the filter must hold in every
    top-level OR branch: `segment = 'Europe' AND year = 2024 OR year = 2025` is rejected.
  - Period values outside the requested years are reported.
- **One retry, then refusal.** A query that is out of scope (or misses a requested year) is regenerated
  once, with notes that state the missing filter and the reason for rejection. If the retried result is
  still out of scope, the rows are withheld and kept only in `rejected_result`. The answer then says
  the change cannot be calculated and contains no figures (`abstained`). The hybrid assistant does not
  use an abstained structured result as evidence. Finding the right figure inside a wider result is not
  accepted.
- **Model-computed columns are not authoritative.** For change questions, result columns named like a
  calculation (`sales_change`, `growth_pct`, `yoy_delta`) that are not real table columns are dropped.
  Metric columns the question doesn't ask about are dropped too, for example `employees_fte` when the
  question asks about sales. The full query result stays in `raw_result`, and the dropped columns are
  listed in `presentation`. The change is calculated in `analytics.py` from the remaining raw values.
- **Year strftime rewrite** (`nl2sql.rewrite_year_strftime`, for any question). Only when every
  generation attempt has been rejected because a date function was applied to a numeric column,
  `strftime('%Y', col)` is replaced by `col`. This applies only to the `'%Y'` format and only to INTEGER
  columns whose values are all years. The attempts are tried in order, and the rewritten query is
  validated and linted again before it is used.
- **Answer layout.** The answer gives the value for each period, as returned and labelled with the
  scope, then the change calculated in Python. If the rows don't give one value per requested period, the
  answer says the change cannot be calculated.

### Phase 4.3 additions: deterministic shares and ratios

These apply to questions about a share, a percentage of the total, a contribution, a distribution or
breakdown, and to ratios ("ratio of operating income to sales", "income as a percentage of sales").
"Revenue per order" and similar "per" questions are averages and are not affected. The prompt is
unchanged; `app/structured/shares.py` checks the executed result.

- **Model-computed fractions are not used.** A result column counts as model-computed when its SQL
  expression divides or uses a window aggregate (`OVER`), or when its alias names a share, percentage,
  ratio or fraction and it is not a real table column. Such columns are dropped and kept only in
  `raw_result`. Shares and ratios are calculated in Python from the raw metric values.
- **What is identified:**
  - the category column;
  - an optional group, such as a year with several values or a second category named after
    "each/by/per";
  - the single metric the question asks about.
  A value named in the question that belongs to the category column ("What share did Europe have?")
  selects the numerator, not a filter, so the denominator still needs every category.
- **Denominator checks:**
  - one row per category within each group, with no missing values;
  - no negative values (shares of a total are not meaningful then);
  - a total that is not zero;
  - no `LIMIT` or row cap;
  - an additive metric (`AVG`/`MIN`/`MAX` are refused);
  - the requested scope: named filters and years.
  For a `SUM`/`COUNT` (or plain column) from one table, an independent read-only query computes the
  total for the requested years and filters. The returned values must add up to it. With joins or
  other expressions, the denominator is the total of the returned rows, and the answer says so.
- **Ratios:** each row needs one raw column for each term and a non-zero denominator.
- **One retry, then refusal.** If the raw values are missing or a check fails, the question is
  regenerated once, naming the raw columns to return and the reason for rejection. If it still fails,
  the rows are withheld and the answer says the shares (or ratio) cannot be calculated reliably.
- **Answer layout:** raw value / denominator = percentage for each category, labelled with the metric,
  scope and group, plus whether the total was checked against an independent query. Validated
  ungrouped shares can be shown as a pie chart even without `SUM` in the SQL.

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

The evaluation scripts and per-phase results were removed to keep the repository small (they remain in
Git history). The benchmarks are kept as reference data: `data/evaluation/structured_benchmark_v1.json`
(28 questions with gold SQL) and `benchmark_v2.json` (document questions). The final structured-data
results are summarised in `data/evaluation/phase4_3_evaluation_report.md`: 28/28 correct in two
deterministic runs (temperature 0, seed 42).
