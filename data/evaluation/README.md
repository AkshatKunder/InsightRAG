# InsightRAG evaluation benchmark

`benchmark_v1.json` is a ground-truth question set for evaluating InsightRAG on
the ArcelorMittal Annual Report 2025. It is the input for Phase 2 evaluation:
retrieval metrics (did the right pages come back?), answer accuracy (is the
answer correct, with the right unit, scale and period?) and abstention (does the
system decline when the report has no answer?).

This folder contains data only. No evaluation metrics are implemented yet.

| File | Purpose |
|---|---|
| `benchmark_v1.json` | The benchmark (25 questions) |
| `benchmark_verification_summary.md` | One-line verification record per question |
| `../../scripts/validate_benchmark.py` | Structural and evidence checks |

## Source document

- File: `data/documents/annual_report.pdf` (local only; not committed)
- Document: ArcelorMittal Annual Report 2025
- 294 physical pages, 291 with extractable text

**Page numbers** are physical PDF page positions starting at 1. They are the
same numbers `app.rag.loader.load_pdf` stores in each chunk's `page` field, so
retrieved chunks can be compared with `expected_pages` directly. On the pages
used here (e.g. 3, 95, 172) they match the printed page footer, but that has
not been checked for the whole PDF, so always use the physical position.

## Categories

| Category | Count | What it tests |
|---|---|---|
| `factual` | 5 | Explicit non-numeric facts (business, segments, CEO, incorporation, history) |
| `numerical` | 5 | Single figures in different units: USD millions, FTE counts, million tonnes |
| `financial` | 5 | Income statement, cash flow and balance sheet figures (operating income, cost of sales, net income, capex, net debt) |
| `comparative` | 4 | Change between 2024 and 2025: both values, direction, absolute and percent change |
| `multi_hop` | 3 | Facts from two different pages combined by lookup or arithmetic |
| `unanswerable` | 3 | Questions the report genuinely cannot answer; the correct behavior is to abstain |
| **Total** | **25** | 22 answerable, 3 unanswerable |

## Schema

Top level:

| Field | Meaning |
|---|---|
| `benchmark_name`, `version`, `created` | Identification |
| `source_document`, `source_description` | The PDF the benchmark is based on |
| `document_pages` | Physical page count of the PDF (294); page numbers must fall in 1..this |
| `page_numbering` | How page numbers are defined |
| `total_questions`, `categories`, `category_counts` | Must agree with `questions` |
| `questions` | List of question objects |

Every question:

| Field | Required | Meaning |
|---|---|---|
| `id` | yes | Unique, `Q###` |
| `category` | yes | One of the six categories |
| `question` | yes | Question text as a user would ask it |
| `answerable` | yes | `false` only for the `unanswerable` category |
| `expected_answer` | yes | Concise reference answer; exactly `"ABSTAIN"` for unanswerable questions |
| `expected_pages` | yes | Pages holding the supporting evidence; empty for unanswerable questions |
| `notes` | no | Traps and ambiguities a grader should know about |

Answerable questions also have:

| Field | Meaning |
|---|---|
| `supporting_evidence` | `[{page, excerpt}]`: verbatim text from the PDF (whitespace collapsed). Every page must be in `expected_pages`. |
| `expected_facts` | `[{metric, value, unit?, period?, segment?, pages}]`: atomic facts the answer must contain. `pages` lists **every** page where that fact can be found, so a retriever is credited for any of them (e.g. total sales is on both page 95 and page 172). |
| `expected_comparison` | Comparative only: `{direction, absolute_change, unit, percent_change}` |
| `calculation` | Derived answers only: `{expression, result, unit, tolerance_percent}` |
| `reasoning` | Multi-hop only (optional): how the hops connect |

Unanswerable questions also have:

| Field | Meaning |
|---|---|
| `expected_behavior` | What a correct response looks like |
| `unanswerable_reason` | Why the report cannot answer it |
| `distractor_pages` / `distractor_evidence` | Pages with near-miss content (related numbers a system might wrongly use) |

Values follow the units in the report: `"USD million"` figures are written
as plain numbers (61352 means $61,352 million). Periods are years (`"2025"`)
for flows and dates (`"2025-12-31"`) for point-in-time values such as headcount
and net debt.

## Answerable vs. unanswerable

An **answerable** question has a definite answer stated in the report (or
computable from stated figures, for the multi-hop questions). The system should
answer it and cite at least one page from each fact's `pages`.

An **unanswerable** question asks for information the report does not
disclose: a future figure (2030 sales), a granularity the report does not have
(sales on one day), or a private detail (one site manager's salary). These are
not just hard questions. Each was checked by searching the full text for the
relevant terms, and the reason is recorded in `unanswerable_reason`. The correct
behavior is to abstain (InsightRAG reports this as `ABSTAINED`) without citing
evidence or estimating a number.

## How the answers were verified

1. The full text of every page was extracted with the project's own loader
   (`app.rag.loader.load_pdf`, pypdf), so page numbers match what the pipeline indexes.
2. Each candidate fact was found by searching that text, then read in context
   on its page to confirm the table header, column order, unit, scale and period.
3. Where a figure appears in more than one place (narrative and table, or
   management report and financial statements), all locations were checked to
   agree and are listed in the fact's `pages`.
4. Derived values (Q016–Q019 changes, Q021 ratio, Q022 difference) were
   recalculated from the source figures, and the report's own change columns
   were checked where present.
5. For unanswerable questions, the full text was searched for the relevant
   terms (e.g. `2030`, quarterly or daily sales, `salary`, `Dunkirk`) to
   confirm no answer exists.
6. `scripts/validate_benchmark.py` re-checks the structure and confirms that
   every excerpt appears verbatim on its cited page.

## Validating

```bash
python scripts/validate_benchmark.py
```

This runs the structural checks and, when `data/documents/annual_report.pdf`
exists, the excerpt check. Use `--no-pdf` to skip the PDF, or pass another file:
`python scripts/validate_benchmark.py path/to/benchmark.json --pdf path/to/report.pdf`.
The validator does not judge whether an answer is semantically correct; that
was done by hand as described above.

## Benchmark versions

| File | Status |
|---|---|
| `benchmark_v1.json` | Original verified benchmark. Immutable; kept for comparability. |
| `benchmark_v2.json` | v1 plus page 3 as alternative evidence for Q017/Q018 (see [benchmark_v2_changes.md](benchmark_v2_changes.md)). Primary benchmark for Phase 2 comparisons. |

v2 adds an optional `alternative_evidence` field (`[{page, excerpt, note}]`):
verbatim excerpts on a page listed in a fact's `pages` but not in
`expected_pages`. The validator checks these like `supporting_evidence`.
Validate either file with `python scripts/validate_benchmark.py data/evaluation/benchmark_v2.json`.

## Retrieval evaluation

`python scripts/evaluate_retrieval.py` scores the retrieval stage against this
benchmark with Recall@K, Precision@K and MRR. It writes results to `results/`.
See [retrieval_evaluation.md](retrieval_evaluation.md) for the metric
definitions, the relevance rules and their limitations.

`python scripts/compare_retrieval.py` compares BM25, vector and hybrid
retrieval, each before and after cross-encoder reranking, on benchmark v2
(primary) and v1 (baseline). It writes
[retrieval_strategy_comparison.md](retrieval_strategy_comparison.md),
[reranker_evaluation.md](reranker_evaluation.md), SVG charts in `charts/`,
and JSON in `results/`. The raw rankings are saved to
`results/retrieval_runs.json`, so they can be re-scored without models:
`python scripts/compare_retrieval.py --runs data/evaluation/results/retrieval_runs.json`.

## Answer, citation, numerical and abstention evaluation

`python scripts/evaluate_answers.py` runs the full production pipeline
(hybrid → rerank → llama3.2:3b → citation selection → claim validation; needs
Ollama) on every question three times. It grades the outputs
deterministically against benchmark v2, with no LLM judge. Reports:
[answer_quality_report.md](answer_quality_report.md),
[numerical_evaluation.md](numerical_evaluation.md) and
[abstention_report.md](abstention_report.md). Raw outputs are saved to
`results/answer_runs.json`; re-grade them without models with
`--runs data/evaluation/results/answer_runs.json`. The grading rules are in
`app/evaluation/answer_checks.py`: value, unit, period and entity must all
agree, so a matching number alone is not enough.

## Phase 2 report

`python scripts/build_phase2_report.py` builds
[phase2_evaluation_report.md](phase2_evaluation_report.md) from the JSON
results. Its qualitative sections come from the hand-written
`phase2_findings.md`.

## Adding questions

1. Find the fact in the PDF and note the physical page number (use `load_pdf`
   or the excerpt check, not the printed page footer).
2. Copy a short excerpt verbatim, including the row label and numbers for table
   values. Record the table's unit and period in `notes` if the excerpt does not show them.
3. List every page where the same fact appears in the fact's `pages`.
4. Use the next free `Q###` id, then update `total_questions` and `category_counts`.
5. For a new unanswerable question, search the full text first and record what
   you searched for in `unanswerable_reason`.
6. Run `python scripts/validate_benchmark.py` and fix any failures.
7. Add the question to `benchmark_verification_summary.md`.

For a substantive change (rewording questions, changing answers), create
`benchmark_v2.json` instead of editing v1, so earlier evaluation results stay comparable.

## Known limitations

- **One document.** All questions come from the 2025 annual report, so results
  say nothing about generalization to other documents.
- **Small sample.** 25 questions with 3–5 per category, so one result moves a
  category score by 20–33%. Treat per-category numbers as indicative.
- **Phrasing.** Questions are written in clear, report-like language. Real
  users may phrase things more loosely or use synonyms.
- **Page-level ground truth.** Evidence is recorded per page, not per chunk.
  A page can be split across several chunks, so chunk-level metrics need to map chunks to pages first.
- **Extraction artifacts.** Excerpts reflect pypdf's text extraction, not the
  visual layout. The page 3 KPI graphic in particular extracts as a run of
  numbers without column alignment (see Q008).
- **Free-text answers.** `expected_answer` is a reference answer, not an
  exact-match string. Automatic grading should compare `expected_facts`
  (value, unit, period) rather than wording.
- **Approximate figures.** The report gives several figures in rounded form as
  well (e.g. $61.4 billion for $61,352 million). Graders should accept
  rounding that is consistent with the exact value.
- **Coverage.** Only three unanswerable questions, and no ambiguous or
  adversarial questions (e.g. false premises) yet.
