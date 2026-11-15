# Retrieval evaluation

This measures how well InsightRAG's retrieval stage finds the benchmark's
verified evidence pages. It scores the ranked chunks only. Answer generation,
citation validation and Ollama are not involved, so these numbers say nothing
about answer quality or abstention.

## Running it

From the project root, after the Qdrant index has been built (`python -m app.ingest`):

```bash
python scripts/evaluate_retrieval.py
```

Options:

| Option | Default | Meaning |
|---|---|---|
| `--k` | `1 3 5 10` | K values to report (positive integers) |
| `--benchmark` | `data/evaluation/benchmark_v1.json` | Ground truth (read only, never modified) |
| `--pdf` | `data/documents/annual_report.pdf` | Source PDF for the in-memory BM25 index |
| `--output-dir` | `data/evaluation/results` | Where the result files are written |

What gets evaluated: the assistant's retrieval stage from `app/main.py`.
Hybrid retrieval (BM25 + vector, weighted RRF, one chunk per page) selects the
top 20 chunks from 50 candidates per retriever. The cross-encoder reranker then
reorders them and keeps the top max(K). The assistant itself keeps 5, but
reranking sorts every candidate before cutting the list, so the first 5 results
here are exactly what the LLM receives. K ≤ 5 therefore describes the
production context, and K = 10 shows what lies just outside it. Comparing
BM25, vector, hybrid and reranked retrieval is a later task.

Outputs (overwritten on each run):

- `results/retrieval_evaluation.json`: the summary, plus one record per answerable
  question. Each record has the expected and relevant pages, each expected
  fact and whether it was found, the ranked results (page, source, every score
  field, whether the result was relevant, a 200-character text preview), the
  first relevant rank, the reciprocal rank and every metric at every K.
- `results/retrieval_evaluation_summary.json`: overall and per-category
  averages, MRR, question counts, excluded IDs, the retrieval configuration
  and a UTC timestamp.

The code lives in `app/evaluation/`. `retrieval_metrics.py` holds the pure
metric functions. `retrieval_evaluator.py` matches results to ground truth and
aggregates the scores. Both can be used without models:
`evaluate_results(benchmark, {question_id: ranked_results}, k_values)`.

## Evidence units and relevance

The benchmark records evidence per **page**, so the evidence unit is a
distinct `(source, page)` pair:

- A retrieved chunk is **relevant** when its `source` file name equals the
  benchmark's `source_document` (`annual_report.pdf`; directory prefixes are
  ignored) **and** its `page` is one of the question's *relevant pages*.
- **Relevant pages** = `expected_pages` ∪ every page in `expected_facts[].pages`.
  The benchmark lists every page where a fact can be found, so a retriever
  that finds total sales on page 172 instead of page 95 is credited.
- Several chunks from the same page are **one** evidence unit and can be
  counted only once. Later chunks from that page still take up a rank, so they
  lower precision instead of raising it.
- Results are scored **in the order the retriever returns them**. Scores are
  stored but never used to reorder results or to decide relevance (there are no score thresholds).
- A result with a missing or invalid `source` or `page` (missing, not an
  integer, < 1) keeps its rank, is never relevant, and is counted in
  `num_invalid_metadata`.

## Metric definitions

For one question with ranked results r₁, r₂, …:

| Metric | Definition |
|---|---|
| **Recall@K** (headline) | Share of the question's `expected_facts` with at least one of their pages among the top K results. Fact-level: a fact counts as found if any page that states it is retrieved. |
| **Page recall@K** | Distinct `expected_pages` in the top K ÷ number of `expected_pages`. Stricter: it asks for every listed page. |
| **Precision@K** | Distinct relevant pages in the top K ÷ min(K, number of results returned). |
| **Reciprocal rank** | 1 ÷ rank of the first relevant result; 0 if none is relevant. |
| **MRR** | Mean reciprocal rank over the evaluated questions, computed over the full returned list (depth max(K)). |

Overall and per-category values are unweighted means over questions.

Why fact-level recall is the headline: page recall penalises a retriever for
not returning *every* page where a fact is repeated. For example, Q006 lists
sales on pages 95 and 172; one of them is enough to answer the question. Fact
recall asks whether each required piece of evidence was found at least once,
which is what the answer depends on. Page recall is kept for completeness.

## Unanswerable questions

The 3 unanswerable questions have no evidence pages, so recall, precision and
reciprocal rank are undefined for them. They are not retrieved or scored, and
they are excluded from every average. The summary reports
`num_answerable` (22), `num_unanswerable` (3), `num_evaluated` (22) and
`excluded_unanswerable_ids`. Abstention is evaluated separately.

## Limitations

- **Page-level relevance can overestimate.** A chunk counts as relevant if it
  comes from a relevant page, even when it covers a different table or section
  of that page (for example, a chunk from page 95 that holds segment data rather
  than the total-sales row). Finer matching, such as checking the chunk text
  against `supporting_evidence` excerpts or fact values, is future work.
- **Relevance is only as complete as the benchmark's page lists.** A page that
  states the fact but is not listed counts as irrelevant. Example: the
  verification notes say the page 3 KPI graphic also shows 2025/2024 shipments
  (54.0 / 54.3 Mt) and 2024 iron ore (42.4 Mt). Q017 and Q018 list only page 95,
  so retrieving page 3 for them is not credited. The ground truth was left
  unchanged; if it is extended, do it in a new benchmark version.
- **Precision is low by construction.** Most questions have 1–3 relevant
  pages, so Precision@10 cannot exceed about 0.1–0.3 even for a perfect
  retriever. Compare precision between strategies, not against 1.0.
- **Hybrid retrieval returns one chunk per page.** This pipeline therefore
  never produces duplicate pages. The duplicate handling matters for
  strategies without page diversity (plain BM25 or vector search).
- **Small sample.** 22 questions, 3–5 per category; one question moves a
  category average by 20–33%.
- **Deterministic but environment-dependent.** Results depend on the local
  index, model versions and hardware numerics. Record the configuration with
  any results you compare.
