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

## Evidence unit and the meaning of K

The benchmark records evidence per **page**, so the evidence unit is a
distinct `(source, page)` pair. All metrics use this one unit.

1. Each retrieved chunk is mapped to `(source file name, page)`. Directory
   prefixes are ignored, so `data/documents/annual_report.pdf` matches the
   benchmark's `source_document` `annual_report.pdf`.
2. The ranked chunk list is collapsed into a **ranking of distinct units**:
   each `(source, page)` stays at the position of its first chunk, and later
   chunks from the same page are dropped. The order is otherwise exactly the
   retriever's. Scores are stored but never used to reorder results or decide
   relevance (there are no score thresholds).
3. A chunk with a missing or invalid `source` or `page` (missing, not an
   integer, < 1) cannot be matched to a page. It stays in the ranking as its
   own non-relevant unit and is counted in `num_invalid_metadata`.

**K is a number of distinct units, not chunks.** "Top K" means the first K
distinct pages of that ranking. Example: chunks from pages `[9, 9, 9, 95]`
become the ranking `[9, 95]`, so page 95 is at position 2 and is inside the top 2.
Duplicate chunks therefore neither inflate a score nor push other pages out of the top K.

In the detailed results, each chunk has `chunk_rank` (its raw position),
`unit_rank` (its position in the distinct ranking, or `null` for a duplicate)
and `duplicate`. Each question has `ranked_pages` (the distinct ranking),
`num_distinct_units` and `num_duplicate_chunks`.

A unit is **relevant** when its source matches `source_document` and its page
is one of the question's **relevant pages** = `expected_pages` ∪ every page in
`expected_facts[].pages`. The benchmark lists every page where a fact can be
found, so a retriever that finds total sales on page 172 instead of page 95 is credited.

## Metric definitions

For one answerable question, let *U* be the distinct-unit ranking, *U*ₖ its
first K units, *R* the set of relevant pages, and *F* the list of expected
facts (each with its set of pages):

| Metric | Definition |
|---|---|
| **Recall@K** (headline) | \|{f ∈ F : f's pages ∩ *U*ₖ ≠ ∅}\| ÷ \|F\|: the share of expected facts with at least one of their pages in the top K. |
| **Page recall@K** | \|`expected_pages` ∩ *U*ₖ\| ÷ \|`expected_pages`\|. Stricter: every listed page must be retrieved. |
| **Precision@K** | \|*R* ∩ *U*ₖ\| ÷ min(K, \|*U*\|): the share of the top-K distinct units that are relevant. If fewer than K distinct units were returned, the denominator is the number returned. |
| **Reciprocal rank** | 1 ÷ position in *U* of the first relevant unit; 0 if none. |
| **MRR** | Mean reciprocal rank over the evaluated questions, using the whole returned ranking. |

Recall and precision count the same thing (relevant units in *U*ₖ); they
differ only in the denominator. Overall and per-category values are unweighted
means over questions.

Why fact-level recall is the headline: page recall penalises a retriever for
not returning *every* page where a fact is repeated. For example, Q006 lists
sales on pages 95 and 172; one of them is enough to answer the question. Fact
recall asks whether each required piece of evidence was found at least once,
which is what the answer depends on. Page recall is kept for completeness.

**Retrieval depth.** The CLI asks the pipeline for max(K) chunks. Hybrid
retrieval already returns one chunk per page, so this gives max(K) distinct
units. A strategy that returns several chunks per page (for example plain BM25
or vector search, in a later task) yields fewer distinct units than chunks.
It should then be asked for more chunks so that the top K units are filled.
`questions_with_fewer_than_max_k_units` in the summary lists any question where that happened.

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
- **Relevance is only as complete as the benchmark's page lists.** A page
  that states a fact but is not listed for it counts as irrelevant. Known case:
  page 3 (the "Company overview" KPI bar chart) shows 2025 and 2024 steel
  shipments (54.0 / 54.3 Mt) and iron ore production (48.8 / 42.4 Mt), but
  Q017 and Q018 list only page 95, so page 3 is not credited for them (the
  same facts do list page 3 in Q009 and Q022). Benchmark v1 is left unchanged;
  `benchmark_v2.json` adds page 3 for these facts (see `benchmark_v2_changes.md`).
- **Precision is low when the retriever returns K pages.** Most questions have
  1–3 relevant pages, so a retriever that returns 10 distinct pages cannot
  score above about 0.1–0.3 Precision@10. Compare precision between
  strategies, not against 1.0.
- **Small sample.** 22 questions, 3–5 per category; one question moves a
  category average by 20–33%.
- **Deterministic but environment-dependent.** Results depend on the local
  index, model versions and hardware numerics. Record the configuration with
  any results you compare.
