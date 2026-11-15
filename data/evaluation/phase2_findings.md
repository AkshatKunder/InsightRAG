# Phase 2 findings (qualitative, hand-written)

This file holds the human interpretation embedded in `phase2_evaluation_report.md`
by `scripts/build_phase2_report.py`. Numbers quoted here come from the
generated result files (benchmark v2, 2026-11-15 runs); if the evaluations are
re-run, re-check these statements against the regenerated tables.

## Executive summary

- **Retrieval is the strongest part of the system.** Hybrid RRF beats both
  single retrievers on MRR and recall at every K. The production stage
  (hybrid + rerank) places at least one relevant page in the top 5 for all 22
  answerable questions, and the five context chunks contain a page with every
  required fact for 21 of 22 questions (the exception is Q020).
- **Most wrong answers are generation errors, not retrieval misses.** The
  LLM often takes a *segment* or *alternative-measure* figure from a correct
  neighbouring table instead of the consolidated figure: Mining-segment iron
  ore (35.3 Mt), North America crude steel (7,755 kt), run-of-mine aggregate
  iron ore (112.3 Mt).
- **Financial questions are answered reliably** (accuracy 1.00 in every run);
  production, shipment, comparative-change and multi-hop questions are not.
- **Abstention is conservative.** All 9 unanswerable trials abstained with no
  citations and no invented numbers. The cost is one answerable multi-hop
  question (Q020) that is refused in every run.
- **Citations are relevant but not precise.** Nearly every answered question
  cites at least one page that holds the fact (fact recall 0.98), but only
  about 40% of cited pages are relevant: of the 3 citations, typically about
  2 are neighbouring pages that do not hold the fact.
- **The built-in claim validator is not a reliable correctness signal.** It
  marks most wrong answers `SUPPORTED`, because the wrong number does appear
  in a cited passage, and marks some correct answers `UNSUPPORTED`.

## Retrieval strategies

- **Hybrid is best overall** and the most robust: it has no question without
  a relevant page in the top 10, and only two (Q012, Q019) without one in the
  top 3. BM25 and vector fail on different questions, which is exactly what
  fusion exploits.
- **BM25 fails on conceptual wording.** Q001 ("primary business") has its
  first relevant page at rank 19 for BM25, but rank 1 for vector. Q009 and
  Q022 are also late (ranks 9 and 6), because the many tables repeating
  "production" and "tonnes" outscore the consolidated table.
- **Vector search fails on exact financial line items.** Cost of sales (Q012,
  rank 15), R&D (Q010, rank 10), shipments (Q017, rank 9), operating income
  (Q011, rank 8) and capex (Q014, rank 7). Table rows with numbers embed
  poorly with a small general-purpose embedding model; BM25 finds these by
  their exact labels.
- **Precision is low for every strategy by design:** most questions have 1–3
  relevant pages, so P@10 cannot exceed about 0.1–0.3. Compare strategies on
  precision, not against 1.0.
- **Benchmark v2 changed only Q017/Q018,** which raised MRR slightly for the
  strategies that retrieve page 3. The Task 2 baseline is reproduced exactly
  on v1, which confirms the comparison harness matches the earlier evaluation.

## Reranker

- **The production blend (0.7 first-stage + 0.3 cross-encoder) is roughly
  neutral on hybrid:** 4 questions improved, 4 degraded, MRR −0.004, Hit@1
  −0.045, Hit@3 +0.045. It never degrades a BM25 or vector ranking and
  improves 6 and 8 questions respectively, so it helps weaker first stages more.
- **The cross-encoder alone hurts hybrid substantially** (MRR −0.147, 10
  questions degraded). MS MARCO MiniLM is trained on web passages and ranks
  narrative text above dense financial tables, which is where most answers
  are. The 0.7 weight on the first-stage score is what keeps the production
  reranker from doing harm; increasing the cross-encoder weight would be a regression.
- **Where the first relevant page moved down with the production rerank:**
  Q006 and Q008 (rank 1 → 2), Q014 (2 → 3), Q018 (1 → 2). Secondary relevant
  pages also moved down in Q002, Q007 and Q017 without changing the first hit. These are table-heavy pages
  displaced by narrative pages. Gains: Q011 (3 → 1) and Q019 (4 → 1), both
  questions whose answer is stated in narrative text.

## Category analysis

- **Financial (5/5 correct in every run):** single consolidated figures with
  distinctive labels; retrieval and generation both work, often with
  rounded billions ($3.6 billion, $4.3 billion), which the grader accepts as
  rounding-consistent.
- **Factual:** names and dates are right (CEO, incorporation), but Q001
  never says "mining", so it misses the "integrated steel and mining" answer.
  Q002 lists the segments with "Others" as if it were a segment or without
  the count, and Q005 omits the merger year in some runs.
- **Numerical:** employees, crude steel, R&D and sales are right; Q009 (iron
  ore production) is wrong in every run, because the run-of-mine aggregate
  112.3 Mt from page 70 is chosen while page 95 is not in the context.
- **Comparative:** the direction is right in every run, but the change
  values are often missing or wrong. Q018 uses the Mining-segment figures
  (35.3 vs 27.9 Mt, +26.5%) instead of the consolidated 48.8 vs 42.4 Mt.
- **Multi-hop (weakest):** Q020 abstains in every run (page 130 with
  segment headcount is not retrieved into the top 5). Q021's division is
  right in two of three runs. Q022 uses North America segment crude steel
  and in one run states the wrong direction.

## Failure cases

| Question | What went wrong | Stage | Evidence |
|---|---|---|---|
| Q009 iron ore production | Answers 112.3 Mt, the aggregate run-of-mine figure for 2025 (page 70), not ArcelorMittal's 48.8 Mt | Retrieval + generation | Context pages 69, 26, 3, 59, 70: page 95 absent; page 3 has 48.8 only as an unlabelled number run |
| Q018 iron ore change | Mining-segment 35.3 vs 27.9 Mt (+26.5%) instead of the consolidated 48.8 vs 42.4 Mt (+15.1%) | Generation (segment vs total) | Pages 97 and 59 state the segment figures explicitly; page 95 is not in the context |
| Q022 production vs shipments | North America crude steel 7,755 kt (page 96) compared with a 10,283 kt figure from another table; one run concludes "lower" | Generation | Context has the segment tables (91, 96) but no page 60 |
| Q020 largest segment workforce | Abstains in every run although sales (page 95) are in the context | Retrieval (page 130 missing from top 5) | Context pages 95, 204, 94, 132, 97 |
| Q008 crude steel (run 3) | States 54.0 Mt (the shipments figure) as crude steel production | Generation reading the flattened page 3 chart | Page 3 text is "55.6 48.8 54.0 61.4 …" with no labels in between |
| Q001 primary business | Describes steel only and omits mining in every run | Generation | Page 3 is retrieved (rank 2) and states "integrated steel and mining" |
| Validator false positives | 11 of 18 incorrect or partial question-runs are still `SUPPORTED` | Citation validation | The validator checks that a number appears in a cited passage, not that it is the right metric or scope |

The common pattern is **scope confusion**: the report contains many tables
with the same metric names at segment level, and the 3B model picks a
plausible but wrong table. Retrieval ranks those segment tables highly
because they share every query term.

## Limitations

- **Small benchmark:** 22 answerable and 3 unanswerable questions from one
  document. One question moves a category score by 20–33%. Treat category
  results as indicative.
- **Generation is non-deterministic:** the production generator sets no
  temperature or seed. Results are the mean of 3 runs with min–max ranges;
  18 of 22 answerable questions get the same label in every run (Q005, Q008,
  Q021 and Q022 vary).
  Three runs bound the variance only roughly.
- **Page-level relevance:** a retrieved page counts as relevant even when the
  chunk holds a different table from that page; retrieval metrics can overstate quality.
- **Deterministic grading is rule-based:** it checks values, units, years,
  segments, direction words and required terms. It does not judge the
  reasoning in free text (e.g. whether Q019's stated cause is complete) and
  can miss paraphrases such as a segment list written without the count.
  Every graded answer was also read by hand for this report. The one grader
  false positive found (list numbering "6." counted as "six segments") was
  fixed and covered by a test. No LLM judge was used.
- **Rounding policy:** rounding-consistent values with at least two
  significant figures are accepted ($3.2 billion for $3,152 million, $61
  billion for $61,352 million). A stricter policy would lower financial accuracy.
- **Unanswerable questions are few and clearly out of scope** (future year,
  a single day, an individual salary). The 100% abstention rate says little
  about harder near-miss questions.
- **Citation precision depends on the selector returning 3 citations.**
  Questions with one relevant page cap precision at about 0.33.
- **System validator rates** are reported for transparency but are part of
  the system under test, not ground truth.
- **Page 3 chart:** v2 credits page 3 for Q017/Q018 retrieval, but its
  extracted text is a flattened number run, so having it in the context does
  not mean the generator can use it (see Q008, run 3).

## Recommendations

1. **Keep hybrid retrieval with the current 0.7/0.3 rerank blend.** Do not
   raise the cross-encoder weight or switch to cross-encoder-only ordering
   (measured −0.147 MRR on hybrid). Consider a reranker suited to tables, or
   try skipping reranking for table-heavy queries.
2. **Address scope confusion at generation time:** tell the model to prefer
   consolidated ("Total", group-level) figures unless a segment is named,
   and include each chunk's table header or section title in the excerpt
   text so "(Mining segment)" or "North America" is visible next to the numbers.
3. **Strengthen citation validation:** check that the cited passage states
   the same *metric and scope* as the claim (segment names, "aggregate ROM",
   "capacity"), not only the same number. This would catch most of the 11
   `SUPPORTED`-but-wrong cases.
4. **Improve multi-hop retrieval:** retrieve per sub-question (e.g. "segment
   employees" and "segment sales") or widen the context from 5 to 8 chunks
   for questions that combine two facts. Q020 fails only because page 130
   is outside the top 5.
5. **Make generation deterministic for evaluation** (temperature 0 and a
   fixed seed via an option on `Generator`) so that later comparisons
   measure system changes rather than sampling noise.
6. **Parse charts and tables with layout information:** the page 3 KPI
   chart and similar graphics lose their labels in pypdf text; the geometry
   approach in `scripts/verify_chart_mapping.py` shows that these can be recovered.
7. **Extend the benchmark** with near-miss unanswerable questions (e.g. a
   metric reported only for a segment) and more multi-hop questions before
   tuning against it, to avoid overfitting 25 questions.

## Conclusions

InsightRAG's retrieval works for this report. Hybrid RRF is the best first
stage, and the production pipeline gets every needed fact page into the
generator's context for 21 of 22 answerable questions. End-to-end answer
accuracy is about two thirds (0.68 mean of three runs, strict grading); a
further sixth of answers are partially correct. Abstention on clearly
unanswerable questions is perfect, and no fabricated numbers were observed:
every wrong figure was copied from a real but inappropriate table. The
weakest points are the 3B model's choice between segment-level and
consolidated figures, multi-hop questions, and a citation validator that
confirms numbers without confirming their meaning. The evaluation framework
is now reproducible end to end: saved raw outputs can be re-scored without
models. That gives Phase 3 a baseline for measuring improvements to
generation prompting, scope-aware validation and multi-hop retrieval.
