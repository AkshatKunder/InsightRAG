# Phase 4.2 findings (qualitative, hand-written)

**Grading change.** Phase 4.1 counted S15 as correct because the right Europe change (−1,159;
−3.87%) appeared among the analytics of a result that had all five segments and model-made
`sales_change` columns. From Phase 4.2 on, a period-change question counts as correct only when all of
the following hold:
- the calculated change is right;
- the presented rows equal the gold (scoped) result;
- no model-computed change column is presented.

Earlier phases are re-graded with this rule from their saved records. On it, Phase 4.1 is 27/28, not
28/28. Phase 4 already failed S15 on the figure. The old grading is still shown for comparison.

**S15, what happens now (both runs identical).**
1. The first generated query is the same as in Phase 4.1 (the prompt is unchanged): all segments plus
   three `*_change` columns.
2. The scope validator finds that the question names `Europe`, a value of `segment_kpis.segment`, and
   that the result has rows for Brazil, Mining, North America and Sustainable Solutions. The result is
   rejected.
3. The single retry states the filter (`segment = 'Europe'` in the WHERE clause) and the reason.
4. The model's three retry attempts all apply the filter, but all use `strftime('%Y', year)` on the
   integer `year` column, which the existing lint rejects. Attempts 2 and 3 also have an AND/OR
   precedence error (`segment = 'Europe' AND year = 2024 OR year = 2025`).
5. A new, narrow deterministic rewrite (`strftime('%Y', <integer year column>)` becomes the column;
   only the `'%Y'` format, only for INTEGER columns whose values are all years, and only after every
   attempt was rejected) turns retry attempt 1 into
   `SELECT year AS period, SUM(sales_usd_m) AS sales FROM segment_kpis WHERE segment = 'Europe' AND year BETWEEN 2024 AND 2025 GROUP BY year`.
   That query is validated again and passes the scope check.
6. The answer lists 2024: 29,952 and 2025: 28,793, then the change calculated in Python: −1,159
   (−3.87%). Rows match the gold result exactly.

**Intermediate result (kept in `results/phase4_2/intermediate_run_scope_only/`).** Before the rewrite was
added, the same retry was rejected by the lint. The engine then correctly refused S15 ("I cannot
calculate this change …") rather than presenting the unfiltered rows: overall 27/28, with S15 counted
as incorrect. That run also showed that a check which only looks for `segment = 'Europe'` somewhere in
the WHERE clause would have accepted the precedence-error query. The check now requires the filter in
every top-level OR branch.

**Regressions.** None. All 28 questions are correct in both runs. Routing (53 questions), valid SQL,
execution success, charts, unanswerable abstention, safety and hybrid metrics are unchanged at 100%.
Execution accuracy rises from 0.913 to 0.957 because S15's rows now equal the gold rows. The full test
suite passes: 471 tests, 12 new.

**Limitations.**
- **Filter detection:** a filter is found only when the question contains a stored value verbatim
  (case-insensitive, whole words). Synonyms, abbreviations ("EU"), misspellings and numeric or date
  filters other than the requested years are not detected. Only tables the query reads are searched;
  if the model queries the wrong table, the filter is not found there.
- **Scope checks:** these apply only to change questions naming two or more years. Single-period
  questions with a category ("Europe sales in 2025") are not scope-checked. The SQL predicate check is
  textual (`=`, `IN`, `LIKE`, WHERE/HAVING/ON, top-level OR branches). It does not prove the semantics
  of arbitrary SQL, for example filters applied through a CTE that is then not used. When the filter
  column is returned, the check is on the rows themselves and is reliable.
- **Rewrite:** it is a generic fix for one model error (a date function on an integer year). The SQL
  that reached the right answer for S15 depends on that rewrite and the 3B model's retry. Other
  wording or data may still end in refusal, which is the intended outcome for an unresolvable scope.
- **Presentation:** model-computed columns are recognised by name (`change`, `diff`, `delta`,
  `growth`, `yoy`, `variance`, `ratio`, `increase`, `decrease`). A misleading column with a neutral
  alias would not be detected. Unasked metric columns are dropped only when at least one metric
  matches the question by name.
- **S16 (shares), out of scope here:** the model computes a share in SQL (window function) despite the
  prompt rule. The share percentages are right because they are normalised again in Python, but the
  rows are fractions rather than the underlying FTE counts. S16 is graded on the figures, as in
  Phase 4.1.
- **Small benchmark:** only one period-change question (S15). The new behaviour is covered mainly by
  unit tests with fake LLMs, not by a broad measured benchmark.
