# Phase 4.3 findings (qualitative, hand-written)

**Grading change.** Up to Phase 4.2, S16 ("What share of 2025 employees … does each segment have?")
was graded on the share figures only. The model computed the share in SQL
(`1.0 * SUM(employees_fte) / SUM(employees_fte) OVER ()`), and the rows were fractions, not employee
counts. The figures came out right only because Python normalised the fractions again. From Phase 4.3,
a share question counts as correct only when all of the following hold:
- the shares are right;
- the presented rows are the raw values of the gold result;
- no model-computed share or ratio column is presented.

This is the same principle as the Phase 4.2 period-change rule. Re-graded on it, Phase 4.2 is 27/28
(S16 fails), not 28/28.

**S16, what happens now (both runs identical).**
1. The first query is unchanged (the prompt is unchanged): `segment, share` with a division and a
   window aggregate. The `share` column is recognised as model-computed and dropped.
2. No raw metric is left, so the result is rejected.
3. The single retry names the raw column to return (`employees_fte`) and forbids division, window
   functions and shares in SQL. The model returns
   `SELECT segment, SUM(employees_fte) AS employees_fte FROM segment_kpis WHERE year = 2025 GROUP BY segment`.
4. The checks pass:
   - one row per segment, no missing or negative values;
   - the year scope is right;
   - an independent read-only query (`SELECT SUM("employees_fte") FROM "segment_kpis" WHERE "year" IN (2025)`)
     gives 104,042, the same as the sum of the returned rows.
5. The answer lists each segment as raw value / 104,042 = share, for example Europe
   43,878 / 104,042 = 42.17%. The rows equal the gold result exactly, and the pie chart is kept.

**Regressions.** None. All 28 questions are correct in both runs, under both the strict and the
figure-only grading. Routing (53 questions), valid SQL, execution success, charts, unanswerable
abstention, safety and hybrid metrics stay at 100%. Execution accuracy (rows = gold rows) rises from
0.957 to 1.000. The other per-questions (S06, S08, S11, S18: "per region/product/industry", "average
revenue per order") are not treated as share or ratio questions and are unchanged. The full test
suite passes: 484 tests, 13 new.

**Fix made during the work.** The new select-list parser exposed a bug in the Phase 4.2 helper
`scope._depth0_scan`: punctuation separators directly after a word ("segment,") were skipped because
the scanner required a word boundary for every split. Only the new comma splitting used this path
(the Phase 4.2 OR-branch splitting is word-based), so Phase 4.2 behaviour is unchanged. All Phase 4.2
tests pass.

**Limitations.**
- **Model dependence:** S16's success depends on the 3B model's single retry. With other wording or
  data, the outcome may be a refusal ("I cannot calculate these shares reliably …"), which is the
  intended safe outcome.
- **Denominator verification:** only for a SUM/COUNT (or plain column) from a single table, with the
  requested years and filters detected from the question. With joins, expressions or tables without
  one year/date column, the denominator is the total of the returned rows. The answer says so, but it
  is not independently checked.
- **Filter detection:** this inherits the Phase 4.2 limits: stored values must appear verbatim in the
  question, and synonyms are not detected. A named value of the share column ("share of Europe") is
  treated as the numerator, not a filter. That is usually right but is a heuristic.
- **Grouping:** only one category column plus an optional group (a period with several values, or a
  second category named after "each/by/per") is supported. Other shapes are refused as ambiguous.
  Shares of a subset of the total ("top 3 customers' share of all revenue") are refused because of the
  LIMIT, not answered with a separate total.
- **Negative parts:** shares with negative parts (for example operating income with losses) are
  refused rather than computed. Ratios allow negative values but refuse zero denominators.
- **Model-computed columns:** these are recognised by the SQL expression (division, `OVER`) or the
  alias. A model-computed value without a division or window that has a neutral alias would be treated
  as raw.
- **Ratios:** ratio questions are matched only in explicit forms ("ratio of A to B", "A as a
  percentage of B", "A-to-B ratio"). Each term must map to exactly one raw numeric column by name.
- **Small benchmark:** S16 is the only share question, and there is no ratio question. The other new
  behaviour is covered by unit tests with fake LLM outputs, not by measured runs.
