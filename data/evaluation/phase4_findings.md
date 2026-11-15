# Phase 4 findings (qualitative, hand-written)

**What works.**

- **Routing:** 100% correct on 53 questions. That covers 28 structured questions and all 25
  annual-report questions, which stay on the document pipeline, so the earlier behaviour is preserved.
- **SQL execution:** every generated query passed validation and executed (no SQL errors).
- **Safety:** all three write requests (delete, update, drop) were refused before any SQL was
  generated, and the database fingerprint was unchanged.
- **Unanswerable questions:** both produced no result instead of a guessed one.
- **Hybrid questions:** 3 of 3 were fully correct after splitting each question into a data part and a
  document part.
- **Stability:** both runs (temperature 0, seed 42) gave identical results.

**What fails (generated SQL, measured).**

| Question | Problem | Effect |
|---|---|---|
| S07 | "Which 3 customers" returns customer IDs, not names | The top-3 order is right, but the names are not shown |
| S12 | "First quarter of 2025" is read as March 2025 | Wrong total |
| S15 | The Europe sales change returns only 2025, although the prompt asks for one row per period | Flagged as incomplete; no change is stated (no fabricated figure). This is the failed half of the 0.5 derived-figure accuracy; the share question S16 passes |

These are limits of the 3B model: vague time expressions, identifiers versus labels, and multi-period
retrieval. The safeguards keep them visible rather than silent:
- The SQL is always shown.
- Results with a missing period are flagged as incomplete.
- Shares and changes are computed in Python.

**Hybrid citation gap.** H01's document answer is correct ($28,793 million), but the citation selector
did not cite page 95 (page cited for 2 of 3 hybrid questions).

**Design decisions.**

- **Safety layers:** generated SQL passes text checks, then an authorizer, then a read-only connection
  with row and time limits. No unvalidated SQL is executed, and metadata and system tables cannot be
  read.
- **Arithmetic:** done in Python, never by the LLM.
- **Relationships:** reported only when verified against the data. The model may only use verified joins.
- **Router:** deterministic and conservative. Words shared with the report do not trigger SQL, so
  document questions keep the Phase 3.1 pipeline.

**Limitations.**

- **Small benchmark:** 28 structured questions over two small datasets, one of them synthetic. The
  results show behaviour, not general NL-to-SQL accuracy.
- **Router phrasing:** the router expects structured questions to name the data ("orders table",
  "segment KPI data") or to use terms that only occur in the data. Unqualified questions default to the
  document.
- **Hybrid splitting:** it handles "Compare A in the table with B in the report" and "A in the table,
  and what … according to the report"; other phrasings send the whole question to both sides.
- **Excel:** formulas are read as their last saved values (`data_only=True`). Merged cells and
  multi-row headers are not interpreted.
- **SQLite only:** there is no server database and no concurrent writers.
