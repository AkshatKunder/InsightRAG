# Phase 4.1 findings (qualitative, hand-written)

**Result.** On structured benchmark v1, all 28 questions are now correct in both deterministic runs,
up from 25 in Phase 4. Routing stays at 100% for the 28 structured questions and the 25 document
questions of benchmark v2. Valid SQL and execution success are 100%, and safety, unanswerable
abstention and hybrid attribution are unchanged at 100%. No question regressed against Phase 4.

**What changed, by Phase 4 failure.**

- **S07 (customer names):** the model still returns customer IDs. The engine now adds the names
  through the verified key with a separate read-only query, keeping the top-3 order. With the
  schema-driven "return the name column" note, the model wrote `customer_name` without the join in all
  three attempts. The engine then fell back to the plain Phase 4 prompt (valid ID-based SQL), and the
  names were resolved deterministically.
- **S12 (Q1 2025):** the prompt now carries the explicit range 2025-01-01 to 2025-03-31, and any query
  without both bounds is sent back for repair. The generated SQL uses
  `BETWEEN '2025-01-01' AND '2025-03-31'`.
- **S15 (Europe sales change):** asked for one row per year, the model now returns both 2024 and 2025,
  and the change is computed in Python (−1,159; −3.87%). **Caveat:** the generated SQL ignores the
  Europe filter, returning all segments, and adds model-made columns named `sales_change` that actually
  hold 2024 values. The benchmark counts S15 as correct because the deterministic Europe change is
  present, but the answer is noisy and its extra columns could mislead. The engine does not yet
  restrict the answer to the requested category or drop model-computed "change" columns.
- **H01 (citation):** the citation fallback now adds page 95. Before this fix it found no chunk,
  because the existing validator reads "The Europe sales figure…" as the entity "The Europe". The
  fallback's own support check and its re-validation drop a leading article before a capitalised word;
  value, unit, metric, year and scope are still checked. Document page citation for hybrid questions:
  0.667 → 1.000.

**Regressions found during the work (and how they were resolved).** A first version put "select name
columns" in the global prompt. With deterministic generation, that one extra line changed the 3B model's
SQL on unrelated questions:
- S13 lost its `ORDER BY`/`LIMIT`.
- S18 gained extra grouping columns.
- S19 picked the wrong customer.
- U02 answered with customer names instead of no answer.

Accuracy fell to 0.739. The instruction was moved into a schema-driven note that appears only for
"which/who/list/top" questions about an entity with a verified key and name column. All four
questions recovered.

**Limitations.**

- **Fallback year check:** the citation fallback relies on the existing validator, which checks that a
  passage mentions the claim's year but not which table column the figure is in. A 2024 table value
  claimed for 2025 could still be cited (not observed in the benchmark; noted from a unit test).
- **Multi-period answers:** the engine computes the requested change correctly but does not filter the
  rows to the requested category, nor remove the model's own derived columns (S15).
- **Phase 3.1 untouched:** the leading-article false negative in the existing validator was worked
  around in the hybrid fallback, not fixed in the Phase 3.1 validator.
- **Coverage of the new rules:** quarter parsing covers "first quarter of 2025", "Q1 2025",
  "2025 Q4" and ordinals; fiscal years that don't follow the calendar and half-years are not handled.
  Name resolution needs a unique, complete name, title or label column and a question that names the
  entity.
- **Small benchmark:** 28 questions on two small datasets (one synthetic). 100% here is not a general
  accuracy figure, and the measured prompt sensitivity shows how fragile a 3B model's SQL can be.
