# Phase 3.1 findings (qualitative, hand-written)

**Result.** Phase 3.1 answers 21 of 22 answerable questions correctly in both deterministic runs (0.955),
up from 0.818 in Phase 3 and 0.682 in Phase 2. Every answerable question is now at least partially correct
(1.000), and no question regressed against Phase 2 or Phase 3. Abstention on the unanswerable questions
is unchanged (3/3 refused, no citations, no invented numbers), and the one false abstention from Phase 3
(Q018) is gone.

**Q018, consolidated figure retrieval (fixed).** Cause: the question's own query ranks the Mining-segment
table (page 97) and narrative (page 59) above the consolidated performance table (page 95), which was at
rank 9 after reranking. Fix: change questions get one extra query shaped like the report's table header
("iron ore production 2025 2024 change % change"), with group-level chunks ordered first. Page 95 now
enters the context and is cited. The change is computed from consolidated evidence (42.4 → 48.8 Mt,
+6.4 Mt, +15.1%), and the Mining-segment sentence the model wrote first is dropped as a scope
contradiction. Questions that name a segment get no rewrite and keep segment evidence (tested).

**Q022, numerical calculation (fixed).** The difference is now computed in code from labelled evidence
values: 55.6 − 54.0 = 1.6 Mt, "higher". The operands come from the recovered page 3 chart, with metric,
unit and year checked. The model's own arithmetic ("lower ... 0.6 million tonnes") is replaced. If an
operand cannot be found with a matching metric, unit, year and scope, nothing is computed.

**Q001, completeness (fixed).** A definition-style rewrite ("ArcelorMittal company overview") retrieves the
"integrated steel and mining companies" sentence. The completeness check flags "steel production" as
missing "mining", and the single revision corrects it. Optional modifiers ("integrated") are not required,
so concise answers such as "a steel and mining company" pass.

**Q002, completeness (detected, not fixed).** The list check finds the document's enumeration ("six
reportable segments: …"), and the revision splits "Sustainable Solutions and Mining" correctly. The model
still adds "Others" as a seventh segment, and the check reports this as incomplete
(`validation["completeness"]`: extra = Others). Rewriting the answer from the evidence list in code would
fix it, but that goes beyond detection, so it was not done in this phase.

**Side effects measured.**

- **Q017:** the calculation re-states the same correct result (0.3 Mt, 0.6%).
- **Q016 and Q019:** no operand row matches the metric (their consolidated row is labelled "Total"), so they
  are unchanged and still correct.
- **Context:** page recall@5 rises from 0.682 to 0.727. Citation precision rises slightly (0.407 → 0.429)
  because operand pages are cited.

**Limitations.**

- Rewrites and calculations rely on two question shapes ("how did <metric> change between Y1 and Y2", "was
  A higher or lower than B") and on table layout ("<Y2> <Y1>" header rows, labels with units in
  parentheses). Other phrasings fall back to Phase 3 behaviour.
- Operands are found only in recovered chart values and labelled table rows. Narrative sentences and rows
  labelled "Total" are not used (Q016, Q019).
- The completeness check covers only enumerations with a stated count and copular definitions. It detects
  but does not repair; the result depends on the single revision.
- The results come from 22 answerable questions and 2 deterministic runs with the same seed. A different
  seed or model version may change individual answers, and 0.955 should not be read as a general accuracy
  figure.
