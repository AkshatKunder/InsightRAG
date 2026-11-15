# Phase 3 findings (qualitative, hand-written)

**What improved.** The fairest comparison is with the Phase 2 pipeline re-run under the same deterministic
settings. Against it, Phase 3 raises answer accuracy from 0.727 to 0.818 (+2 questions correct),
correct-or-partial answers from 0.818 to 0.909, and fully matched numeric facts from 0.682 to 0.818.
Against the original sampled Phase 2 results, accuracy rises from 0.682 to 0.818. The gains trace to
specific changes:

- **Q020 (multi-hop), abstained → correct.** Splitting the question into sub-queries put the
  segment-headcount page (130) next to the sales table (page 95). The 5-chunk context now holds a page
  for every required fact in every question (context fact recall 1.000; Phase 2: 0.977). Per-sub-query
  citation selection cites both pages.
- **Q009 (production), incorrect → correct.** The page 3 KPI chart now reaches the generator as labelled
  values ("Iron ore production (million tonnes)*: 2025 = 48.8"), so the model no longer substitutes the
  run-of-mine aggregate (112.3 Mt).
- **Q022 (multi-hop comparison), incorrect → partially correct** (vs the deterministic baseline). The
  arithmetic check caught "55.6 ... lower than 54.0", and the single revision fixed the direction. The
  stated difference (0.6 instead of 1.6 Mt) is still wrong.
- **Stability.** With temperature 0 and seed 42 applied (`options_applied: true`), every question got the
  same label in both runs (22/22). Phase 2's sampled runs agreed on 18/22.

**What regressed.** Q018 (iron ore change) went from *partially correct* to *abstained*. The model keeps
choosing the Mining-segment figures (27.9 → 35.3 Mt) over the consolidated ones (42.4 → 48.8 Mt). The
scope check flags them as a different reporting scope, the revision does not find the group figures,
and the system abstains. This is the intended trade (a refusal instead of a wrong figure), but it lowers
the label for this question. Page 95, which states the consolidated change, is still not in Q018's context.

**Citation validation.** With the answers held fixed (Phase 2 answers, enriched citations), the Phase 3
validator cuts false positives from 11 to 6 of 18 wrong or partial answers, and false negatives from 6 to
5 of 45 correct ones. Its recorded false-positive rate on Phase 3 answers (4 of 6) looks high only
because few wrong answers remain. Those four are Q001, which omits "mining" (not a numerical error), and
Q022's wrong difference "by 0.6 million tonnes" (a derived value with no second operand in the sentence).

**Abstention.** All unanswerable questions are still refused, with no citations and no invented numbers.
The false-abstention rate on answerable questions is unchanged (0.045): Q020's refusal was replaced by
Q018's.

**Limitations.**

- Scope detection relies on report layout: headed "Performance for the year ended" tables, "in the
  <Name> segment" phrases, and the report's own list of segment names. Segment tables without such a
  heading are treated as consolidated. Only one chart (page 3) could be decoded reliably.
- The revision step costs one extra LLM call, and only when a contradiction is found. It fixed a
  direction error but not the follow-on arithmetic.
- "Unsupported value" (a figure not found in the citations) marks a claim UNSUPPORTED but does not
  trigger revision or abstention: an early run showed it also hits correct answers written as formulas
  (Q021 "$61,352 / 125,554").
- The results come from 22 answerable questions and 2 deterministic runs; one question moves accuracy by
  0.045. Retrieval components, embeddings and RRF weights were not changed, and the Qdrant index did not
  need rebuilding (chunk text is unchanged; scope metadata is attached after retrieval).

**Next steps.** Add query rewriting toward consolidated figures for "change" questions (Q018 needs
page 95). Recompute stated differences from the cited values (Q022). Extend the non-numeric checks to
cover completeness of names and lists (Q001, Q002).
