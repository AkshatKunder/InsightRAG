# Benchmark v1: verification summary

Source: ArcelorMittal Annual Report 2025 (`data/documents/annual_report.pdf`, 294 physical pages).
Page numbers are physical PDF positions, as used by `app.rag.loader.load_pdf`.

Every answer below was located in the PDF text and read in context (table
header, unit, period) on each listed page. All 48 evidence excerpts in
`benchmark_v1.json` were then confirmed verbatim on their cited pages by
`scripts/validate_benchmark.py`. Verified 2026-11-15.

**Totals:** 25 questions (22 answerable, 3 unanswerable).
By category: factual 5, numerical 5, financial 5, comparative 4, multi-hop 3, unanswerable 3.

## Answerable questions

| ID | Category | Question (short) | Expected answer | Evidence pages |
|---|---|---|---|---|
| Q001 | factual | Primary business | Leading integrated steel and mining company | 3 |
| Q002 | factual | Reportable segments | Six: North America, Brazil, Europe, India and JVs, Sustainable Solutions, Mining | 94, 59, 201 |
| Q003 | factual | Chief Executive Officer | Aditya Mittal | 106, 107 |
| Q004 | factual | Country and date of incorporation | Luxembourg, June 8, 2001 | 3 |
| Q005 | factual | Predecessor companies and merger year | Mittal Steel Company N.V. and Arcelor, 2007 | 3 |
| Q006 | numerical | Total sales 2025 | $61,352 million (≈ $61.4 billion) | 95, 172 |
| Q007 | numerical | Employees end-2025 | ≈ 125,554 FTE | 130, 3 |
| Q008 | numerical | Crude steel production 2025 | 55.6 million tonnes | 60, 3 |
| Q009 | numerical | Iron ore production 2025 | 48.8 million tonnes | 95, 3 |
| Q010 | numerical | R&D expense 2025 | $335 million | 29, 212 |
| Q011 | financial | Operating income 2025 | $3,628 million | 95, 172 |
| Q012 | financial | Cost of sales 2025 | $56,976 million | 172 |
| Q013 | financial | Net income attributable to equity holders 2025 | $3,152 million (≈ $3.2 billion) | 172, 98, 281 |
| Q014 | financial | Capital expenditures 2025 | $4,337 million (≈ $4.3 billion) | 67, 176, 203 |
| Q015 | financial | Net debt at Dec 31, 2025 | $7.9 billion (13.4 debt − 5.5 cash) | 99 |
| Q016 | comparative | Sales change 2024→2025 | −$1,089 million (−1.7%): 62,441 → 61,352 | 95 |
| Q017 | comparative | Steel shipments change | −0.3 Mt (−0.6%): 54.3 → 54.0 Mt | 95 |
| Q018 | comparative | Iron ore production change | +6.4 Mt (+15.1%): 42.4 → 48.8 Mt | 95 |
| Q019 | comparative | Operating income change and reason | +$318 million (+9.6%): 3,310 → 3,628, mainly the $1.9 bn AMNS Calvert gain | 95, 172 |
| Q020 | multi_hop | Largest segment workforce and its sales | Europe: 43,878 FTE (p130); sales $28,793 million (p95) | 130, 95 |
| Q021 | multi_hop | Sales per employee 2025 | ≈ $488,650 per FTE (61,352 M / 125,554) | 95, 130 |
| Q022 | multi_hop | Crude steel production vs. shipments 2025 | Production higher by 1.6 Mt (55.6 vs 54.0) | 60, 95 |

## Unanswerable questions

| ID | Question | Expected | Why the report cannot answer | Near-miss pages |
|---|---|---|---|---|
| Q023 | Total sales in 2030 | ABSTAIN | No sales forecast; "2030" appears only for carbon targets and the buyback window | 5, 67, 95 |
| Q024 | Sales on March 15, 2025 | ABSTAIN | Only annual sales; no daily, weekly or quarterly breakdown | 95, 172 |
| Q025 | Base salary of the Dunkirk plant general manager | ABSTAIN | Pay disclosed only for the Executive Chairman, the CEO, executive aggregates and a holding-company average | 8, 118, 119, 121 |

## Verification notes

- **Units.** Pages 95, 172, 176 and 203 are in $ millions; page 99 (net debt)
  is in $ billions. Narrative sentences give rounded billions ($61.4 billion,
  $3.6 billion, $3.2 billion, $4.3 billion), consistent with the table values.
- **Consolidated vs. segment.** Q006/Q011 use the "Total" row. Segment rows
  on page 95 are before inter-segment eliminations and do not add up to the total.
- **Production vs. shipments.** Q008/Q009 are production, Q017 is shipments,
  Q022 compares them explicitly.
- **Page 60 columns** are BOF, EAF, Total, Achievable Capacity: 55.6 is
  production, 74.6 is capacity.
- **Page 3 KPI graphic** lists 2025 values (55.6, 48.8, 54.0, 61.4) and then
  2024 values (57.9, 42.4, 54.3, 62.4). The 2025 values were cross-checked
  against pages 60 and 95. The 2024 crude steel figure (57.9) appears only in
  this graphic, so no question depends on it.
- **CEO vs. Executive Chairman.** Lakshmi N. Mittal is Executive Chairman;
  Aditya Mittal is CEO (page 106 board table, page 107 biography).
- **Net income.** $3,152 million is attributable to equity holders of the
  parent; $3,243 million includes non-controlling interests.
- **Q021** uses year-end headcount; the report does not state sales per
  employee. A 1% tolerance is recorded in the `calculation` field.
- **Replaced candidates.** A comparative question on crude steel 2024→2025 was
  dropped because the 2024 figure appears only in the page 3 graphic. Q022
  compares 2025 production with shipments instead, using two independently
  stated figures.
