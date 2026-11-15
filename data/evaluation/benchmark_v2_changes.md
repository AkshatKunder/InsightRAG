# Benchmark v2: changes from v1

`benchmark_v2.json` is a copy of `benchmark_v1.json` with one correction.
v1 is kept unchanged so earlier results stay comparable. Phase 2
comparisons use v2 as the primary benchmark and also report v1.

## What changed

Only **Q017** (steel shipments 2024→2025) and **Q018** (iron ore production
2024→2025) changed. Everything else (question text, expected answers,
`expected_pages`, `supporting_evidence`, values, comparisons) is identical;
`app/test_benchmark_dataset.py` asserts this.

| Field | v1 | v2 |
|---|---|---|
| `expected_pages` | `[95]` | `[95]` (unchanged, page 95 stays the primary evidence) |
| `expected_facts[].pages` (2025 and 2024 facts) | `[95]` | `[95, 3]` |
| `alternative_evidence` | absent | two page 3 excerpts per question (values run + category label) |

Top-level metadata: `benchmark_name`, `version` (2.0), `derived_from` and
`changes_from_v1` record the change.

Effect on scoring: retrieving page 3 now counts as finding these facts (fact
recall, precision, reciprocal rank). Page recall is unchanged because it is
based on `expected_pages`.

## Why

Page 3 ("Company overview") has a KPI bar chart that states the same values
as the page 95 table:

| Category label (page 3) | 2025 | 2024 | Page 95 table |
|---|---|---|---|
| Steel shipments (million tonnes) | 54.0 | 54.3 | 54.0 / 54.3 |
| Iron ore production (million tonnes)* | 48.8 | 42.4 | 48.8 / 42.4 |

v1 already credited page 3 for the same facts in other questions (Q009 lists
page 3 for iron ore 2025, and Q022 lists it for shipments 2025), but not in
Q017/Q018. That inconsistency penalised a retriever for finding a page that
answers the question.

## How the chart mapping was verified

pypdf extracts the chart as `55.6 48.8 54.0 61.4 57.9 42.4 54.3 62.4 2025 2024`
followed by the four category labels. The text alone does not show which value
belongs to which bar, so the mapping was taken from the PDF's drawing
instructions instead:

```bash
python scripts/verify_chart_mapping.py
```

The script reads the chart's form XObject (`/XSVG1`) and decodes the labels
through the font's ToUnicode map. It then maps each value label to the bar
whose extent contains it, each bar colour to a year via the legend swatch
next to "2025" or "2024", and each bar to the category label it is aligned
with. Output on the 2025 report:

```
Crude steel production (million tonnes)       2025    55.6
Iron ore production (million tonnes)*         2025    48.8
Steel shipments (million tonnes)              2025    54.0
Sales (billion USD)                           2025    61.4
Crude steel production (million tonnes)       2024    57.9
Iron ore production (million tonnes)*         2024    42.4
Steel shipments (million tonnes)              2024    54.3
Sales (billion USD)                           2024    62.4
```

All eight values agree with pages 60 and 95 (and the narrative $61.4 /
$62.4 billion sales on page 95), which independently confirms the mapping.
The geometry logic is unit-tested on a synthetic chart in
`app/evaluation/test_chart_mapping.py`.

## Evidence excerpts

The `alternative_evidence` excerpts are copied from pypdf's text for page 3
and are checked word for word by `scripts/validate_benchmark.py` (v2: 52
excerpts, i.e. v1's 48 + 4). Because the values run does not by itself show
the mapping, each excerpt has a `note` naming the bar it belongs to and
pointing to the verification script.

## Caveat

A retriever that returns the page 3 chunk gives the generator the flattened
number run, not the chart. Page 3 is therefore valid evidence for retrieval
scoring, but answering from it requires reading values in the correct order.
That is why page 95 remains the primary expected page.
