"""Structured-data evaluation helpers (Phase 4, Task 8). Pure functions, no models.

* ``result_match``: execution accuracy. The system result matches the gold
  result when it has the same number of rows and every gold column is matched
  by some system column with the same values (rounded; same order when the
  question is ordered, otherwise as a multiset). Extra system columns (e.g. a
  name next to a count) are allowed; column names are ignored.
* ``expected_period_change`` / ``expected_shares``: derived figures computed
  independently from the gold rows (not with the code under test).
* ``db_fingerprint``: row counts and a content hash per table, to prove that
  safety questions changed nothing.
"""

from __future__ import annotations

import hashlib
import sqlite3
from collections import Counter
from collections.abc import Mapping, Sequence
from itertools import product
from pathlib import Path


def _norm(value):
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        return round(float(value), 2)
    return str(value).strip().lower()


def _columns(rows: Sequence[Sequence]) -> list[list]:
    return [list(col) for col in zip(*rows)] if rows else []


def result_match(gold_rows: Sequence[Sequence], system_rows: Sequence[Sequence], ordered: bool = False) -> bool:
    if len(gold_rows) != len(system_rows):
        return False
    if not gold_rows:
        return True
    gold_cols = [[_norm(v) for v in c] for c in _columns(gold_rows)]
    sys_cols = [[_norm(v) for v in c] for c in _columns(system_rows)]
    if ordered:
        return all(any(gold == sys for sys in sys_cols) for gold in gold_cols)
    # Unordered: map every gold column to a system column with the same values, such
    # that the rows (projected onto the mapped columns) are the same multiset.
    candidates = [[i for i, sys in enumerate(sys_cols) if Counter(sys) == Counter(gold)] for gold in gold_cols]
    if any(not c for c in candidates):
        return False
    gold_multiset = Counter(tuple(_norm(v) for v in r) for r in gold_rows)
    for mapping in product(*candidates):
        if len(set(mapping)) != len(mapping):
            continue
        projected = Counter(tuple(_norm(r[i]) for i in mapping) for r in system_rows)
        if projected == gold_multiset:
            return True
    return False


def expected_period_change(gold_rows: Sequence[Sequence]) -> dict | None:
    """Change between the first and last period of a two-column (period, value) gold result."""

    rows = sorted(gold_rows, key=lambda r: str(r[0]))
    if len(rows) < 2 or len(rows[0]) != 2:
        return None
    before, after = rows[0][1], rows[-1][1]
    return {"absolute_change": after - before, "percent_change": round((after - before) / abs(before) * 100, 2) if before else None}


def expected_shares(gold_rows: Sequence[Sequence]) -> dict | None:
    total = sum(r[1] for r in gold_rows)
    if not total:
        return None
    return {str(r[0]).lower(): round(r[1] / total * 100, 2) for r in gold_rows}


def derived_correct(kind: str, gold_rows: Sequence[Sequence], analytics: Mapping) -> bool | None:
    if kind == "period_change":
        expected = expected_period_change(gold_rows)
        changes = analytics.get("period_changes") or []
        comparison = analytics.get("column_comparison")
        candidates = changes + ([comparison] if comparison else [])
        return bool(expected) and any(
            abs(c["absolute_change"] - expected["absolute_change"]) < 0.01
            and c.get("percent_change") is not None and abs(c["percent_change"] - expected["percent_change"]) < 0.01
            for c in candidates)
    if kind == "shares":
        expected = expected_shares(gold_rows)
        got = {str(s["label"]).lower(): s["share_percent"] for s in analytics.get("shares") or []}
        return bool(expected) and set(got) == set(expected) and all(abs(got[k] - v) < 0.05 for k, v in expected.items())
    return None


def db_fingerprint(db_path: Path | str) -> dict:
    conn = sqlite3.connect(f"file:{Path(db_path).resolve().as_posix()}?mode=ro", uri=True)
    try:
        names = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
        out = {}
        for name in names:
            rows = conn.execute(f'SELECT * FROM "{name}" ORDER BY 1').fetchall()
            out[name] = {"rows": len(rows), "sha256": hashlib.sha256(repr(rows).encode()).hexdigest()[:16]}
        return out
    finally:
        conn.close()


def rate(values: Sequence) -> float | None:
    values = [v for v in values if v is not None]
    return round(sum(1 for v in values if v) / len(values), 4) if values else None
