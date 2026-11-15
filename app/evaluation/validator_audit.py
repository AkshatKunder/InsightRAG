"""Agreement between the system's claim validator and independent answer grading.

Pure function over saved pipeline runs and grading results: re-runs a
``validate`` callable on each saved answer and its citations, and cross-tabulates
the status against the deterministic correctness label. Used to measure
validator false positives (wrong answers marked ``SUPPORTED``) and false
negatives (correct answers not ``SUPPORTED``) before and after a change.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence


def audit(
    runs: Mapping,
    graded_runs: Sequence[Mapping],
    validate: Callable[[str, list, str], Mapping],
    questions: Mapping[str, str],
    enrich: Callable[[list], list] | None = None,
) -> dict:
    table: Counter = Counter()
    rows = []
    for run, graded in zip(runs["runs"], graded_runs):
        records = {record["id"]: record for record in run["questions"]}
        for item in graded["answerable"]:
            if item["label"] in ("abstained", "error"):
                continue
            record = records[item["id"]]
            citations = enrich(record["citations"]) if enrich else record["citations"]
            status = validate(record["answer"], citations, questions[item["id"]])["overall_status"]
            correct = item["label"] == "correct"
            table[(correct, status)] += 1
            rows.append({"run": graded.get("run"), "id": item["id"], "label": item["label"],
                         "old_status": item.get("validation_status"), "status": status})
    wrong = sum(v for (ok, _), v in table.items() if not ok)
    right = sum(v for (ok, _), v in table.items() if ok)
    false_pos = sum(v for (ok, s), v in table.items() if not ok and s == "SUPPORTED")
    false_neg = sum(v for (ok, s), v in table.items() if ok and s != "SUPPORTED")
    return {
        "wrong_or_partial_answers": wrong,
        "correct_answers": right,
        "false_positives": false_pos,
        "false_positive_rate": round(false_pos / wrong, 4) if wrong else None,
        "false_negatives": false_neg,
        "false_negative_rate": round(false_neg / right, 4) if right else None,
        "table": {f"{'correct' if ok else 'wrong_or_partial'}|{status}": n for (ok, status), n in sorted(table.items())},
        "rows": rows,
    }
