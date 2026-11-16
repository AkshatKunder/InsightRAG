"""Phase 4.3: compare the structured-data evaluation with Phases 4, 4.1 and 4.2 and write the report.

Usage (after ``scripts/evaluate_structured.py --output-dir data/evaluation/results/phase4_3 ...``):

    python scripts/build_phase4_3_report.py

Reads the saved ``structured_eval.json`` of Phase 4, 4.1, 4.2 and 4.3 only (earlier results are not
modified). Earlier runs are re-graded from their saved records with the Phase 4.3 rules: a period change needs the
right figure, scoped rows and no model-computed change columns (Phase 4.2 rule); shares need the right
figures, raw-value rows equal to the gold rows and no model-computed share/ratio columns (Phase 4.3 rule).
The figure-only grading is shown as well. Writes ``data/evaluation/phase4_3_evaluation_report.md`` and
``results/phase4_3/phase4_3_comparison.json``.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.evaluation.reporting import fmt, md_table, signed, write_text  # noqa: E402
from app.structured.scope import DERIVED_NAME_RE  # noqa: E402
from app.structured.shares import FRACTION_NAME_RE  # noqa: E402

EVAL = PROJECT_ROOT / "data" / "evaluation"
PHASES = {"phase4": "Phase 4", "phase4_1": "Phase 4.1", "phase4_2": "Phase 4.2", "phase4_3": "Phase 4.3"}
METRICS = [
    ("Routing accuracy (28 structured + 25 document questions)", "routing_accuracy"),
    ("Valid SQL generated", "sql_generation_valid"),
    ("Execution success", "execution_success"),
    ("Execution accuracy (result = gold result)", "execution_accuracy"),
    ("Execution accuracy, SQL-only questions", "execution_accuracy_sql_only"),
    ("Derived figures correct (figure only)", "derived_accuracy"),
    ("Chart type as expected", "chart_accuracy"),
    ("Unanswerable: no result returned", "unanswerable_abstention"),
    ("Safety: refused and database unchanged", "safety_refused_and_db_unchanged"),
    ("Hybrid: fully correct", "hybrid_correct"),
    ("Hybrid: document page cited", "hybrid_document_page_cited"),
]


def _mean(values):
    values = [v for v in values if v is not None]
    return round(sum(values) / len(values), 4) if values else None


def strict(rec: dict, derived: dict[str, str]) -> bool:
    kind = derived.get(rec["id"])
    if kind is None:
        return bool(rec["correct"])
    name_re = DERIVED_NAME_RE if kind == "period_change" else FRACTION_NAME_RE
    model_cols = [c for c in rec.get("columns") or [] if name_re.search(c)]
    return bool(rec.get("derived_correct")) and bool(rec.get("result_match")) and not model_cols


def lenient(rec: dict, derived: dict[str, str]) -> bool:
    return bool(rec.get("derived_correct")) if rec["id"] in derived else bool(rec["correct"])


def build() -> dict:
    benchmark = json.loads((EVAL / "structured_benchmark_v1.json").read_text(encoding="utf-8"))
    pc_ids = {q["id"]: q["expected_derived"] for q in benchmark["questions"] if q.get("expected_derived")}
    data = {k: json.loads((EVAL / "results" / k / "structured_eval.json").read_text(encoding="utf-8")) for k in PHASES}
    metrics = {key: {p: _mean([r["summary"].get(key) for r in d["runs"]]) for p, d in data.items()} for _, key in METRICS}
    for name, fn in (("overall_correct_strict", strict), ("overall_correct_lenient", lenient)):
        metrics[name] = {p: _mean([sum(fn(rec, pc_ids) for rec in r["records"]) / len(r["records"]) for r in d["runs"]])
                         for p, d in data.items()}
    per_q: dict[str, dict] = {}
    for p, d in data.items():
        for run in d["runs"]:
            for rec in run["records"]:
                per_q.setdefault(rec["id"], {}).setdefault(p, []).append(strict(rec, pc_ids))
    changes = []
    for qid, by in per_q.items():
        rate = {p: sum(v) / len(v) for p, v in by.items()}
        status = ("improved" if rate["phase4_3"] > rate["phase4_2"] else
                  "regressed" if rate["phase4_3"] < rate["phase4_2"] else "unchanged")
        changes.append({"id": qid, **by, "status_vs_phase4_2": status})
    details = []
    for qid in sorted(pc_ids):
        for p in ("phase4_2", "phase4_3"):
            rec = next(r for r in data[p]["runs"][0]["records"] if r["id"] == qid)
            details.append({"id": qid, "phase": PHASES[p], "strict_correct": strict(rec, pc_ids),
                            "derived_correct": rec.get("derived_correct"), "result_match": rec.get("result_match"),
                            "sql": rec.get("sql"), "columns": rec.get("columns"), "rows": (rec.get("rows") or [])[:12],
                            "scope": rec.get("scope"), "presentation": rec.get("presentation"),
                            "abstained": rec.get("abstained"), "answer": rec.get("answer"),
                            "share_validation": rec.get("share_validation")})
    routing_errors = [r for run in data["phase4_3"]["runs"] for r in run["routing"] if r["predicted"] != r["expected"]]
    return {"generated": datetime.now(timezone.utc).isoformat(timespec="seconds"), "metrics": metrics,
            "changes": changes, "details": details, "routing_errors_phase4_3": routing_errors,
            "runs": {p: len(d["runs"]) for p, d in data.items()},
            "generation_options": data["phase4_3"]["generation_options"]}


def render(data: dict, findings: str) -> str:
    m = data["metrics"]
    labels = METRICS + [("Overall correct, Phase 4.3 grading (period change and shares: figures + scoped raw rows + "
                         "no model-computed columns)", "overall_correct_strict"),
                        ("Overall correct, figure-only grading (Phase 4.1 rule)", "overall_correct_lenient")]
    rows = [[label] + [fmt(m[k][p]) for p in PHASES]
            + [signed(m[k]["phase4_3"] - m[k]["phase4_2"]) if None not in (m[k]["phase4_2"], m[k]["phase4_3"]) else "—"]
            for label, k in labels]
    changed = [c for c in data["changes"] if c["status_vs_phase4_2"] != "unchanged"]
    counts = {s: sum(1 for c in data["changes"] if c["status_vs_phase4_2"] == s) for s in ("improved", "regressed", "unchanged")}
    yn = lambda xs: " / ".join("yes" if x else "no" for x in xs or [])  # noqa: E731
    return "\n".join([
        "# InsightRAG Phase 4.3: deterministic share and ratio calculations",
        "",
        f"_Generated by `scripts/build_phase4_3_report.py` on {data['generated']} from saved results (runs: "
        f"{', '.join(f'{PHASES[p]} {n}' for p, n in data['runs'].items())}; generation options "
        f"{json.dumps(data['generation_options'])}). Benchmarks `structured_benchmark_v1.json` and `benchmark_v2.json` "
        "and all earlier results are unchanged. Earlier phases are re-graded here, from their saved records, with "
        "the Phase 4.3 rules for period changes and shares. The qualitative section is hand-written "
        "(`data/evaluation/phase4_3_findings.md`)._",
        "",
        "## 1. Metrics (measured; mean over runs)",
        "",
        md_table(["Metric", "Phase 4", "Phase 4.1", "Phase 4.2", "Phase 4.3", "Δ vs 4.2"], rows, ["l", "r", "r", "r", "r", "r"]),
        "",
        "## 2. Per-question changes vs Phase 4.2 (Phase 4.3 grading)",
        "",
        f"Improved {counts['improved']}, regressed {counts['regressed']}, unchanged {counts['unchanged']}.",
        "",
        md_table(["ID"] + [f"{v} (runs)" for v in PHASES.values()] + ["Change"],
                 [[c["id"]] + [yn(c.get(p)) for p in PHASES] + [c["status_vs_phase4_2"]] for c in changed],
                 ["l"] * (len(PHASES) + 2)) if changed else "No question changed.",
        "",
        "Routing errors in Phase 4.3: " + (", ".join(f"{r['id']} ({r['expected']} → {r['predicted']})"
                                                       for r in data["routing_errors_phase4_3"]) or "none") + ".",
        "",
        "## 3. Period-change and share questions, Phase 4.2 vs Phase 4.3 (run 1)",
        "",
        "\n\n".join(
            f"**{d['id']} — {d['phase']}**: correct (Phase 4.3 grading): {'yes' if d['strict_correct'] else 'no'}; "
            f"figure correct: {'yes' if d['derived_correct'] else 'no'}; rows = gold rows: {'yes' if d['result_match'] else 'no'}"
            + (f"; abstained: yes" if d["abstained"] else "")
            + f"\n\n```sql\n{d['sql'] or '(no SQL)'}\n```\n\nPresented columns: `{d['columns']}`  \nRows: `{d['rows']}`"
            + (f"  \nScope check: `{json.dumps(d['scope'])}`" if d["scope"] else "")
            + (f"  \nPresentation: `{json.dumps(d['presentation'])}`" if d["presentation"] else "")
            + (f"  \nShare validation: `{json.dumps(d['share_validation'], default=str)}`" if d.get("share_validation") else "")
            + f"\n\nAnswer:\n\n```\n{(d['answer'] or '').strip()}\n```"
            for d in data["details"]),
        "",
        "## 4. Findings, regressions and limitations (qualitative)",
        "",
        findings or "_No qualitative notes recorded._",
    ])


def main() -> int:
    data = build()
    out = EVAL / "results" / "phase4_3" / "phase4_3_comparison.json"
    out.write_text(json.dumps(data, indent=2, default=str) + "\n", encoding="utf-8")
    findings_path = EVAL / "phase4_3_findings.md"
    findings = findings_path.read_text(encoding="utf-8").split("\n", 2)[-1].strip() if findings_path.is_file() else ""
    write_text(EVAL / "phase4_3_evaluation_report.md", render(data, findings))
    print(f"Wrote {EVAL / 'phase4_3_evaluation_report.md'} and {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
