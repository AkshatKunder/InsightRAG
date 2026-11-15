"""Phase 3.1: compare Phase 3.1 with Phase 2 and Phase 3 and write the evaluation report.

Usage (from the project root, after the Phase 3.1 answer evaluation):

    python scripts/evaluate_answers.py --pipeline phase3_1 --temperature 0 --seed 42 --repeats 2 \
        --output-dir data/evaluation/results/phase3_1/phase3_1_pipeline \
        --report-dir data/evaluation/phase3_1/phase3_1_pipeline
    python scripts/build_phase3_1_report.py

Reads saved results only (Phase 2: ``results/``; Phase 3: ``results/phase3/phase3_pipeline``;
Phase 3.1: ``results/phase3_1/phase3_1_pipeline``). Writes
``data/evaluation/phase3_1_evaluation_report.md`` and
``data/evaluation/results/phase3_1/phase3_1_comparison.json``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.evaluation.reporting import md_table, svg_grouped_bars, write_text  # noqa: E402
from app.evaluation.retrieval_evaluator import load_benchmark  # noqa: E402


def _phase3_module():
    spec = importlib.util.spec_from_file_location("build_phase3_report", PROJECT_ROOT / "scripts" / "build_phase3_report.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    return module


p3r = _phase3_module()
EVAL, RESULTS = p3r.EVAL, p3r.RESULTS
VARIANTS = {
    "phase2": RESULTS,
    "phase3": RESULTS / "phase3" / "phase3_pipeline",
    "phase3_1": RESULTS / "phase3_1" / "phase3_1_pipeline",
}
NAMES = {"phase2": "Phase 2 (sampled, 3 runs)", "phase3": "Phase 3 (det., 2 runs)", "phase3_1": "Phase 3.1 (det., 2 runs)"}
FOCUS = ("Q001", "Q002", "Q018", "Q022")


def _metrics(benchmark: dict, variant: dict) -> dict:
    a = variant["summary"]["v2"]["metrics"]["answerable"]
    u = variant["summary"]["v2"]["metrics"]["unanswerable"]
    n = variant["numerical"]["aggregate"]
    return {
        "runs": variant["summary"]["v2"]["num_runs"],
        "generation": variant["runs"]["config"].get("generation") or variant["runs"]["config"].get("llm_options"),
        "answer_accuracy": a["answer_accuracy"], "correct_or_partial": a["correct_or_partial_rate"],
        "completeness": a["mean_completeness"], "citation_precision": a["citation_precision_micro"],
        "citation_fact_recall": a["citation_fact_recall"], "numeric_facts_matched": n["overall"]["matched"],
        "numeric_by_figure_type": {k: v["matched"] for k, v in n["by_figure_type"].items()},
        "comparison_absolute_change": n["comparisons"]["absolute_change_ok"],
        "derived_result": n["calculations"]["result_ok"], "context_fact_recall": a["context_fact_recall"],
        "correct_abstention": u["correct_abstention_rate"], "false_answer_unanswerable": u["false_answer_rate"],
        "false_abstention": a["false_abstention_rate"], "by_category": variant["summary"]["v2"]["by_category"],
        "stable_questions": variant["summary"]["v2"]["stable_questions"],
        "context_retrieval": p3r.context_retrieval(benchmark, variant["runs"]),
    }


def _labels(variant: dict) -> dict:
    return {s["id"]: s for s in variant["summary"]["v2"]["answerable_stability"]}


def per_question(variants: dict) -> list[dict]:
    labels = {name: _labels(v) for name, v in variants.items()}
    rows = []
    for qid, item in labels["phase3_1"].items():
        maj = {name: p3r.majority(labels[name][qid]["labels"]) for name in variants}

        def change(other):
            delta = p3r.RANK[maj["phase3_1"]] - p3r.RANK[maj[other]]
            return "improved" if delta > 0 else "regressed" if delta < 0 else "unchanged"

        rows.append({"id": qid, "category": item["category"], "labels": {n: labels[n][qid]["labels"] for n in variants},
                     "majority": maj, "vs_phase2": change("phase2"), "vs_phase3": change("phase3")})
    return rows


def focus_details(variant: dict) -> list[dict]:
    """Phase 3.1 run-1 records for the targeted questions: answer, calculation, completeness, citations."""

    records = {r["id"]: r for r in variant["runs"]["runs"][0]["questions"]}
    graded = {q["id"]: q for q in variant["graded"]["runs"][0]["answerable"]}
    out = []
    for qid in FOCUS:
        record = records.get(qid, {})
        out.append({
            "id": qid, "label": graded.get(qid, {}).get("label"), "answer": record.get("answer"),
            "context_pages": [c["page"] for c in record.get("context", [])],
            "cited_pages": [c["page"] for c in record.get("citations", [])],
            "calculation": record.get("calculation"), "completeness": record.get("completeness"),
            "revised_from": record.get("revised_from"), "validation_status": (record.get("validation") or {}).get("overall_status"),
        })
    return out


def build(args) -> dict:
    benchmark = load_benchmark(args.benchmark)
    variants = {name: p3r.load_variant(path) for name, path in VARIANTS.items()}
    rows = per_question(variants)
    new = variants["phase3_1"]
    completeness_flags = Counter()
    for run in new["runs"]["runs"]:
        for record in run["questions"]:
            if record.get("completeness"):
                completeness_flags[(record["id"], record["completeness"]["status"])] += 1
    return {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "benchmark": args.benchmark.name,
        "variants": NAMES,
        "metrics": {name: _metrics(benchmark, v) for name, v in variants.items()},
        "per_question": rows,
        "changes_vs_phase2": dict(Counter(r["vs_phase2"] for r in rows)),
        "changes_vs_phase3": dict(Counter(r["vs_phase3"] for r in rows)),
        "focus": focus_details(new),
        "completeness_flags": {f"{qid}|{status}": n for (qid, status), n in sorted(completeness_flags.items())},
        "calculations": sorted({(r["id"], (r.get("calculation") or {}).get("sentence")) for run in new["runs"]["runs"]
                                for r in run["questions"] if r.get("calculation")}),
        "revisions": p3r.revisions(benchmark, new),
        "rejected": p3r.rejected_answers(benchmark, new),
    }


def render(data: dict, findings: str) -> str:
    m = data["metrics"]
    p2, p3, p31 = m["phase2"], m["phase3"], m["phase3_1"]
    keys = [
        ("Answer accuracy (correct)", "answer_accuracy"), ("Correct or partially correct", "correct_or_partial"),
        ("Mean completeness", "completeness"), ("Citation precision (micro)", "citation_precision"),
        ("Citation fact recall", "citation_fact_recall"), ("Numeric facts fully matched", "numeric_facts_matched"),
        ("Absolute change correct", "comparison_absolute_change"), ("Derived result within tolerance", "derived_result"),
        ("Context fact recall (5 chunks)", "context_fact_recall"), ("Unanswerable: correct abstention", "correct_abstention"),
        ("Unanswerable: false answers", "false_answer_unanswerable"), ("False abstention (answerable)", "false_abstention"),
    ]
    rows = [[label, p3r._m(p2[k]), p3r._m(p3[k]), p3r._m(p31[k]), p3r._delta(p2[k], p31[k]), p3r._delta(p3[k], p31[k])] for label, k in keys]
    cats = list(p31["by_category"])
    write_text(EVAL / "charts" / "phase3_1_accuracy_by_category.svg", svg_grouped_bars(
        "Answer accuracy by category: Phase 2, Phase 3, Phase 3.1 (benchmark v2)", cats,
        {NAMES[n]: [p3r._mean(m[n]["by_category"][c]["accuracy"]) for c in cats] for n in ("phase2", "phase3", "phase3_1")},
        y_label="Accuracy"))
    types = sorted(set(p31["numeric_by_figure_type"]) | set(p2["numeric_by_figure_type"]))
    lines = [
        "# InsightRAG Phase 3.1: evaluation report",
        "",
        f"_Generated by `scripts/build_phase3_1_report.py` on {data['generated']} from saved results. Benchmark "
        f"`{data['benchmark']}` (v1 and v2 unchanged). The **Qualitative** section is hand-written "
        "(`data/evaluation/phase3_1_findings.md`); all numbers come from the result files._",
        "",
        md_table(["Variant", "Runs", "Generation settings"],
                 [[NAMES[n], m[n]["runs"], json.dumps(m[n]["generation"]) if not isinstance(m[n]["generation"], str) else m[n]["generation"]]
                  for n in m], ["l", "r", "l"]),
        "",
        "## 1. Headline comparison (measured; Δ calculated)",
        "",
        md_table(["Metric", "Phase 2", "Phase 3", "Phase 3.1", "Δ vs Phase 2", "Δ vs Phase 3"], rows, ["l", "r", "r", "r", "r", "r"]),
        "",
        "## 2. Context retrieval (5 chunks given to the generator)",
        "",
        md_table(["Variant", "Fact recall@5", "Page recall@5", "Precision@5", "MRR", "Questions missing a fact page"],
                 [[NAMES[n], m[n]["context_retrieval"]["recall"]["5"], m[n]["context_retrieval"]["page_recall"]["5"],
                   m[n]["context_retrieval"]["precision"]["5"], m[n]["context_retrieval"]["mrr"],
                   ", ".join(m[n]["context_retrieval"]["questions_missing_a_fact_page"]) or "none"] for n in m],
                 ["l", "r", "r", "r", "r", "l"]),
        "",
        "## 3. Category-level accuracy",
        "",
        md_table(["Category", "Phase 2", "Phase 3", "Phase 3.1", "Δ vs Phase 3"],
                 [[c, p3r._m(p2["by_category"][c]["accuracy"]), p3r._m(p3["by_category"][c]["accuracy"]),
                   p3r._m(p31["by_category"][c]["accuracy"]), p3r._delta(p3["by_category"][c]["accuracy"], p31["by_category"][c]["accuracy"])]
                  for c in cats]),
        "",
        "![Accuracy by category](charts/phase3_1_accuracy_by_category.svg)",
        "",
        "## 4. Per-question changes (majority label across runs)",
        "",
        f"Versus Phase 2: {data['changes_vs_phase2']}. Versus Phase 3: {data['changes_vs_phase3']}.",
        "",
        md_table(["ID", "Category", "Phase 2", "Phase 3", "Phase 3.1", "vs Phase 2", "vs Phase 3"],
                 [[r["id"], r["category"], " / ".join(r["labels"]["phase2"]), " / ".join(r["labels"]["phase3"]),
                   " / ".join(r["labels"]["phase3_1"]), r["vs_phase2"], r["vs_phase3"]] for r in data["per_question"]],
                 ["l"] * 7),
        "",
        "## 5. Targeted questions (Phase 3.1, run 1)",
        "",
        md_table(["ID", "Label", "Context pages", "Cited pages", "Answer", "Calculation operands", "Completeness"],
                 [[f["id"], f["label"], ", ".join(map(str, f["context_pages"])), ", ".join(map(str, f["cited_pages"])),
                   (f["answer"] or "").replace("\n", " ")[:170],
                   "; ".join(f"{o['label']} = {o['raw']} (p{o['page']})" for o in (f["calculation"] or {}).get("operands", [])) or "—",
                   (f"{f['completeness']['status']}; missing {f['completeness']['missing'] or 'none'}; extra {f['completeness']['extra'] or 'none'}"
                    if f["completeness"] else "—")] for f in data["focus"]],
                 ["l"] * 7),
        "",
        "## 6. Numerical accuracy by figure type (numeric facts fully matched)",
        "",
        md_table(["Figure type", "Phase 2", "Phase 3", "Phase 3.1"],
                 [[t, p3r._m(p2["numeric_by_figure_type"].get(t)), p3r._m(p3["numeric_by_figure_type"].get(t)),
                   p3r._m(p31["numeric_by_figure_type"].get(t))] for t in types]),
        "",
        "## 7. Programmatic calculations, revisions and completeness (Phase 3.1, measured)",
        "",
        "Computed results (operands from evidence; never from the LLM):",
        "",
        md_table(["ID", "Computed sentence"], [[qid, s] for qid, s in data["calculations"]], ["l", "l"]) if data["calculations"] else "None.",
        "",
        "Completeness checks (question-runs): " + (", ".join(f"{k} × {v}" for k, v in data["completeness_flags"].items()) or "none") + ".",
        "",
        md_table(["Run", "ID", "First answer graded", "Final graded", "Reason (first)"],
                 [[r["run"], r["id"], r["original_label"], r["final_label"], (r["reasons"] or [""])[0][:120]] for r in data["revisions"]],
                 ["r", "l", "l", "l", "l"]) if data["revisions"] else "No revisions.",
        "",
        "Answers replaced by abstention after contradiction: "
        + (", ".join(f"{r['id']} (run {r['run']}, was {r['rejected_answer_label']})" for r in data["rejected"]) or "none") + ".",
        "",
        f"Stability: {p31['stable_questions']}/22 questions with the same label in both Phase 3.1 runs.",
        "",
        "## 8. Findings and limitations (qualitative)",
        "",
        findings or "_No qualitative notes recorded._",
    ]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--benchmark", type=Path, default=EVAL / "benchmark_v2.json")
    parser.add_argument("--findings", type=Path, default=EVAL / "phase3_1_findings.md")
    parser.add_argument("--output", type=Path, default=EVAL / "phase3_1_evaluation_report.md")
    args = parser.parse_args(argv)
    data = build(args)
    out = RESULTS / "phase3_1" / "phase3_1_comparison.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    findings = args.findings.read_text(encoding="utf-8").split("\n", 2)[-1].strip() if args.findings.is_file() else ""
    write_text(args.output, render(data, findings))
    print(f"Wrote {args.output} and {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
