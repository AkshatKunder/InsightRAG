"""Phase 3 Task 7: compare Phase 3 against Phase 2 and write the evaluation report.

Usage (from the project root, after the answer evaluations have run):

    python scripts/evaluate_answers.py --pipeline phase2 --temperature 0 --seed 42 --repeats 1 \
        --output-dir data/evaluation/results/phase3/baseline_phase2_pipeline \
        --report-dir data/evaluation/phase3/baseline_phase2_pipeline
    python scripts/evaluate_answers.py --pipeline phase3 --temperature 0 --seed 42 --repeats 2 \
        --output-dir data/evaluation/results/phase3/phase3_pipeline \
        --report-dir data/evaluation/phase3/phase3_pipeline
    python scripts/build_phase3_report.py

Three variants are compared on benchmark v2 (v1/v2 files are only read):

* ``phase2``: the saved Phase 2 results (production sampling, 3 runs);
* ``baseline``: the Phase 2 pipeline configuration re-run with the same
  deterministic generation settings as Phase 3 (isolates the effect of the
  Phase 3 changes from the effect of deterministic sampling);
* ``phase3``: the Phase 3 pipeline.

It also re-scores the 5-chunk generator contexts with the retrieval metrics,
audits the claim validator against the independent grading, and grades the
answers that Phase 3 replaced with abstentions. Writes
``data/evaluation/phase3_evaluation_report.md`` and
``data/evaluation/results/phase3/phase3_comparison.json``.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.evaluation.answer_evaluator import evaluate_answerable  # noqa: E402
from app.evaluation.reporting import fmt, md_table, signed, svg_grouped_bars, write_text  # noqa: E402
from app.evaluation.retrieval_evaluator import evaluate_results, load_benchmark  # noqa: E402
from app.evaluation.validator_audit import audit  # noqa: E402

EVAL = PROJECT_ROOT / "data" / "evaluation"
RESULTS = EVAL / "results"
VARIANTS = {
    "phase2": RESULTS,
    "baseline": RESULTS / "phase3" / "baseline_phase2_pipeline",
    "phase3": RESULTS / "phase3" / "phase3_pipeline",
}
NAMES = {"phase2": "Phase 2 (saved, sampled, 3 runs)", "baseline": "Phase 2 pipeline, deterministic", "phase3": "Phase 3 pipeline, deterministic"}
RANK = {"correct": 3, "partially_correct": 2, "abstained": 1, "incorrect": 0, "error": 0}


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_variant(directory: Path) -> dict:
    missing = [n for n in ("answer_runs.json", "answer_evaluation.json", "answer_evaluation_summary.json", "numerical_evaluation.json")
               if not (directory / n).is_file()]
    if missing:
        raise SystemExit(f"Missing {', '.join(missing)} in {directory}; run scripts/evaluate_answers.py first.")
    return {
        "runs": _load(directory / "answer_runs.json"),
        "graded": _load(directory / "answer_evaluation.json"),
        "summary": _load(directory / "answer_evaluation_summary.json"),
        "numerical": _load(directory / "numerical_evaluation.json"),
    }


def _mean(entry):
    return entry["mean"] if isinstance(entry, dict) and entry else entry


def majority(labels: list[str]) -> str:
    counts = Counter(labels)
    top = max(counts.values())
    return max((label for label, n in counts.items() if n == top), key=lambda label: RANK.get(label, 0))


def context_retrieval(benchmark: dict, runs: dict, k_values=(1, 3, 5)) -> dict:
    """Retrieval metrics of the 5-chunk generator context, averaged over runs."""

    answerable = {q["id"] for q in benchmark["questions"] if q.get("answerable")}
    reports = [
        evaluate_results(benchmark, {r["id"]: r["context"] for r in run["questions"] if r["id"] in answerable}, list(k_values))
        for run in runs["runs"]
    ]
    out = {"mrr": round(statistics.mean(r["summary"]["overall"]["mrr"] for r in reports), 4)}
    for metric in ("recall", "page_recall", "precision"):
        out[metric] = {str(k): round(statistics.mean(r["summary"]["overall"][metric][str(k)] for r in reports), 4) for k in k_values}
    complete = Counter()
    for report in reports:
        for q in report["questions"]:
            complete[q["id"]] += q["metrics"]["recall"][str(max(k_values))] == 1.0
    out["questions_missing_a_fact_page"] = sorted(qid for qid in answerable if complete[qid] < len(reports))
    return out


def per_question(variants: dict) -> list[dict]:
    rows = []
    first = variants["phase3"]["summary"]["v2"]["answerable_stability"]
    by_variant = {name: {s["id"]: s for s in v["summary"]["v2"]["answerable_stability"]} for name, v in variants.items()}
    for item in first:
        qid = item["id"]
        labels = {name: by_variant[name][qid]["labels"] for name in variants}
        maj = {name: majority(labs) for name, labs in labels.items()}
        change = RANK[maj["phase3"]] - RANK[maj["phase2"]]
        rows.append({
            "id": qid, "category": item["category"], "labels": labels, "majority": maj,
            "vs_phase2": "improved" if change > 0 else "regressed" if change < 0 else "unchanged",
            "vs_baseline": ("improved" if RANK[maj["phase3"]] > RANK[maj["baseline"]] else
                            "regressed" if RANK[maj["phase3"]] < RANK[maj["baseline"]] else "unchanged"),
        })
    return rows


def rejected_answers(benchmark: dict, variant: dict) -> list[dict]:
    """Grade every answer Phase 3 discarded by abstaining on a contradiction."""

    questions = {q["id"]: q for q in benchmark["questions"]}
    out = []
    for run in variant["runs"]["runs"]:
        for record in run["questions"]:
            if not record.get("rejected_answer"):
                continue
            question = questions[record["id"]]
            label = "n/a (unanswerable)"
            if question.get("answerable"):
                graded = evaluate_answerable(question, {**record, "answer": record["rejected_answer"],
                                                        "validation": record.get("rejected_validation")})
                label = graded["label"]
            reasons = [c["explanation"] for c in (record.get("rejected_validation") or {}).get("claims", [])]
            out.append({"run": run["run"], "id": record["id"], "rejected_answer": record["rejected_answer"],
                        "rejected_answer_label": label, "reasons": reasons})
    return out


def revisions(benchmark: dict, variant: dict) -> list[dict]:
    """Answers regenerated once after a contradiction, with the original and final grades."""

    questions = {q["id"]: q for q in benchmark["questions"]}
    final_labels = {(g["run"], q["id"]): q["label"] for g in variant["graded"]["runs"] for q in g["answerable"]}
    out = []
    for run in variant["runs"]["runs"]:
        for record in run["questions"]:
            if not record.get("revised_from"):
                continue
            question = questions[record["id"]]
            original = "n/a (unanswerable)"
            if question.get("answerable"):
                original = evaluate_answerable(question, {**record, "answer": record["revised_from"], "validation": {}})["label"]
            out.append({"run": run["run"], "id": record["id"], "original_label": original,
                        "final_label": final_labels.get((run["run"], record["id"]), "n/a (unanswerable)"),
                        "reasons": record.get("revision_reasons", []), "original": record["revised_from"], "final": record["answer"]})
    return out


def validator_audits(variants: dict, questions: dict, pdf_path: Path) -> dict:
    from app.rag.citation_validator import validate_citation

    result = {}
    for name, variant in variants.items():
        statuses = {
            (graded["run"], q["id"]): q["validation_status"]
            for graded in variant["graded"]["runs"] for q in graded["answerable"]
        }
        # As recorded during the run (the validator version that ran then).
        table = Counter()
        for graded in variant["graded"]["runs"]:
            for q in graded["answerable"]:
                if q["label"] in ("abstained", "error"):
                    continue
                table[(q["label"] == "correct", statuses[(graded["run"], q["id"])])] += 1
        wrong = sum(v for (ok, _), v in table.items() if not ok)
        right = sum(v for (ok, _), v in table.items() if ok)
        fp = sum(v for (ok, s), v in table.items() if not ok and s == "SUPPORTED")
        fn = sum(v for (ok, s), v in table.items() if ok and s != "SUPPORTED")
        result[name] = {"recorded": {"wrong_or_partial_answers": wrong, "correct_answers": right, "false_positives": fp,
                                     "false_positive_rate": round(fp / wrong, 4) if wrong else None,
                                     "false_negatives": fn, "false_negative_rate": round(fn / right, 4) if right else None}}
    # Phase 2 answers re-validated with the Phase 3 validator (citations enriched with scope metadata).
    if pdf_path.is_file():
        from app.rag.chart_extractor import chart_values_by_page
        from app.rag.chunker import chunk_pages
        from app.rag.document_context import ScopeIndex
        from app.rag.loader import load_pdf

        pages = load_pdf(pdf_path)
        index = ScopeIndex(chunk_pages(pages), pages, chart_values_by_page(pdf_path))
        p2 = variants["phase2"]
        full = audit(p2["runs"], p2["graded"]["runs"], lambda a, c, q: validate_citation(a, c, question=q), questions, index.enrich)
        result["phase2_answers_with_phase3_validator"] = {k: v for k, v in full.items() if k != "rows"}
    return result


def build(args) -> dict:
    benchmark = load_benchmark(args.benchmark)
    variants = {name: load_variant(path) for name, path in VARIANTS.items()}
    questions = {q["id"]: q["question"] for q in benchmark["questions"]}
    metrics = {}
    for name, v in variants.items():
        a = v["summary"]["v2"]["metrics"]["answerable"]
        u = v["summary"]["v2"]["metrics"]["unanswerable"]
        n = v["numerical"]["aggregate"]
        metrics[name] = {
            "runs": v["summary"]["v2"]["num_runs"],
            "generation": v["runs"]["config"].get("generation") or v["runs"]["config"].get("llm_options"),
            "answer_accuracy": a["answer_accuracy"], "correct_or_partial": a["correct_or_partial_rate"],
            "completeness": a["mean_completeness"], "false_abstention": a["false_abstention_rate"],
            "citation_precision": a["citation_precision_micro"], "citation_fact_recall": a["citation_fact_recall"],
            "numeric_value_support_in_citations": a["matched_numeric_fact_citation_support"],
            "answers_with_numbers_not_in_context": a["answers_with_numbers_not_in_context"],
            "context_fact_recall": a["context_fact_recall"],
            "numeric_facts_matched": n["overall"]["matched"],
            "numeric_by_figure_type": {k: val["matched"] for k, val in n["by_figure_type"].items()},
            "comparison_direction": n["comparisons"]["direction_ok"],
            "comparison_absolute_change": n["comparisons"]["absolute_change_ok"],
            "derived_result": n["calculations"]["result_ok"],
            "correct_abstention": u["correct_abstention_rate"], "false_answer_unanswerable": u["false_answer_rate"],
            "by_category": v["summary"]["v2"]["by_category"],
            "stable_questions": v["summary"]["v2"]["stable_questions"],
            "context_retrieval": context_retrieval(benchmark, v["runs"]),
        }
    data = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "benchmark": args.benchmark.name,
        "variants": {name: NAMES[name] for name in variants},
        "metrics": metrics,
        "per_question": per_question(variants),
        "rejected_answers_phase3": rejected_answers(benchmark, variants["phase3"]),
        "revisions_phase3": revisions(benchmark, variants["phase3"]),
        "validator": validator_audits(variants, questions, args.pdf),
        "retrieval_stage": _load(RESULTS / "strategy_comparison_summary.json")["tables"]["v2"]["hybrid+rerank"]
        if (RESULTS / "strategy_comparison_summary.json").is_file() else None,
    }
    counts = Counter(r["vs_phase2"] for r in data["per_question"])
    data["question_changes_vs_phase2"] = dict(counts)
    data["question_changes_vs_baseline"] = dict(Counter(r["vs_baseline"] for r in data["per_question"]))
    return data


def _m(entry) -> str:
    if isinstance(entry, dict) and "mean" in entry:
        if entry["min"] == entry["max"]:
            return fmt(entry["mean"])
        return f"{fmt(entry['mean'])} ({fmt(entry['min'])}–{fmt(entry['max'])})"
    return fmt(entry)


def _delta(a, b) -> str:
    a, b = _mean(a), _mean(b)
    return signed(b - a) if isinstance(a, (int, float)) and isinstance(b, (int, float)) else "—"


def render(data: dict, findings: str) -> str:
    m = data["metrics"]
    p2, bl, p3 = m["phase2"], m["baseline"], m["phase3"]
    rows = []
    for label, key in [
        ("Answer accuracy (correct)", "answer_accuracy"), ("Correct or partially correct", "correct_or_partial"),
        ("Mean completeness", "completeness"), ("False abstention (answerable)", "false_abstention"),
        ("Citation precision (micro)", "citation_precision"), ("Citation fact recall", "citation_fact_recall"),
        ("Matched numeric facts with value in a cited passage", "numeric_value_support_in_citations"),
        ("Numeric facts fully matched", "numeric_facts_matched"), ("Comparison direction correct", "comparison_direction"),
        ("Absolute change correct", "comparison_absolute_change"), ("Derived result within tolerance", "derived_result"),
        ("Context fact recall (5 chunks)", "context_fact_recall"),
        ("Unanswerable: correct abstention", "correct_abstention"), ("Unanswerable: false answers", "false_answer_unanswerable"),
    ]:
        rows.append([label, _m(p2[key]), _m(bl[key]), _m(p3[key]), _delta(p2[key], p3[key]), _delta(bl[key], p3[key])])

    cats = list(p3["by_category"])
    write_text(EVAL / "charts" / "phase3_accuracy_by_category.svg", svg_grouped_bars(
        "Answer accuracy by category: Phase 2 vs Phase 3 (benchmark v2)", cats,
        {"Phase 2 (sampled)": [_mean(p2["by_category"][c]["accuracy"]) for c in cats],
         "Phase 2 pipeline (det.)": [_mean(bl["by_category"][c]["accuracy"]) for c in cats],
         "Phase 3 (det.)": [_mean(p3["by_category"][c]["accuracy"]) for c in cats]},
        y_label="Accuracy"))

    figure_types = sorted(set(p3["numeric_by_figure_type"]) | set(p2["numeric_by_figure_type"]))
    cr = {k: m[k]["context_retrieval"] for k in m}
    val = data["validator"]
    lines = [
        "# InsightRAG Phase 3: evaluation report",
        "",
        f"_Generated by `scripts/build_phase3_report.py` on {data['generated']} from saved results. Benchmark: "
        f"`{data['benchmark']}` (v1 and v2 unchanged). The section marked **Qualitative** is hand-written "
        "(`data/evaluation/phase3_findings.md`); all numbers are measured or calculated from the result files._",
        "",
        "## Variants compared",
        "",
        md_table(["Variant", "Runs", "Generation settings"],
                 [[NAMES[k], m[k]["runs"], json.dumps(m[k]["generation"]) if not isinstance(m[k]["generation"], str) else m[k]["generation"]] for k in m],
                 ["l", "r", "l"]),
        "",
        "*Phase 2 pipeline (det.)* re-runs the Phase 2 configuration with the same deterministic settings as Phase 3, "
        "so **its column vs Phase 3 isolates the Phase 3 changes**; the saved Phase 2 column shows the original baseline.",
        "",
        "## 1. Headline comparison (measured; Δ calculated)",
        "",
        md_table(["Metric", "Phase 2", "Phase 2 pipeline (det.)", "Phase 3", "Δ vs Phase 2", "Δ vs det. baseline"], rows,
                 ["l", "r", "r", "r", "r", "r"]),
        "",
        "## 2. Retrieval (measured)",
        "",
        "Retrieval components (BM25, vector, RRF weights, embeddings, reranker) are unchanged, so the Phase 2 strategy "
        "comparison (`retrieval_strategy_comparison.md`) still applies. Phase 3 changes which chunks reach the generator "
        "(multi-hop sub-queries), measured on the 5-chunk context:",
        "",
        md_table(["Context (5 chunks)", "Fact recall@5", "Page recall@5", "Precision@5", "MRR", "Questions missing a fact page"],
                 [[NAMES[k], cr[k]["recall"]["5"], cr[k]["page_recall"]["5"], cr[k]["precision"]["5"], cr[k]["mrr"],
                   ", ".join(cr[k]["questions_missing_a_fact_page"]) or "none"] for k in cr],
                 ["l", "r", "r", "r", "r", "l"]),
        "",
        "## 3. Category-level accuracy (measured)",
        "",
        md_table(["Category", "Phase 2", "Phase 2 pipeline (det.)", "Phase 3", "Δ vs Phase 2"],
                 [[c, _m(p2["by_category"][c]["accuracy"]), _m(bl["by_category"][c]["accuracy"]), _m(p3["by_category"][c]["accuracy"]),
                   _delta(p2["by_category"][c]["accuracy"], p3["by_category"][c]["accuracy"])] for c in cats]),
        "",
        "![Accuracy by category](charts/phase3_accuracy_by_category.svg)",
        "",
        "## 4. Per-question changes (measured labels; majority across runs)",
        "",
        f"Versus Phase 2: {data['question_changes_vs_phase2']}. Versus the deterministic Phase 2 pipeline: {data['question_changes_vs_baseline']}.",
        "",
        md_table(["ID", "Category", "Phase 2 (runs)", "Phase 2 pipeline (det.)", "Phase 3 (runs)", "vs Phase 2", "vs det. baseline"],
                 [[r["id"], r["category"], " / ".join(r["labels"]["phase2"]), " / ".join(r["labels"]["baseline"]),
                   " / ".join(r["labels"]["phase3"]), r["vs_phase2"], r["vs_baseline"]] for r in data["per_question"]],
                 ["l", "l", "l", "l", "l", "l", "l"]),
        "",
        "## 5. Numerical accuracy by figure type (measured; numeric facts fully matched)",
        "",
        md_table(["Figure type", "Phase 2", "Phase 2 pipeline (det.)", "Phase 3"],
                 [[t, _m(p2["numeric_by_figure_type"].get(t)), _m(bl["numeric_by_figure_type"].get(t)), _m(p3["numeric_by_figure_type"].get(t))]
                  for t in figure_types]),
        "",
        "## 6. Citation validation (measured)",
        "",
        "False positive = a wrong or partial answer the validator marks `SUPPORTED`; false negative = a correct answer it does not.",
        "",
        md_table(["Answers / validator", "Wrong or partial", "False positives", "FP rate", "Correct", "False negatives", "FN rate"],
                 [[NAMES[k] + " (as recorded)", v["recorded"]["wrong_or_partial_answers"], v["recorded"]["false_positives"],
                   fmt(v["recorded"]["false_positive_rate"]), v["recorded"]["correct_answers"], v["recorded"]["false_negatives"],
                   fmt(v["recorded"]["false_negative_rate"])] for k, v in val.items() if k in NAMES]
                 + ([["Phase 2 answers, Phase 3 validator", val["phase2_answers_with_phase3_validator"]["wrong_or_partial_answers"],
                      val["phase2_answers_with_phase3_validator"]["false_positives"], fmt(val["phase2_answers_with_phase3_validator"]["false_positive_rate"]),
                      val["phase2_answers_with_phase3_validator"]["correct_answers"], val["phase2_answers_with_phase3_validator"]["false_negatives"],
                      fmt(val["phase2_answers_with_phase3_validator"]["false_negative_rate"])]]
                    if "phase2_answers_with_phase3_validator" in val else []),
                 ["l", "r", "r", "r", "r", "r", "r"]),
        "",
        "The last row holds the answers fixed and swaps only the validator, which isolates the Task 3 change. The "
        "deterministic baseline's statuses were recorded with the validator version current when it ran.",
        "",
        "## 7. Abstention (measured)",
        "",
        f"Unanswerable questions: correct abstention Phase 2 {_m(p2['correct_abstention'])}, Phase 3 {_m(p3['correct_abstention'])}; "
        f"false answers Phase 2 {_m(p2['false_answer_unanswerable'])}, Phase 3 {_m(p3['false_answer_unanswerable'])}. "
        f"False abstention on answerable questions: Phase 2 {_m(p2['false_abstention'])}, Phase 3 {_m(p3['false_abstention'])}.",
        "",
        "Answers Phase 3 discarded because every claim contradicted the evidence (graded as if they had been kept):",
        "",
        md_table(["Run", "ID", "Discarded answer graded as", "Reason (first claim)", "Discarded answer"],
                 [[r["run"], r["id"], r["rejected_answer_label"], (r["reasons"] or [""])[0][:110], r["rejected_answer"][:140]]
                  for r in data["rejected_answers_phase3"]], ["r", "l", "l", "l", "l"])
        if data["rejected_answers_phase3"] else "None.",
        "",
        "Answers regenerated once after the validator found a contradiction (Task 5 revision step):",
        "",
        md_table(["Run", "ID", "First answer graded", "Final graded", "Reason", "First answer → final answer"],
                 [[r["run"], r["id"], r["original_label"], r["final_label"], (r["reasons"] or [""])[0][:90],
                   f"{r['original'][:90]} → {r['final'][:90]}"] for r in data["revisions_phase3"]],
                 ["r", "l", "l", "l", "l", "l"])
        if data["revisions_phase3"] else "None.",
        "",
        "## 8. Generation stability (measured)",
        "",
        f"Questions with the same label in every run: Phase 2 {p2['stable_questions']}/22 (sampled, 3 runs); "
        f"Phase 3 {p3['stable_questions']}/22 ({p3['runs']} runs at the recorded settings).",
        "",
        "## 9. Findings, limitations and next steps (qualitative)",
        "",
        findings or "_No qualitative notes recorded._",
    ]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--benchmark", type=Path, default=EVAL / "benchmark_v2.json")
    parser.add_argument("--pdf", type=Path, default=PROJECT_ROOT / "data" / "documents" / "annual_report.pdf")
    parser.add_argument("--findings", type=Path, default=EVAL / "phase3_findings.md")
    parser.add_argument("--output", type=Path, default=EVAL / "phase3_evaluation_report.md")
    args = parser.parse_args(argv)
    data = build(args)
    out = RESULTS / "phase3" / "phase3_comparison.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    findings = args.findings.read_text(encoding="utf-8").split("\n", 2)[-1].strip() if args.findings.is_file() else ""
    write_text(args.output, render(data, findings))
    print(f"Wrote {args.output} and {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
