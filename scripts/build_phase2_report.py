"""Phase 2 Task 7: build the consolidated evaluation report from saved results.

Usage (from the project root, after the other evaluation scripts have run):

    python scripts/build_phase2_report.py

Reads only machine-readable results in data/evaluation/results/ and the
benchmark files; it runs no models. Every number in the report comes from
those files. Qualitative analysis (failure-case discussion, recommendations)
is written by hand in data/evaluation/phase2_findings.md and embedded under
headings that mark it as qualitative.

Writes data/evaluation/phase2_evaluation_report.md and
data/evaluation/results/phase2_summary.json (the headline numbers, for
comparing future runs).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.evaluation.reporting import fmt, md_table, signed, write_text  # noqa: E402

EVAL_DIR = PROJECT_ROOT / "data" / "evaluation"
RESULTS = EVAL_DIR / "results"

REQUIRED = {
    "strategy": "strategy_comparison_summary.json",
    "reranker": "reranker_evaluation_summary.json",
    "answers": "answer_evaluation_summary.json",
    "numerical": "numerical_evaluation.json",
    "abstention": "abstention_evaluation.json",
    "task2": "retrieval_evaluation_summary.json",
}
LABELS = {
    "bm25": "BM25", "vector": "Vector", "hybrid": "Hybrid (RRF)",
    "bm25+rerank": "BM25 + rerank", "vector+rerank": "Vector + rerank", "hybrid+rerank": "Hybrid + rerank (production)",
    "bm25+ce": "BM25 + CE only", "vector+ce": "Vector + CE only", "hybrid+ce": "Hybrid + CE only",
}


def load_results(results_dir: Path) -> dict:
    data, missing = {}, []
    for key, name in REQUIRED.items():
        path = results_dir / name
        if not path.is_file():
            missing.append(name)
            continue
        data[key] = json.loads(path.read_text(encoding="utf-8"))
    if missing:
        raise SystemExit(f"Missing result files in {results_dir}: {', '.join(missing)}. Run the evaluation scripts first.")
    return data


def _mmm(entry) -> str:
    if not entry:
        return "—"
    if entry["min"] == entry["max"]:
        return fmt(entry["mean"])
    return f"{fmt(entry['mean'])} (range {fmt(entry['min'])}–{fmt(entry['max'])})"


def _mean(entry):
    return entry["mean"] if entry else None


def headline(data: dict) -> dict:
    v2 = data["strategy"]["tables"]["v2"]
    a = data["answers"]["v2"]["metrics"]["answerable"]
    u = data["answers"]["v2"]["metrics"]["unanswerable"]
    num = data["numerical"]["aggregate"]
    return {
        "retrieval": {s: {"mrr": v2[s]["mrr"], "recall@3": v2[s]["recall"]["3"], "recall@5": v2[s]["recall"]["5"],
                          "hit@1": v2[s]["hit_rate"]["1"]} for s in v2},
        "answer_accuracy": _mean(a["answer_accuracy"]),
        "correct_or_partial": _mean(a["correct_or_partial_rate"]),
        "citation_precision_micro": _mean(a["citation_precision_micro"]),
        "citation_fact_recall": _mean(a["citation_fact_recall"]),
        "supported_claim_rate": _mean(a["supported_claim_rate"]),
        "unsupported_claim_rate": _mean(a["unsupported_claim_rate"]),
        "numeric_fact_match_rate": _mean(num["overall"]["matched"]),
        "correct_abstention_rate": _mean(u["correct_abstention_rate"]),
        "false_answer_rate": _mean(u["false_answer_rate"]),
        "false_abstention_rate": _mean(a["false_abstention_rate"]),
        "answer_runs": data["answers"]["v2"]["num_runs"],
    }


def _before_after_mrr(name: str, table: dict, comparison: dict) -> str:
    before, after = (part.strip() for part in name.split("->"))
    return f"{fmt(table[before]['mrr'])} → {fmt(table[after]['mrr'])} ({signed(comparison['deltas']['mrr'])})"


def _findings_sections(path: Path) -> dict[str, str]:
    """Split the hand-written findings file into '## ' sections."""

    if not path.is_file():
        return {}
    sections, current, buf = {}, None, []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^## (.+)$", line)
        if match:
            if current:
                sections[current] = "\n".join(buf).strip()
            current, buf = match.group(1).strip(), []
        elif current:
            buf.append(line)
    if current:
        sections[current] = "\n".join(buf).strip()
    return sections


def render(data: dict, benchmarks: dict, findings: dict[str, str]) -> str:
    s, r, ans, num, ab = data["strategy"], data["reranker"], data["answers"], data["numerical"]["aggregate"], data["abstention"]
    k_values = s["k_values"]
    v2, v1 = s["tables"]["v2"], s["tables"]["v1"]
    a = ans["v2"]["metrics"]["answerable"]
    u = ans["v2"]["metrics"]["unanswerable"]
    a1 = ans["v1"]["metrics"]["answerable"]
    h = headline(data)
    base = ["bm25", "vector", "hybrid"]
    all_s = [x for x in LABELS if x in v2]

    def q(name):
        return findings.get(name, "_No qualitative notes recorded._")

    lines = [
        "# InsightRAG Phase 2: evaluation report",
        "",
        f"_Generated by `scripts/build_phase2_report.py` on {datetime.now(timezone.utc).isoformat(timespec='seconds')} "
        "from the JSON files in `data/evaluation/results/`. Sections marked **Qualitative** are hand-written "
        "(`data/evaluation/phase2_findings.md`); everything else is generated from measured results._",
        "",
        "**How to read this report.** *Measured* = produced by running the system and scoring it. *Calculated* = "
        "derived from measured values (differences, best-of, rates). *Qualitative* = human reading of the "
        "per-question outputs. *Limitations and assumptions* are listed in section 12.",
        "",
        "## 1. Executive summary",
        "",
        "### Measured headline numbers (benchmark v2)",
        "",
        md_table(["Measure", "Value"], [
            ["Retrieval MRR: hybrid / BM25 / vector (no rerank)", f"{fmt(v2['hybrid']['mrr'])} / {fmt(v2['bm25']['mrr'])} / {fmt(v2['vector']['mrr'])}"],
            ["Retrieval fact recall@5: hybrid / BM25 / vector", f"{fmt(v2['hybrid']['recall']['5'])} / {fmt(v2['bm25']['recall']['5'])} / {fmt(v2['vector']['recall']['5'])}"],
            ["Production retrieval (hybrid + rerank): MRR, recall@5", f"{fmt(v2['hybrid+rerank']['mrr'])}, {fmt(v2['hybrid+rerank']['recall']['5'])}"],
            [f"Answer accuracy (22 answerable, mean of {h['answer_runs']} runs)", _mmm(a["answer_accuracy"])],
            ["Correct or partially correct", _mmm(a["correct_or_partial_rate"])],
            ["Citation precision (micro)", _mmm(a["citation_precision_micro"])],
            ["Citation fact recall", _mmm(a["citation_fact_recall"])],
            ["Numeric facts fully matched (value, unit, period, entity)", _mmm(num["overall"]["matched"])],
            ["Correct abstention on unanswerable questions", _mmm(u["correct_abstention_rate"])],
            ["False answers on unanswerable questions", _mmm(u["false_answer_rate"])],
            ["False abstentions on answerable questions", _mmm(a["false_abstention_rate"])],
        ], ["l", "r"]),
        "",
        "### Qualitative summary",
        "",
        q("Executive summary"),
        "",
        "## 2. Project and evaluation methodology",
        "",
        "InsightRAG answers questions about one annual report with a local pipeline: pypdf extraction → "
        "400-word chunks (100 overlap) → BAAI/bge-small-en-v1.5 embeddings in Qdrant + BM25 → weighted RRF "
        "fusion (0.4 BM25 / 0.6 vector, k = 60) with one chunk per page → cross-encoder rerank "
        "(ms-marco-MiniLM-L-6-v2, blended 0.7 first-stage / 0.3 cross-encoder) → top 5 chunks to "
        "llama3.2:3b (Ollama) → evidence selection (3 citations) → claim-level citation validation and abstention.",
        "",
        md_table(["Stage evaluated", "Script", "Method"], [
            ["Retrieval metrics", "`scripts/evaluate_retrieval.py`", "Recall/Page recall/Precision@K and MRR over distinct (source, page) units"],
            ["Strategy comparison", "`scripts/compare_retrieval.py`", "BM25, vector, hybrid on one shared index and expanded query; each reduced to its top 20 distinct pages"],
            ["Reranker", "`scripts/compare_retrieval.py`", "Same pools before/after the production reranker and the cross-encoder alone"],
            ["Answers, citations, numbers, abstention", "`scripts/evaluate_answers.py`", f"Full production pipeline, {h['answer_runs']} runs per question; deterministic grading, no LLM judge"],
            ["This report", "`scripts/build_phase2_report.py`", "Reads the JSON results only"],
        ], ["l", "l", "l"]),
        "",
        "Every evaluation can be re-scored without models from the saved raw outputs "
        "(`results/retrieval_runs.json`, `results/answer_runs.json`).",
        "",
        "## 3. Benchmark",
        "",
        md_table(["Version", "File", "Questions", "Verbatim excerpts", "Role"], [
            ["v1", "`benchmark_v1.json`", benchmarks["v1"]["total_questions"], benchmarks["v1"]["excerpts"], "Original; immutable baseline"],
            ["v2", "`benchmark_v2.json`", benchmarks["v2"]["total_questions"], benchmarks["v2"]["excerpts"], "Primary; adds page 3 as alternative evidence for Q017/Q018"],
        ], ["l", "l", "r", "r", "l"]),
        "",
        "Categories: " + ", ".join(f"{k} {v}" for k, v in benchmarks["v2"]["category_counts"].items())
        + f". {s['counts']['answerable']} answerable, {s['counts']['unanswerable']} unanswerable. The v2 change was "
        "verified from the PDF's drawing instructions (`scripts/verify_chart_mapping.py`, see `benchmark_v2_changes.md`).",
        "",
        "## 4. Retrieval strategy comparison (measured)",
        "",
        f"K = {', '.join(map(str, k_values))} distinct pages; R = fact recall, PR = page recall, P = precision.",
        "",
        md_table(["Strategy", "R@1", "R@3", "R@5", "R@10", "PR@5", "P@3", "Hit@1", "Hit@3", "MRR"],
                 [[LABELS[x], v2[x]["recall"]["1"], v2[x]["recall"]["3"], v2[x]["recall"]["5"], v2[x]["recall"]["10"],
                   v2[x]["page_recall"]["5"], v2[x]["precision"]["3"], v2[x]["hit_rate"]["1"], v2[x]["hit_rate"]["3"], v2[x]["mrr"]]
                  for x in base]),
        "",
        "![Fact recall by strategy](charts/strategy_recall.svg)",
        "",
        "Calculated: questions with no relevant page in the top 3: "
        + "; ".join(f"{LABELS[x]}: {', '.join(s['failures_top3'][x]) or 'none'}" for x in base) + ".",
        "",
        f"Benchmark correction impact (calculated, MRR v1 → v2): "
        + "; ".join(f"{LABELS[x]} {fmt(v1[x]['mrr'])} → {fmt(v2[x]['mrr'])}" for x in all_s) + ".",
        "",
        f"Task 2 baseline check: {s['task2_baseline_check']}",
        "",
        "Full detail: `retrieval_strategy_comparison.md`.",
        "",
        "### Qualitative",
        "",
        q("Retrieval strategies"),
        "",
        "## 5. Reranker comparison (measured)",
        "",
        md_table(["Comparison", "MRR before → after", "Hit@1 before → after", "Hit@3 before → after", "R@3 Δ", "Improved / degraded / unchanged"],
                 [[name, _before_after_mrr(name, v2, c),
                   f"{fmt(c['hit_rate']['1']['before'])} → {fmt(c['hit_rate']['1']['after'])}",
                   f"{fmt(c['hit_rate']['3']['before'])} → {fmt(c['hit_rate']['3']['after'])}",
                   signed(c["deltas"]["recall"]["3"]),
                   f"{c['counts']['improved']} / {c['counts']['degraded']} / {c['counts']['unchanged']}"]
                  for name, c in r["v2"].items()], ["l", "r", "r", "r", "r", "r"]),
        "",
        "Questions where the production rerank moved relevant evidence down: "
        + "; ".join(f"{name}: {', '.join(c['questions_with_relevant_moved_down']) or 'none'}" for name, c in r["v2"].items() if "+rerank" in name) + ".",
        "",
        "![MRR before and after reranking](charts/reranker_mrr.svg)",
        "",
        "Full detail: `reranker_evaluation.md`.",
        "",
        "### Qualitative",
        "",
        q("Reranker"),
        "",
        "## 6. Answer correctness (measured)",
        "",
        md_table(["Measure", "Value"], [
            ["Correct", _mmm(a["answer_accuracy"])],
            ["Correct or partially correct", _mmm(a["correct_or_partial_rate"])],
            ["Mean completeness", _mmm(a["mean_completeness"])],
            ["False abstention", _mmm(a["false_abstention_rate"])],
            ["Answers with a number not in the generator's context (count per run)", _mmm(a["answers_with_numbers_not_in_context"])],
            ["Context fact recall (ceiling for the generator)", _mmm(a["context_fact_recall"])],
            ["Questions with the same label in every run", f"{ans['v2']['stable_questions']} of {len(ans['v2']['answerable_stability'])}"],
        ], ["l", "r"]),
        "",
        md_table(["ID", "Category", "Labels per run"],
                 [[x["id"], x["category"], " / ".join(x["labels"])] for x in ans["v2"]["answerable_stability"]], ["l", "l", "l"]),
        "",
        "![Answer accuracy by category](charts/answer_accuracy_by_category.svg)",
        "",
        "Full detail (answers, grading reasons): `answer_quality_report.md`.",
        "",
        "## 7. Citation quality (measured)",
        "",
        md_table(["Measure", "v2", "v1"], [
            ["Citation precision (micro)", _mmm(a["citation_precision_micro"]), _mmm(a1["citation_precision_micro"])],
            ["Citation precision (macro)", _mmm(a["citation_precision_macro"]), _mmm(a1["citation_precision_macro"])],
            ["Citation fact recall", _mmm(a["citation_fact_recall"]), _mmm(a1["citation_fact_recall"])],
            ["Matched numeric facts whose value appears in a cited passage", _mmm(a["matched_numeric_fact_citation_support"]), "—"],
            ["Answers with a number not in any cited passage (count per run)", _mmm(a["answers_with_numbers_not_in_citations"]), "—"],
            ["System validator: supported claim rate", _mmm(a["supported_claim_rate"]), "—"],
            ["System validator: unsupported claim rate", _mmm(a["unsupported_claim_rate"]), "—"],
            ["System validator: insufficient-evidence claim rate", _mmm(a["insufficient_evidence_claim_rate"]), "—"],
        ], ["l", "r", "r"]),
        "",
        "The *system validator* rows are the pipeline's own claim checks (part of the system under test); the "
        "other rows are independent deterministic checks against the benchmark.",
        "",
        "## 8. Numerical accuracy (measured)",
        "",
        md_table(["Dimension", "Rate"], [[d, _mmm(num["overall"][d])] for d in
                                         ("value_found", "unit_ok", "period_ok", "entity_ok", "metric_named", "matched", "rounded_share_of_matched")], ["l", "r"]),
        "",
        md_table(["Figure type", "Facts", "Matched"], [[k, v.get("n_facts"), _mmm(v.get("matched"))] for k, v in num["by_figure_type"].items()]),
        "",
        md_table(["Comparison / derived check", "Rate"], [
            ["Direction correct (comparative)", _mmm(num["comparisons"]["direction_ok"])],
            ["Absolute change correct", _mmm(num["comparisons"]["absolute_change_ok"])],
            ["Percent change correct", _mmm(num["comparisons"]["percent_change_ok"])],
            ["Derived result within tolerance", _mmm(num["calculations"]["result_ok"])],
        ], ["l", "r"]),
        "",
        "Full detail: `numerical_evaluation.md`.",
        "",
        "## 9. Abstention performance (measured)",
        "",
        md_table(["Measure", "Value"], [
            ["Unanswerable trials (questions × runs)", ab["unanswerable_trials"]],
            ["Correct abstention rate", fmt(ab["correct_abstention_rate"])],
            ["False answer rate", fmt(ab["false_answer_rate"])],
            ["Citations shown alongside a false answer", fmt(ab["citations_shown_rate"])],
            ["Trials with invented numbers", ab["trials_with_invented_numbers"]],
            ["Trials with near-miss numbers copied from context", ab["trials_with_numbers_copied_from_context"]],
            ["False abstention rate on answerable question-runs", fmt(ab["answerable_false_abstention_rate"])],
        ], ["l", "r"]),
        "",
        "Full detail: `abstention_report.md`.",
        "",
        "## 10. Category-level analysis (measured)",
        "",
        md_table(["Category", "Retrieval MRR (hybrid+rerank)", "Retrieval R@5 (hybrid+rerank)", "Answer accuracy", "Completeness"],
                 [[c, s["by_category_v2"]["hybrid+rerank"][c]["mrr"], s["by_category_v2"]["hybrid+rerank"][c]["recall"]["5"],
                   _mmm(ans["v2"]["by_category"][c]["accuracy"]), _mmm(ans["v2"]["by_category"][c]["completeness"])]
                  for c in ans["v2"]["by_category"]]),
        "",
        "### Qualitative",
        "",
        q("Category analysis"),
        "",
        "## 11. Failure-case analysis (qualitative)",
        "",
        q("Failure cases"),
        "",
        "## 12. Limitations and assumptions",
        "",
        q("Limitations"),
        "",
        "## 13. Recommendations (qualitative)",
        "",
        q("Recommendations"),
        "",
        "## 14. Conclusions",
        "",
        q("Conclusions"),
        "",
        "## Appendix: result files",
        "",
        md_table(["File", "Content"], [
            ["`results/retrieval_evaluation*.json`", "Task 2 baseline (production retrieval, v1)"],
            ["`results/retrieval_runs.json`", "Raw rankings for all 9 retrieval variants × 25 questions"],
            ["`results/strategy_comparison*.json`", "Task 3 scores (v2 and v1)"],
            ["`results/reranker_evaluation*.json`", "Task 4 before/after comparisons"],
            ["`results/answer_runs.json`", "Raw pipeline outputs (answers, context, citations, validation) for every run"],
            ["`results/answer_evaluation*.json`", "Task 5 grading per run and aggregated"],
            ["`results/numerical_evaluation.json`", "Task 6 numeric checks per fact and dimension"],
            ["`results/abstention_evaluation.json`", "Task 6 abstention detail"],
            ["`results/phase2_summary.json`", "Headline numbers of this report"],
        ], ["l", "l"]),
    ]
    return "\n".join(lines)


def benchmark_info(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    excerpts = sum(len(q.get(k) or []) for q in data["questions"]
                   for k in ("supporting_evidence", "alternative_evidence", "distractor_evidence"))
    return {"total_questions": data["total_questions"], "excerpts": excerpts, "category_counts": data["category_counts"]}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--results-dir", type=Path, default=RESULTS)
    parser.add_argument("--findings", type=Path, default=EVAL_DIR / "phase2_findings.md")
    parser.add_argument("--output", type=Path, default=EVAL_DIR / "phase2_evaluation_report.md")
    args = parser.parse_args(argv)

    data = load_results(args.results_dir)
    benchmarks = {"v1": benchmark_info(EVAL_DIR / "benchmark_v1.json"), "v2": benchmark_info(EVAL_DIR / "benchmark_v2.json")}
    findings = _findings_sections(args.findings)
    write_text(args.output, render(data, benchmarks, findings))
    summary_path = args.results_dir / "phase2_summary.json"
    summary_path.write_text(json.dumps({
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "benchmark_primary": "benchmark_v2.json",
        **headline(data),
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output} and {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
