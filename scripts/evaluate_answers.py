"""Phase 2 Tasks 5-6: answer, citation, numerical and abstention evaluation.

Usage (from the project root; needs the Qdrant index and Ollama with llama3.2:3b):

    python scripts/evaluate_answers.py                    # 3 runs of all 25 questions
    python scripts/evaluate_answers.py --repeats 1
    python scripts/evaluate_answers.py --runs data/evaluation/results/answer_runs.json   # re-grade, no models

The production pipeline (app/main.py settings) answers every question
``--repeats`` times; the generator is used unchanged, so repeats show
sampling variance. Answers are graded deterministically (no LLM judge)
against benchmark v2 (primary) and v1.

Outputs (results/ and data/evaluation/):

    results/answer_runs.json                 raw pipeline outputs (answers, context, citations, validation)
    results/answer_evaluation.json           per-run, per-question grading (v2)
    results/answer_evaluation_summary.json   aggregates over runs (v2 and v1)
    results/numerical_evaluation.json        per-dimension numeric checks
    results/abstention_evaluation.json       unanswerable questions and false abstentions
    answer_quality_report.md, numerical_evaluation.md, abstention_report.md
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.evaluation.answer_evaluator import aggregate_runs, evaluate_run  # noqa: E402
from app.evaluation.answer_pipeline import (  # noqa: E402
    CONTEXT_SIZE,
    MAX_CITATIONS,
    RETRIEVE_CANDIDATE_K,
    RETRIEVE_TOP_K,
    PipelineComponents,
    run_benchmark,
)
from app.evaluation.numerical_evaluator import abstention_summary, aggregate_numerical, numerical_run_summary  # noqa: E402
from app.evaluation.reporting import fmt, md_table, svg_grouped_bars, write_text  # noqa: E402
from app.evaluation.retrieval_evaluator import load_benchmark  # noqa: E402

log = logging.getLogger("evaluate_answers")

EVAL_DIR = PROJECT_ROOT / "data" / "evaluation"
DEFAULT_PDF = PROJECT_ROOT / "data" / "documents" / "annual_report.pdf"


def build_components(pdf_path: Path, pipeline: str = "phase3", options: dict | None = None) -> tuple[PipelineComponents, callable]:
    """The shared production pipeline (app/rag/pipeline.py), in its Phase 3 or Phase 2 (legacy) configuration."""

    from app.rag.generator import MODEL_NAME
    from app.rag.pipeline import PipelineConfig, build_pipeline
    from app.rag.reranker import MODEL_NAME as RERANKER_MODEL

    config = {"phase2": PipelineConfig.legacy, "phase3": PipelineConfig.phase3}.get(pipeline, PipelineConfig)()
    rag = build_pipeline(str(pdf_path), config, generation_options=options)
    unused = lambda *a, **k: None  # noqa: E731  (answer_question replaces the step callables)
    components = PipelineComponents(
        retrieve=unused, rerank=unused, generate=unused, select_citations=unused, validate=unused,
        answer_question=rag.answer,
        description={
            "pdf": pdf_path.name,
            "llm": MODEL_NAME,
            "llm_options": options or "production defaults (no temperature/seed override)",
            "reranker_model": RERANKER_MODEL,
            "pipeline_variant": pipeline,
            "pipeline_config": rag.settings()["pipeline"],
            "pipeline": f"app/rag/pipeline.py ({pipeline}): hybrid top {RETRIEVE_TOP_K} from {RETRIEVE_CANDIDATE_K} -> "
                        f"rerank top {CONTEXT_SIZE} -> generate -> select_citations({MAX_CITATIONS}) -> validate_citation",
        },
    )
    components.settings = rag.settings  # generation settings, read after the run
    return components, rag.retriever.vector_retriever.vector_store.close


def grade(benchmark: dict, runs: dict) -> dict:
    run_results = [evaluate_run(benchmark, run) for run in runs["runs"]]
    numerical = [numerical_run_summary(r) for r in run_results]
    return {
        "run_results": run_results,
        "aggregate": aggregate_runs(run_results),
        "numerical_runs": numerical,
        "numerical": aggregate_numerical(numerical),
        "abstention": abstention_summary(run_results),
    }


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


def _mmm(entry) -> str:
    if not entry:
        return "—"
    if entry["min"] == entry["max"]:
        return fmt(entry["mean"])
    return f"{fmt(entry['mean'])} ({fmt(entry['min'])}–{fmt(entry['max'])})"


def render_answer_report(meta: dict, graded: dict, graded_v1: dict, chart_dir: Path) -> str:
    agg = graded["aggregate"]
    a, u = agg["metrics"]["answerable"], agg["metrics"]["unanswerable"]
    a1 = graded_v1["aggregate"]["metrics"]["answerable"]
    runs = graded["run_results"]

    categories = list(agg["by_category"])
    write_text(chart_dir / "answer_accuracy_by_category.svg", svg_grouped_bars(
        f"Answer accuracy by category (mean of {agg['num_runs']} runs, benchmark v2)",
        categories,
        {
            "Correct": [agg["by_category"][c]["accuracy"]["mean"] for c in categories],
            "Correct or partial": [agg["by_category"][c]["correct_or_partial"]["mean"] for c in categories],
            "Completeness": [agg["by_category"][c]["completeness"]["mean"] for c in categories],
        },
        y_label="Share of questions",
    ))

    rows = [
        ["Answer accuracy (correct)", _mmm(a["answer_accuracy"])],
        ["Correct or partially correct", _mmm(a["correct_or_partial_rate"])],
        ["Mean completeness", _mmm(a["mean_completeness"])],
        ["False abstention rate (answerable)", _mmm(a["false_abstention_rate"])],
        ["Pipeline error rate", _mmm(a["error_rate"])],
        ["Citation precision (micro)", _mmm(a["citation_precision_micro"])],
        ["Citation precision (macro)", _mmm(a["citation_precision_macro"])],
        ["Citation fact recall", _mmm(a["citation_fact_recall"])],
        ["Answers with no citation (count per run)", _mmm(a["answers_without_citations"])],
        ["Matched numeric facts whose value is in a cited passage", _mmm(a["matched_numeric_fact_citation_support"])],
        ["Answers with a number not in any cited passage (count per run)", _mmm(a["answers_with_numbers_not_in_citations"])],
        ["Answers with a number not in the generator context (count per run)", _mmm(a["answers_with_numbers_not_in_context"])],
        ["Context fact recall (facts with a page in the 5 context chunks; all answerable)", _mmm(a["context_fact_recall"])],
        ["System validator: supported claim rate", _mmm(a["supported_claim_rate"])],
        ["System validator: unsupported claim rate", _mmm(a["unsupported_claim_rate"])],
        ["System validator: insufficient-evidence claim rate", _mmm(a["insufficient_evidence_claim_rate"])],
        ["Unanswerable: correct abstention rate", _mmm(u["correct_abstention_rate"])],
        ["Unanswerable: false answer rate", _mmm(u["false_answer_rate"])],
        ["Unanswerable: citations shown", _mmm(u["citations_shown_rate"])],
    ]
    lines = [
        "# Answer and citation quality evaluation",
        "",
        "_Generated by `scripts/evaluate_answers.py`. Do not edit by hand; re-run the script._",
        "",
        f"- Generated: {meta['timestamp']} (pipeline runs: {meta['runs_timestamp']})",
        f"- Pipeline: {meta['config'].get('pipeline')}",
        f"- LLM: {meta['config'].get('llm')} ({meta['config'].get('llm_options')}); {agg['num_runs']} runs of every question",
        f"- Benchmark: `{meta['benchmarks']['v2']}` (primary); v1 shown where it differs",
        f"- Pipeline errors: {len(meta['errors'])}" + (f" ({meta['errors']})" if meta["errors"] else ""),
        "- Grading is deterministic (`app/evaluation/answer_checks.py`), independent of the system's own validator. No LLM judge.",
        "",
        "## Summary (measured; mean over runs, range in brackets)",
        "",
        md_table(["Metric", "Value"], rows, ["l", "r"]),
        "",
        "Definitions: *correct* = every expected fact matched on value, unit, period and entity "
        "(comparative: right direction and both values or the change; derived: result within tolerance). "
        "*Citation precision* = cited pages that are relevant pages. *Citation fact recall* = facts with at least "
        "one cited page among the pages that state them. *System validator* rates come from the pipeline's "
        "own claim validation, reported for transparency: it is part of the system under test.",
        "",
        f"Benchmark v1 differs only in Q017/Q018 fact pages; v1 citation precision (micro) "
        f"{_mmm(a1['citation_precision_micro'])}, citation fact recall {_mmm(a1['citation_fact_recall'])}; "
        f"answer accuracy is unaffected ({_mmm(a1['answer_accuracy'])}).",
        "",
        "## By category",
        "",
        md_table(
            ["Category", "Accuracy", "Correct or partial", "Completeness"],
            [[c, _mmm(agg["by_category"][c]["accuracy"]), _mmm(agg["by_category"][c]["correct_or_partial"]),
              _mmm(agg["by_category"][c]["completeness"])] for c in categories],
        ),
        "",
        "![Answer accuracy by category](charts/answer_accuracy_by_category.svg)",
        "",
        "## Per question (labels per run)",
        "",
        f"Stable across runs: {agg['stable_questions']} of {len(agg['answerable_stability'])} answerable questions.",
        "",
    ]
    per_q_rows = []
    for index, stability in enumerate(agg["answerable_stability"]):
        first = runs[0]["answerable"][index]
        cit = first.get("citations") or {}
        per_q_rows.append([
            stability["id"], stability["category"], " / ".join(stability["labels"]),
            ", ".join(map(str, cit.get("cited_pages", []))) or "—",
            fmt(cit.get("precision")), first.get("validation_status") or "—",
            ", ".join(first.get("numbers_not_in_context", []) or []) or "",
        ])
    lines += [
        md_table(["ID", "Category", "Labels (run 1/2/…)", "Cited pages (run 1)", "Cit. precision (run 1)",
                  "Validator (run 1)", "Numbers not in context (run 1)"], per_q_rows,
                 ["l", "l", "l", "l", "r", "l", "l"]),
        "",
        "## Answers and grading detail (run 1)",
        "",
    ]
    for q in runs[0]["answerable"]:
        failed = [f"{c['metric']}: {c['reason']}" for c in q.get("fact_checks", []) if not c.get("matched")]
        extra = []
        if q.get("comparison"):
            comp = q["comparison"]
            extra.append(f"direction {comp['detected_direction']} (expected {comp['expected_direction']}), "
                         f"change stated: abs {fmt(comp['absolute_change_ok'])}, % {fmt(comp['percent_change_ok'])}")
        if q.get("calculation"):
            calc = q["calculation"]
            extra.append(f"derived result stated: {fmt(calc['result_ok'])}"
                         + (f", direction ok: {fmt(calc['direction_ok'])}" if calc.get("direction_ok") is not None else ""))
        lines.append(f"- **{q['id']}** ({q['label']}): {(q.get('answer') or '').strip()!r}")
        if failed or extra:
            lines.append(f"  - {'; '.join(failed + extra)}")
    lines += [
        "",
        "## Unanswerable questions",
        "",
        "See `abstention_report.md` for per-run detail.",
    ]
    return "\n".join(lines)


def render_numerical_report(meta: dict, graded: dict) -> str:
    num = graded["numerical"]
    dims = ("value_found", "unit_ok", "period_ok", "entity_ok", "metric_named", "matched")
    header = ["Group", "Facts"] + list(dims)

    def rows(group):
        return [[k, v.get("n_facts")] + [_mmm(v.get(d)) for d in dims] for k, v in num[group].items()]

    first = graded["numerical_runs"][0]
    fact_rows = [
        [f["id"], f["metric"] + (f" ({f['segment']})" if f.get("segment") else ""), f["period"] or "—",
         f"{f['expected_value']} {f['expected_unit']}", f.get("matched_text") or "—",
         fmt(f["matched"]), fmt(f.get("rounded")) if f["matched"] else "", "" if f["matched"] else f["reason"]]
        for f in first["facts"]
    ]
    return "\n".join([
        "# Numerical accuracy evaluation",
        "",
        "_Generated by `scripts/evaluate_answers.py`. Do not edit by hand; re-run the script._",
        "",
        f"- Generated: {meta['timestamp']}; {num['num_runs']} runs; benchmark `{meta['benchmarks']['v2']}`",
        "- Every numeric expected fact of every answered question is checked on separate dimensions "
        "(`app/evaluation/numerical_evaluator.py`). Values must be rounding-consistent and have at least two "
        "significant figures; units must be in the same class; the year paired with the value must be the "
        "fact's year; segment facts must name the segment.",
        "",
        "## Overall (measured; mean over runs, range in brackets)",
        "",
        md_table(["Dimension", "Rate"], [[d, _mmm(num["overall"][d])] for d in dims + ("rounded_share_of_matched",)], ["l", "r"]),
        "",
        f"Accuracy of numeric questions (numerical, financial, comparative, multi-hop; label = correct): "
        f"{_mmm(num['numeric_question_accuracy'])}.",
        "",
        "## By figure type",
        "",
        md_table(header, rows("by_figure_type")),
        "",
        "## By question category",
        "",
        md_table(header, rows("by_category")),
        "",
        "## Comparisons and derived values",
        "",
        md_table(
            ["Check", "Rate"],
            [["Comparison direction correct", _mmm(num["comparisons"]["direction_ok"])],
             ["Absolute change stated correctly", _mmm(num["comparisons"]["absolute_change_ok"])],
             ["Percent change stated correctly", _mmm(num["comparisons"]["percent_change_ok"])],
             ["Derived result within tolerance (Q021, Q022)", _mmm(num["calculations"]["result_ok"])],
             ["Derived direction correct (Q022)", _mmm(num["calculations"]["direction_ok"])]],
            ["l", "r"],
        ),
        "",
        "## Failure reasons (all runs)",
        "",
        md_table(["Reason", "Count"], sorted(num["failure_reasons_total"].items(), key=lambda kv: -kv[1]), ["l", "r"]),
        "",
        "## Fact-level detail (run 1)",
        "",
        md_table(["ID", "Metric", "Period", "Expected", "Matched text", "Matched", "Rounded", "Failure reason"],
                 fact_rows, ["l", "l", "l", "l", "l", "l", "l", "l"]),
    ])


def render_abstention_report(meta: dict, graded: dict) -> str:
    ab = graded["abstention"]
    rows = []
    for qid, entries in ab["per_question"].items():
        for e in entries:
            rows.append([qid, e["run"], fmt(e["abstained"]), e["validation_status"] or "—", e["citations_selected"],
                         fmt(e["citations_shown"]), ", ".join(e["invented_numbers"]) or "",
                         ", ".join(e["numbers_copied_from_context"]) or "", (e["answer"] or "").strip()])
    false_abst = ab["answerable_false_abstentions"]
    return "\n".join([
        "# Abstention evaluation",
        "",
        "_Generated by `scripts/evaluate_answers.py`. Do not edit by hand; re-run the script._",
        "",
        f"- Generated: {meta['timestamp']}; benchmark `{meta['benchmarks']['v2']}`",
        "- Abstained = the pipeline's validation status is `ABSTAINED` (the answer declines and states no figure). "
        "Citations are *shown* when the status is not `ABSTAINED` and citations were selected (as in `app/main.py`).",
        "",
        "## Unanswerable questions (measured)",
        "",
        md_table(["Metric", "Value"], [
            ["Trials (questions × runs)", ab["unanswerable_trials"]],
            ["Correct abstention rate", fmt(ab["correct_abstention_rate"])],
            ["False answer rate", fmt(ab["false_answer_rate"])],
            ["Citations shown with a false answer", fmt(ab["citations_shown_rate"])],
            ["Trials with invented numbers (not in context)", ab["trials_with_invented_numbers"]],
            ["Trials with numbers copied from context (near-miss figures)", ab["trials_with_numbers_copied_from_context"]],
            ["Errors", ab["errors"]],
        ], ["l", "r"]),
        "",
        md_table(["ID", "Run", "Abstained", "Status", "Citations selected", "Citations shown", "Invented numbers",
                  "Numbers from context", "Answer"], rows, ["l", "r", "l", "l", "r", "l", "l", "l", "l"]),
        "",
        "## False abstentions on answerable questions",
        "",
        f"Rate: {fmt(ab['answerable_false_abstention_rate'])} of answerable question-runs.",
        "",
        md_table(["Run", "ID", "Category", "Answer"], [[f["run"], f["id"], f["category"], (f["answer"] or "").strip()] for f in false_abst], ["r", "l", "l", "l"])
        if false_abst else "None.",
    ])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--benchmark", type=Path, default=EVAL_DIR / "benchmark_v2.json")
    parser.add_argument("--baseline-benchmark", type=Path, default=EVAL_DIR / "benchmark_v1.json")
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=EVAL_DIR / "results")
    parser.add_argument("--report-dir", type=Path, default=EVAL_DIR)
    parser.add_argument("--runs", type=Path, help="Re-grade an existing answer_runs.json instead of running the pipeline")
    parser.add_argument("--pipeline", choices=("phase3_1", "phase3", "phase2"), default="phase3_1",
                        help="phase3_1 = current pipeline; phase3 = Phase 3 configuration; phase2 = legacy configuration")
    parser.add_argument("--temperature", type=float, help="Ollama temperature (default: production setting)")
    parser.add_argument("--seed", type=int, help="Ollama seed (default: production setting)")
    args = parser.parse_args(argv)
    if args.repeats < 1:
        parser.error("--repeats must be >= 1")
    args.options = {k: v for k, v in (("temperature", args.temperature), ("seed", args.seed)) if v is not None} or None
    return args


def _dump(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main(argv=None, components_factory=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)
    primary, baseline = load_benchmark(args.benchmark), load_benchmark(args.baseline_benchmark)
    os.chdir(PROJECT_ROOT)
    if components_factory is None:
        components_factory = lambda pdf: build_components(pdf, args.pipeline, args.options)  # noqa: E731

    if args.runs:
        runs = json.loads(args.runs.read_text(encoding="utf-8"))
    else:
        components, close = components_factory(args.pdf)
        try:
            runs = run_benchmark(primary, components, args.repeats, progress=log.info)
        finally:
            close()
        runs["timestamp"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        if hasattr(components, "settings"):
            # Generation settings as actually applied (options_applied is False if Ollama rejected them).
            runs["config"]["generation"] = components.settings()["generation"]
        _dump(args.output_dir / "answer_runs.json", runs)

    graded = grade(primary, runs)
    graded_v1 = grade(baseline, runs)
    meta = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "runs_timestamp": runs.get("timestamp"),
        "benchmarks": {"v2": args.benchmark.name, "v1": args.baseline_benchmark.name},
        "config": runs["config"],
        "errors": runs.get("errors", []),
    }
    _dump(args.output_dir / "answer_evaluation.json", {**meta, "runs": graded["run_results"]})
    _dump(args.output_dir / "answer_evaluation_summary.json", {
        **meta,
        "v2": graded["aggregate"],
        "v1": {"metrics": graded_v1["aggregate"]["metrics"], "by_category": graded_v1["aggregate"]["by_category"]},
    })
    _dump(args.output_dir / "numerical_evaluation.json", {
        **meta, "aggregate": graded["numerical"],
        "runs": [{"run": r["run"], **s} for r, s in zip(graded["run_results"], graded["numerical_runs"])],
    })
    _dump(args.output_dir / "abstention_evaluation.json", {**meta, **graded["abstention"]})

    chart_dir = args.report_dir / "charts"
    write_text(args.report_dir / "answer_quality_report.md", render_answer_report(meta, graded, graded_v1, chart_dir))
    write_text(args.report_dir / "numerical_evaluation.md", render_numerical_report(meta, graded))
    write_text(args.report_dir / "abstention_report.md", render_abstention_report(meta, graded))

    a = graded["aggregate"]["metrics"]["answerable"]
    u = graded["aggregate"]["metrics"]["unanswerable"]
    print(f"Answer accuracy {_mmm(a['answer_accuracy'])}; correct or partial {_mmm(a['correct_or_partial_rate'])}; "
          f"citation precision {_mmm(a['citation_precision_micro'])}; correct abstention {_mmm(u['correct_abstention_rate'])}")
    if meta["errors"]:
        print(f"PIPELINE ERRORS: {meta['errors']}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
