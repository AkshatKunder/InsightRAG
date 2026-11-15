"""Answer/citation grading, numerical breakdown, abstention summary and pipeline runner (fakes only)."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from app.evaluation.answer_evaluator import aggregate_runs, evaluate_answerable, evaluate_run, evaluate_unanswerable
from app.evaluation.answer_pipeline import PipelineComponents, run_benchmark, run_question
from app.evaluation.numerical_evaluator import abstention_summary, aggregate_numerical, figure_type, numerical_run_summary

ROOT = Path(__file__).resolve().parents[2]
SRC = "report.pdf"
SALES_TABLE = "(in $ millions) Sales for the year ended December 31 Total 61,352 62,441 (1,089) (1.7) %"

SALES_Q = {
    "id": "Q1", "category": "numerical", "question": "What were total sales in 2025?", "answerable": True,
    "expected_pages": [95],
    "expected_facts": [{"metric": "total_sales", "value": 61352, "unit": "USD million", "period": "2025", "pages": [95, 172]}],
}
CHANGE_Q = {
    "id": "Q2", "category": "comparative", "question": "How did total sales change between 2024 and 2025?", "answerable": True,
    "expected_pages": [95],
    "expected_facts": [
        {"metric": "total_sales", "value": 61352, "unit": "USD million", "period": "2025", "pages": [95]},
        {"metric": "total_sales", "value": 62441, "unit": "USD million", "period": "2024", "pages": [95]},
    ],
    "expected_comparison": {"direction": "decrease", "absolute_change": -1089, "unit": "USD million", "percent_change": -1.7},
}
NONE_Q = {"id": "Q3", "category": "unanswerable", "question": "Sales in 2030?", "answerable": False,
          "expected_pages": [], "distractor_pages": [95]}
BENCH = {"source_document": SRC, "questions": [SALES_Q, CHANGE_Q, NONE_Q]}


def record(answer, status="SUPPORTED", cited=(95,), context_pages=(95, 3), claims=None, text=SALES_TABLE):
    return {
        "answer": answer,
        "context": [{"source": SRC, "page": p, "text": text} for p in context_pages],
        "citations": [{"source": SRC, "page": p, "text": text} for p in cited],
        "validation": {"overall_status": status, "claims": claims if claims is not None else [{"status": status}]},
        "error": None,
    }


# ---------------------------------------------------------------- answerable grading


def test_correct_answer_with_relevant_citation():
    out = evaluate_answerable(SALES_Q, record("Total sales were $61,352 million in 2025."))
    assert out["label"] == "correct" and out["completeness"] == 1.0
    assert out["citations"]["precision"] == 1.0
    assert out["citations"]["fact_recall"] == 1.0
    assert out["citations"]["matched_numeric_facts_supported_by_citation"] == 1
    assert out["numbers_not_in_citations"] == [] and out["numbers_not_in_context"] == []


def test_wrong_value_is_incorrect_and_ungrounded():
    out = evaluate_answerable(SALES_Q, record("Total sales were $70,000 million in 2025."))
    assert out["label"] == "incorrect"
    assert out["numbers_not_in_context"] == ["$70,000 million"]


def test_irrelevant_citation_lowers_precision():
    out = evaluate_answerable(SALES_Q, record("Total sales were $61,352 million in 2025.", cited=(95, 204)))
    assert out["citations"]["precision"] == 0.5
    alt = evaluate_answerable(SALES_Q, record("Total sales were $61,352 million in 2025.", cited=(172,)))
    assert alt["citations"]["precision"] == 1.0  # alternative fact page counts


def test_comparative_rules():
    full = evaluate_answerable(CHANGE_Q, record("Sales decreased by $1,089 million (1.7%), from $62,441 million in 2024 to $61,352 million in 2025."))
    assert full["label"] == "correct" and full["completeness"] == 1.0
    change_only = evaluate_answerable(CHANGE_Q, record("Sales decreased by 1.7% in 2025 compared with 2024."))
    assert change_only["label"] == "correct" and change_only["completeness"] < 1.0
    wrong_direction = evaluate_answerable(CHANGE_Q, record("Sales increased from $62,441 million in 2024 to $61,352 million in 2025."))
    assert wrong_direction["label"] == "partially_correct"
    assert wrong_direction["comparison"]["direction_ok"] is False


def test_abstention_on_answerable_question():
    out = evaluate_answerable(SALES_Q, record("I could not find this information in the provided document.", status="ABSTAINED", cited=()))
    assert out["label"] == "abstained"


def test_pipeline_error_is_reported():
    out = evaluate_answerable(SALES_Q, {"answer": None, "error": "RuntimeError: ollama down"})
    assert out["label"] == "error"


# ---------------------------------------------------------------- unanswerable


def test_unanswerable_abstained():
    out = evaluate_unanswerable(NONE_Q, record("I could not find this information in the provided document.", status="ABSTAINED", cited=()))
    assert out["abstained"] and not out["false_answer"] and not out["citations_shown"]


def test_unanswerable_false_answer_with_copied_and_invented_numbers():
    copied = evaluate_unanswerable(NONE_Q, record("Sales in 2030 will be $61,352 million.", status="SUPPORTED"))
    assert copied["false_answer"] and copied["citations_shown"]
    assert copied["numbers_copied_from_context"] == ["$61,352 million"] and copied["invented_numbers"] == []
    assert copied["cited_distractor_pages"] == [95]
    invented = evaluate_unanswerable(NONE_Q, record("Sales in 2030 will be $80 billion.", status="UNSUPPORTED"))
    assert invented["invented_numbers"] == ["$80 billion"]


# ---------------------------------------------------------------- runs, aggregation, numerical


def run(index, sales_answer):
    return {"run": index, "questions": [
        {"id": "Q1", **record(sales_answer)},
        {"id": "Q2", **record("Sales decreased by 1.7% from $62,441 million in 2024 to $61,352 million in 2025.")},
        {"id": "Q3", **record("I could not find this information in the provided document.", status="ABSTAINED", cited=())},
    ]}


def test_evaluate_run_metrics_and_aggregation():
    r1 = evaluate_run(BENCH, run(1, "Total sales were $61,352 million in 2025."))
    r2 = evaluate_run(BENCH, run(2, "Total sales were $62,441 million in 2025."))
    m = r1["metrics"]
    assert m["answerable"]["answer_accuracy"] == 1.0
    assert m["unanswerable"]["correct_abstention_rate"] == 1.0
    assert m["answerable"]["labels"]["correct"] == 2

    agg = aggregate_runs([r1, r2])
    assert agg["metrics"]["answerable"]["answer_accuracy"] == {"mean": 0.75, "min": 0.5, "max": 1.0}
    assert agg["stable_questions"] == 1
    assert agg["answerable_stability"][0]["labels"] == ["correct", "incorrect"]


def test_missing_question_in_run_is_an_error_not_skipped():
    result = evaluate_run(BENCH, {"run": 1, "questions": []})
    assert [q["label"] for q in result["answerable"]] == ["error", "error"]
    assert result["metrics"]["answerable"]["error_rate"] == 1.0


def test_numerical_summary_dimensions():
    r = evaluate_run(BENCH, run(1, "Total sales were $61.4 billion in 2024."))  # wrong period
    summary = numerical_run_summary(r)
    facts = {(f["id"], f["period"]): f for f in summary["facts"]}
    assert facts[("Q1", "2025")]["value_found"] and facts[("Q1", "2025")]["unit_ok"]
    assert facts[("Q1", "2025")]["period_ok"] is False and not facts[("Q1", "2025")]["matched"]
    assert summary["comparisons"]["direction_ok"] == 1.0
    assert summary["by_figure_type"]["financial"]["n_facts"] == 3
    agg = aggregate_numerical([summary, summary])
    assert agg["overall"]["matched"]["mean"] == summary["overall"]["matched"]
    assert figure_type("steel_shipments", "million tonnes") == "shipments"
    assert figure_type("employees_fte", "FTE employees") == "employees"


def test_abstention_summary():
    r1 = evaluate_run(BENCH, run(1, "I could not find this information in the provided document."))
    r1["answerable"][0]["label"] = "abstained"
    summary = abstention_summary([r1])
    assert summary["correct_abstention_rate"] == 1.0 and summary["false_answer_rate"] == 0.0
    assert summary["answerable_false_abstentions"][0]["id"] == "Q1"


# ---------------------------------------------------------------- pipeline runner


def fake_components(answer="Total sales were $61,352 million in 2025.", fail_on=None):
    def generate(question, context):
        if fail_on and fail_on in question:
            raise RuntimeError("ollama unavailable")
        return answer

    return PipelineComponents(
        retrieve=lambda q: [{"source": SRC, "page": p, "text": SALES_TABLE, "score": 1.0} for p in (95, 3, 7)],
        rerank=lambda q, docs: [{**d, "reranker_score": 1.0, "final_score": 1.0} for d in docs[:2]],
        generate=generate,
        select_citations=lambda q, a, ctx: ctx[:1],
        validate=lambda a, c: {"overall_status": "SUPPORTED", "is_valid": True, "reason": "ok",
                               "claims": [{"claim_text": a, "status": "SUPPORTED", "source": SRC, "page": 95, "quantities": []}]},
    )


def test_run_question_records_everything():
    out = run_question("What were total sales in 2025?", fake_components())
    assert out["error"] is None
    assert [c["page"] for c in out["context"]] == [95, 3]
    assert out["citations"][0]["text"] == SALES_TABLE
    assert out["validation"]["overall_status"] == "SUPPORTED"
    json.dumps(out)


def test_run_benchmark_repeats_and_records_errors():
    runs = run_benchmark(BENCH, fake_components(fail_on="2030"), repeats=2)
    assert len(runs["runs"]) == 2 and len(runs["runs"][0]["questions"]) == 3
    assert runs["errors"] == [{"run": 1, "id": "Q3", "error": "RuntimeError: ollama unavailable"},
                              {"run": 2, "id": "Q3", "error": "RuntimeError: ollama unavailable"}]
    with pytest.raises(ValueError):
        run_benchmark(BENCH, fake_components(), repeats=0)


# ---------------------------------------------------------------- CLI


def _cli():
    spec = importlib.util.spec_from_file_location("evaluate_answers", ROOT / "scripts" / "evaluate_answers.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_cli_end_to_end_with_fakes(tmp_path):
    cli = _cli()
    bench = json.loads((ROOT / "data" / "evaluation" / "benchmark_v2.json").read_text(encoding="utf-8"))
    abstain = {q["question"] for q in bench["questions"] if not q["answerable"]}

    def factory(pdf):
        comps = fake_components()
        base_generate = comps.generate
        comps.generate = lambda q, ctx: "I could not find this information in the provided document." if q in abstain else base_generate(q, ctx)
        comps.validate = lambda a, c: {"overall_status": "ABSTAINED" if a.startswith("I could not") else "SUPPORTED", "claims": []}
        return comps, lambda: None

    out, reports = tmp_path / "results", tmp_path / "reports"
    code = cli.main(["--repeats", "2", "--output-dir", str(out), "--report-dir", str(reports)], components_factory=factory)

    assert code == 0
    summary = json.loads((out / "answer_evaluation_summary.json").read_text(encoding="utf-8"))
    assert summary["v2"]["num_runs"] == 2
    assert summary["v2"]["metrics"]["unanswerable"]["correct_abstention_rate"]["mean"] == 1.0
    # Only Q006 asks for total sales in 2025, so exactly one question is correct.
    assert summary["v2"]["metrics"]["answerable"]["answer_accuracy"]["mean"] == pytest.approx(1 / 22, abs=1e-4)
    for name in ("answer_runs.json", "answer_evaluation.json", "numerical_evaluation.json", "abstention_evaluation.json"):
        assert (out / name).is_file()
    for name in ("answer_quality_report.md", "numerical_evaluation.md", "abstention_report.md"):
        assert (reports / name).read_text(encoding="utf-8").startswith("#")

    # Re-grading saved runs needs no models and gives the same summary metrics.
    code = cli.main(["--runs", str(out / "answer_runs.json"), "--output-dir", str(tmp_path / "again"),
                     "--report-dir", str(tmp_path / "again_r")], components_factory=lambda pdf: pytest.fail("no models"))
    again = json.loads((tmp_path / "again" / "answer_evaluation_summary.json").read_text(encoding="utf-8"))
    assert code == 0 and again["v2"]["metrics"] == summary["v2"]["metrics"]
