"""Phase 3 comparison helpers and the validator audit (synthetic data, no models)."""

import importlib.util
import sys
from pathlib import Path

from app.evaluation.validator_audit import audit

ROOT = Path(__file__).resolve().parents[2]
SRC = "r.pdf"


def _module():
    spec = importlib.util.spec_from_file_location("build_phase3_report", ROOT / "scripts" / "build_phase3_report.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_majority_prefers_better_label_on_ties():
    m = _module()
    assert m.majority(["correct", "incorrect", "correct"]) == "correct"
    assert m.majority(["incorrect", "partially_correct"]) == "partially_correct"


def test_context_retrieval_scores_generator_context():
    m = _module()
    bench = {"source_document": SRC, "questions": [
        {"id": "Q1", "category": "a", "question": "?", "answerable": True, "expected_pages": [5]},
        {"id": "Q2", "category": "a", "question": "?", "answerable": True, "expected_pages": [7]},
        {"id": "Q3", "category": "u", "question": "?", "answerable": False, "expected_pages": []},
    ]}
    runs = {"runs": [{"questions": [
        {"id": "Q1", "context": [{"source": SRC, "page": 5}]},
        {"id": "Q2", "context": [{"source": SRC, "page": 1}]},
        {"id": "Q3", "context": [{"source": SRC, "page": 9}]},
    ]}]}
    out = m.context_retrieval(bench, runs)
    assert out["recall"]["5"] == 0.5 and out["mrr"] == 0.5
    assert out["questions_missing_a_fact_page"] == ["Q2"]


def test_per_question_classification():
    m = _module()

    def variant(labels):
        return {"summary": {"v2": {"answerable_stability": [
            {"id": "Q1", "category": "a", "labels": labels[0]}, {"id": "Q2", "category": "b", "labels": labels[1]}]}}}

    rows = m.per_question({
        "phase2": variant([["incorrect", "incorrect"], ["correct", "correct"]]),
        "baseline": variant([["incorrect"], ["correct"]]),
        "phase3": variant([["correct", "correct"], ["abstained", "abstained"]]),
    })
    assert [(r["id"], r["vs_phase2"], r["vs_baseline"]) for r in rows] == [("Q1", "improved", "improved"), ("Q2", "regressed", "regressed")]


def test_validator_audit_counts_false_positives_and_negatives():
    runs = {"runs": [{"questions": [
        {"id": "Q1", "answer": "right", "citations": []},
        {"id": "Q2", "answer": "wrong", "citations": []},
        {"id": "Q3", "answer": "skip", "citations": []},
    ]}]}
    graded = [{"run": 1, "answerable": [
        {"id": "Q1", "label": "correct", "validation_status": "SUPPORTED"},
        {"id": "Q2", "label": "incorrect", "validation_status": "SUPPORTED"},
        {"id": "Q3", "label": "abstained", "validation_status": "ABSTAINED"},
    ]}]
    statuses = {"right": "UNSUPPORTED", "wrong": "SUPPORTED"}
    result = audit(runs, graded, lambda a, c, q: {"overall_status": statuses[a]}, {"Q1": "q1", "Q2": "q2", "Q3": "q3"})
    assert result["false_positives"] == 1 and result["false_negatives"] == 1
    assert result["false_positive_rate"] == 1.0 and result["false_negative_rate"] == 1.0
    assert len(result["rows"]) == 2


def test_phase3_1_per_question_compares_against_phase2_and_phase3():
    spec = importlib.util.spec_from_file_location("build_phase3_1_report", ROOT / "scripts" / "build_phase3_1_report.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    def variant(labels):
        return {"summary": {"v2": {"answerable_stability": [{"id": "Q1", "category": "a", "labels": labels}]}}}

    rows = module.per_question({
        "phase2": variant(["incorrect"]), "phase3": variant(["abstained", "abstained"]), "phase3_1": variant(["correct", "correct"]),
    })
    assert rows == [{"id": "Q1", "category": "a", "labels": {"phase2": ["incorrect"], "phase3": ["abstained", "abstained"],
                                                             "phase3_1": ["correct", "correct"]},
                     "majority": {"phase2": "incorrect", "phase3": "abstained", "phase3_1": "correct"},
                     "vs_phase2": "improved", "vs_phase3": "improved"}]
